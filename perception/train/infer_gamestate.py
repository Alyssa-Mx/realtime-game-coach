#!/usr/bin/env python3
"""用 v6 LoRA 从一帧画面直接产出 gamestate —— 给下游架构用的正式入口。

设计要点（都是踩出来的，不要改）：
  1. **每个字段一张裁剪图、一条 prompt**，不是整帧一次出 JSON。
     整帧档实测比裁剪档差：armor 0.386 vs 0.986、hp 0.833 vs 0.977（P7 215 帧），
     根因是 2560x1440 整帧 → 3600 个视觉 token，一个 token 覆盖 32x32 像素，
     而血条只有 4px 高、护甲耐久条 10x4px，特征被压进单个 token 里看不见。
  2. **裁剪口径必须和训练一致**：四周 +8px、最近邻放大 x3（物资单格是 +2px）。
     用 tasks.crop() 就对了，别自己写。
  3. **武器两个框用同一条中性 prompt**（WEAPON_ONE_PROMPT=1），不写"左框/右框/手持/收起/亮暗"。
     实测导向性描述会让模型靠"这个框常出现哪些枪"的先验答题，slot2 因此掉一半。
  4. **物资读单格**（sup_* 11 个任务），不要读整格 supplies。
     整格一次读 11 个数在稀有格上会系统性漏读（P7 上 0.44），单格是 0.995。

用法:
  python train/infer_gamestate.py <frame.jpg> [match_id]      # 单帧，打印 gamestate JSON
  python train/infer_gamestate.py --serve                     # 批量：从 stdin 逐行读图片路径
env: ADAPTER(默认 data/sft/v6/final)  FIELDS(逗号分隔，默认全部)  BATCH(默认 16)
"""
import json, os, sys, time, torch, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('WEAPON_ONE_PROMPT', '1')          # v5/v6 是用中性统一武器 prompt 训的
import tasks as T
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

MODEL = os.environ.get('BASE_MODEL', 'Qwen/Qwen3-VL-8B-Instruct')
ADAPTER = os.environ.get('ADAPTER', 'data/sft/v7/final')
BATCH = int(os.environ.get('BATCH', '32'))   # 一次装下全部 31 个字段。解码一步的耗时几乎与批大小无关（瓶颈是把 8B 权重读一遍），
                                            # 所以要最小化的是**串行步数**：单批 31 只要 35 步，拆两批是 18+35=53 步。
                                            # 2026-09-07 实测 BATCH=16 3.473s -> BATCH=32 3.351s。**不要按输出长度分组**，那会增加步数。
# 21 个整框字段 + 11 个物资单格；**不含** 整格 supplies（单格更准，见上）
FIELDS = os.environ.get('FIELDS', ','.join([k for k in T.GRID_TASKS if k != 'supplies'] + list(T.SUP_KEYS))).split(',')

_proc = _tok = _model = None
def load():
    global _proc, _tok, _model
    if _model is not None: return
    _proc = AutoProcessor.from_pretrained(MODEL); _tok = _proc.tokenizer
    _proc.tokenizer.padding_side = 'left'                 # 批量 generate 必须左 padding，否则切出来是垃圾
    m = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to('cuda:0')
    # merge_and_unload：把 LoRA 的 A/B 小矩阵直接加进原权重（W+BA），之后按普通模型跑。
    # 数学上完全等价、不掉任何精度，但每层少算一遍旁路 —— 7 模块 x 36 层 = 504 次额外小矩阵乘/前向，
    # 解码时这些启动开销是大头。2026-09-07 实测：5.285s -> 3.473s（1.52x），只在加载时多花几秒。
    _model = PeftModel.from_pretrained(m, ADAPTER if ADAPTER.startswith('/') else T.R + '/' + ADAPTER).merge_and_unload().eval()

def _prompt(text):
    msgs = [{'role': 'system', 'content': T.SYSTEM},
            {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}]
    return _proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def read_frame(frame_bgr, match=None, fields=FIELDS):
    """一帧 BGR ndarray → {字段: 解析后的 dict}。match 用于取该场次的 ROI 覆盖框（没有就用基准框）。"""
    load()
    # 逐字段放大倍数：FAST=1（默认）用 eval/scale_by_field.json 里实测过的低倍数，视觉 token 降 3.9 倍且不掉分
    fast = os.environ.get('FAST', '1') == '1'
    ims = [T.crop(frame_bgr, k, match, scale=(T.SCALE_INFER.get(k, T.SCALE) if fast else T.SCALE)) for k in fields]
    out = {}
    for i in range(0, len(fields), BATCH):
        ks, chunk = fields[i:i + BATCH], ims[i:i + BATCH]
        enc = _proc(text=[_prompt(T.PROMPTS[k]) for k in ks], images=chunk, padding=True, return_tensors='pt').to('cuda:0')
        with torch.no_grad():
            o = _model.generate(**enc, max_new_tokens=192, do_sample=False)
        for j, k in enumerate(ks):
            txt = _tok.decode(o[j][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            out[k] = T.parse_pred(k, txt) if hasattr(T, 'parse_pred') else T.parse_json(txt)
    return out

def to_gamestate(raw, game_t=None, t_video=None):
    """把逐字段结果拼成 gamestate 行 —— 键名和口径与 data/frames/<match>/gamestate.jsonl 一致。"""
    g = lambda k, d=None: (raw.get(k) or {}).get(k.split('_')[-1] if False else k, d)
    s = {}
    if game_t is not None: s['game_t'] = game_t
    if t_video is not None: s['t_video'] = t_video
    for k in ('alive', 'hp', 'signal', 'energy', 'scope', 'zone_countdown', 'zone_dist_m',
              'in_vehicle', 'stance', 'compass'):
        s[k] = (raw.get(k) or {}).get(k)
    if 'game_t' in raw and game_t is None: s['game_t'] = (raw['game_t'] or {}).get('game_t')
    am = raw.get('ammo_mag') or {}
    s['ammo_mag'], s['ammo_reserve'] = am.get('ammo_mag'), am.get('ammo_reserve')
    for k in ('helmet', 'armor'):                          # 模型出 {等级, dur_pct}；gamestate 存等级 + dur/full
        d = raw.get(k) or {}
        s[k] = d.get(k)
        s[k + '_dur_pct'] = d.get('dur_pct')                # 注意：*_dur/*_full 只在离线标注里有，实时侧没有
    s['supplies'] = {c: (raw.get('sup_' + c) or {}).get('count') for c in T.SUP_POS} if any('sup_' + c in raw for c in T.SUP_POS) else None
    for k in ('killfeed', 'team_msgs'): s[k] = (raw.get(k) or {}).get('lines')
    s['banner'] = raw.get('banner')                         # {raw, kind}
    tp = raw.get('team_panel') or {}
    s['team_panel'] = {'danger': tp.get('danger') or [], 'downed': tp.get('downed') or []}
    w1, w2 = (raw.get('weapon_slot1') or {}).get('weapon'), (raw.get('weapon_slot2') or {}).get('weapon')
    s['weapon_main'] = w1                                   # 左框=当前手持（开车时是"方向盘"）
    s['weapon_other'] = w2                                  # 右框=收起的另一把
    return s

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    t0 = time.time()
    path = args[0]; match = args[1] if len(args) > 1 else None
    f = cv2.imread(path)
    assert f is not None, '读不到图: ' + path
    raw = read_frame(f, match)
    print(json.dumps(to_gamestate(raw), ensure_ascii=False, indent=1))
    print('# %d 个字段，%.2f s' % (len(FIELDS), time.time() - t0), file=sys.stderr)
