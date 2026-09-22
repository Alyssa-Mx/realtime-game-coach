#!/usr/bin/env python3
"""vLLM 后端的逐字段 HUD 识别 —— 和 infer_gamestate.py 同口径，只换推理引擎。

为什么值得换（2026-09-07 实测的 HF 基线 3.35s/帧）：
  prefill 1.82s(54%)：HF 的 padding=True 把 31 条补到最长（4329 真实 token -> 9021 个位置），
                      **52% 的算力在算空位**。vLLM 不做 padding。
  decode  1.58s(46%)：35 步 x 45ms。8B bf16 权重 16GB / H20 带宽 4TB/s，理论下限 4ms/步，
                      差的 11 倍是 HF generate 的 Python 逐步调度。vLLM 用 CUDA graph。

用的是**合并后的普通模型**（train/merge_lora.py 产出），不走 vLLM 的 LoRA 通路 ——
多模态 + LoRA 容易有坑，合并后它就是个普通 Qwen3-VL。

用法:
  python train/infer_vllm.py gamestate <frame.jpg> [match]   # 单帧 -> gamestate JSON
  python train/infer_vllm.py bench [match=P8] [轮数=10]   # 量延迟（**每轮换帧**，否则前缀缓存会骗你）
  python train/infer_vllm.py eval  [match=P8] [帧数=0(全量)] [out.jsonl]   # 出预测供 score 用
env: MODEL(默认 data/sft/v7/merged)  GPU_UTIL(0.85)  MAXLEN(2048)
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('WEAPON_ONE_PROMPT', '1')          # v5 起用中性统一武器 prompt，和训练一致
import tasks as T
import cv2
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

MODEL = os.environ.get('MODEL', 'data/sft/v7/merged')
MODEL = MODEL if MODEL.startswith('/') else T.R + '/' + MODEL
FIELDS = [k for k in T.GRID_TASKS if k != 'supplies'] + list(T.SUP_KEYS)

_proc = AutoProcessor.from_pretrained(MODEL)
_llm = LLM(model=MODEL, dtype='bfloat16',
           gpu_memory_utilization=float(os.environ.get('GPU_UTIL', '0.85')),
           max_model_len=int(os.environ.get('MAXLEN', '2048')),
           limit_mm_per_prompt={'image': 1},
           max_num_seqs=64)                              # 31 条要一批装下
_sp = SamplingParams(temperature=0.0, max_tokens=192)     # 贪心，和 HF do_sample=False 对齐

def _text(prompt):
    return _proc.apply_chat_template(
        [{'role': 'system', 'content': T.SYSTEM},
         {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}],
        tokenize=False, add_generation_prompt=True)

_TXT = {k: _text(T.PROMPTS[k]) for k in FIELDS}           # prompt 文本不随帧变，建一次就够

def read_frame(frame_bgr, match=None, fields=FIELDS):
    """一帧 BGR -> {字段: 原始输出文本}。31 条一次交给 vLLM，由它自己调度。"""
    reqs = [{'prompt': _TXT[k],
             'multi_modal_data': {'image': T.crop(frame_bgr, k, match,
                                                  scale=T.SCALE_INFER.get(k, T.SCALE))}}
            for k in fields]
    outs = _llm.generate(reqs, _sp, use_tqdm=False)
    return {k: outs[i].outputs[0].text.strip() for i, k in enumerate(fields)}

def read_frame_parsed(frame_bgr, match=None, fields=FIELDS):
    """和 infer_gamestate.read_frame 同口径：返回解析后的 dict，不是原始文本。"""
    raw = read_frame(frame_bgr, match, fields)
    return {k: (T.parse_pred(k, v) if hasattr(T, 'parse_pred') else T.parse_json(v)) for k, v in raw.items()}

def to_gamestate(raw, game_t=None, t_video=None):
    """直接复用 infer_gamestate 的映射 —— gamestate 口径只能有一份，不要在这儿重写。"""
    import infer_gamestate as G
    return G.to_gamestate(raw, game_t, t_video)

def shutdown(code=0, exit_process=True):
    """必须显式关 EngineCore，否则退出会挂住 —— 而且挂住期间它被 init 收养、继续占 85% 显存。

    2026-09-07 因此泄了 3 张卡（249GB）。排查时踩的两个坑，都记在这儿：
      1. **按脚本名杀进程够不着它** —— 按"脚本名/模块名"匹配，而它的 cmdline 是字面的
         `VLLM::EngineCore`，不像脚本调用，会被过滤。
      2. **nvidia-smi 报的 PID 和当前进程命名空间里的 PID 可能不是一套编号**。
         我拿 `ps -p <nvidia-smi给的PID>` 查，得到"进程不存在"，据此错判成"显存泄漏、
         要重置卡"，于是一路换卡绕开 —— 每跑一次多泄一张。
      **正确的排查是** `ps -eo pid,ppid,cmd | grep VLLM`，然后按 PID `kill`。

    exit_process=False：**只关引擎，不动调用方进程** —— 常驻服务（如教练侧）必须用这个。
      教练侧 2026-09-07 指出：教练是常驻进程，换场景时不能被 os._exit 带走。
      注意关完之后本模块的 _llm 就不可用了，要再推理得重新起进程。
    exit_process=True（默认，命令行/批处理用）：关引擎后 os._exit，绕过可能挂住的退出流程。
    """
    try:
        _llm.llm_engine.engine_core.shutdown()
    except Exception as e:
        print('engine shutdown 失败: %r' % e, file=sys.stderr)
    if not exit_process:
        return                           # 常驻场景：引擎已关，进程继续活着
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)                       # 绕过可能挂住的 atexit / 非守护线程

if __name__ == '__main__':
    try:
        mode = sys.argv[1]
        if mode == 'gamestate':                       # 单帧 -> gamestate JSON，给下游架构用
            path = sys.argv[2]; match = sys.argv[3] if len(sys.argv) > 3 else None
            f = cv2.imread(path); assert f is not None, '读不到图: ' + path
            t0 = time.time()
            print(json.dumps(to_gamestate(read_frame_parsed(f, match)), ensure_ascii=False, indent=1))
            print('# %d 个字段，%.2f s' % (len(FIELDS), time.time() - t0), file=sys.stderr)
        elif mode == 'bench':
            # **每轮必须换帧**：vLLM 默认开前缀缓存，同一张图重复问会整段命中 KV 缓存，
            # 量出来是"重复问同一个问题"的速度，不是线上速度。2026-09-07 我第一版就是这么量错的
            # （同帧 5 轮得 0.302s，换帧后才是真数）。
            P = sys.argv[2] if len(sys.argv) > 2 else 'P8'
            R = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            match = T.PRECISE[P]
            idx = {r['game_t']: r['frame'] for r in map(json.loads, open(T.R + '/data/frames/%s/index.jsonl' % match, encoding='utf-8'))}
            gts = sorted(idx)[::max(1, len(idx) // (R + 2))][:R + 1]
            fs = [cv2.imread(T.R + '/data/frames/%s/%s' % (match, idx[g])) for g in gts]
            read_frame(fs[0], match)                          # 预热：首次要建 CUDA graph（这一帧之后不再用）
            ts = []
            for f in fs[1:]:
                t0 = time.time(); r = read_frame(f, match); ts.append(time.time() - t0)
            ts.sort()
            print('vLLM 单帧 %d 字段：%.3f s（%d 轮取中位，%s）'
                  % (len(FIELDS), ts[len(ts) // 2], R, ' '.join('%.2f' % x for x in ts)), flush=True)
            print('  样例 hp=%r armor=%r banner=%r' % (r.get('hp'), r.get('armor'), r.get('banner')[:60]), flush=True)
        elif mode == 'eval':
            P = sys.argv[2] if len(sys.argv) > 2 else 'P8'
            N = int(sys.argv[3]) if len(sys.argv) > 3 else 0
            out = sys.argv[4] if len(sys.argv) > 4 else T.R + '/data/eval_v3/v7_vllm/s0.jsonl'
            m = T.PRECISE[P]
            idx = {r['game_t']: r['frame'] for r in map(json.loads, open(T.R + '/data/frames/%s/index.jsonl' % m, encoding='utf-8'))}
            rows = [json.loads(l) for l in open(T.R + '/data/frames/%s/gamestate.jsonl' % m, encoding='utf-8') if json.loads(l)['game_t'] in idx]
            if N: rows = rows[::max(1, len(rows) // N)][:N]
            os.makedirs(os.path.dirname(out), exist_ok=True)
            w = open(out, 'w', encoding='utf-8'); t0 = time.time()
            for i, r in enumerate(rows):
                f = cv2.imread(T.R + '/data/frames/%s/%s' % (m, idx[r['game_t']]))
                d = read_frame(f, m)
                for k in FIELDS:
                    tg = T.target(r, k)
                    if tg is None: continue          # 无真值的字段不写（与 eval_v3.py 一致，否则打分时 target=None 会炸）
                    # 与 eval_v3.py 的 field 行同格式，好让 score_v3.py 直接吃
                    w.write(json.dumps({'kind': 'field', 'tier': 'S1', 'match': m, 'game_t': r['game_t'],
                                        'task': k, 'target': (tg[0] if tg else None),
                                        'imputed': bool(tg[1]) if tg else False,
                                        'pred_raw': d[k], 'block': i}, ensure_ascii=False) + '\n')
                if i % 50 == 0: print('%d/%d  %.1f s' % (i, len(rows), time.time() - t0), flush=True)
            w.close()
            print('DONE %d 帧 %.1f s（%.3f s/帧）' % (len(rows), time.time() - t0, (time.time() - t0) / len(rows)), flush=True)
    except BaseException:
        import traceback; traceback.print_exc()
        shutdown(1)          # 异常路径也必须关引擎，否则 EngineCore 被 init 收养、继续占 85% 显存
    shutdown()
