#!/usr/bin/env python3
"""同一个模型（v3，四档都训过），只改输入方式：整图 S4 vs 切图 S1，用 vLLM 量速度。

为什么必须用 v3：v7 只训过单字段裁剪，拿它跑整图会瞎输出，量出来的慢是"胡说八道的慢"，
证明不了架构。v3 是唯一四档配对训练过的版本，**换成整图它是会正经答题的**，
所以这个对比里唯一的变量就是"切不切"。

精度那一半已有现成数据（eval/v3_val_final.md，P7 215 帧配对）：
  S1 21字段均值 0.950 / 帧全对 0.298     S4 0.874 / 0.060

块的构造走 multiscale.make_item，和 v3 训练/评测用的是同一份代码，不另写。
**每轮换帧** —— vLLM 默认开前缀缓存，同帧重复会整段命中，量出来是假的。
用法: python train/bench_v3_tiers_vllm.py [match=P7] [轮数=10]
"""
import json, os, sys, time, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks as T, multiscale as M
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

MODEL = os.environ.get('MODEL', T.R + '/data/sft/v3/merged')
P = sys.argv[1] if len(sys.argv) > 1 else 'P7'
R = int(sys.argv[2]) if len(sys.argv) > 2 else 10
m = T.PRECISE[P]

proc = AutoProcessor.from_pretrained(MODEL)
llm = LLM(model=MODEL, dtype='bfloat16', gpu_memory_utilization=0.85,
          max_model_len=8192,                  # 整帧 2560x1440 = 3600 视觉 token，2048 装不下
          limit_mm_per_prompt={'image': 1}, max_num_seqs=64)
sp = SamplingParams(temperature=0.0, max_tokens=1200)   # 整帧要吐一大坨 JSON，给足

def text(p):
    return proc.apply_chat_template(
        [{'role': 'system', 'content': T.SYSTEM},
         {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': p}]}],
        tokenize=False, add_generation_prompt=True)

idx = {r['game_t']: r['frame'] for r in map(json.loads, open(T.R + '/data/frames/%s/index.jsonl' % m, encoding='utf-8'))}
rows = [json.loads(l) for l in open(T.R + '/data/frames/%s/gamestate.jsonl' % m, encoding='utf-8') if json.loads(l)['game_t'] in idx]
rows = rows[::max(1, len(rows) // (R + 2))][:R + 1]
frames = [cv2.imread(T.R + '/data/frames/%s/%s' % (m, idx[r['game_t']])) for r in rows]
print('%s：%d 帧（第 1 帧只用于预热）' % (P, len(rows)), flush=True)

def blocks(f, row, tier):
    seed = M.frame_hash(m, row['t_video'], 0) % (1 << 30)
    return [(im, pr) for im, pr, ans, meta in M.make_item(f, m, row, tier, seed, jitter=0)]

def run(f, row, tier):
    bl = blocks(f, row, tier)
    outs = llm.generate([{'prompt': text(p), 'multi_modal_data': {'image': im}} for im, p in bl], sp, use_tqdm=False)
    return len(bl), sum(len(o.outputs[0].token_ids) for o in outs)

for t in ('S1', 'S4'): run(frames[0], rows[0], t)        # 预热：建 CUDA graph
res = {}
for tier in ('S1', 'S4'):
    ts = []; nb = nt = 0
    for f, r in zip(frames[1:], rows[1:]):
        t0 = time.time(); nb, nt = run(f, r, tier); ts.append(time.time() - t0)
    ts.sort(); res[tier] = (ts[len(ts) // 2], nb, nt)
    print('%s  %.3f s/帧   %d 个请求   生成 %d token   (%s)'
          % (tier, ts[len(ts) // 2], nb, nt, ' '.join('%.2f' % x for x in ts[:6])), flush=True)

print('\n| 输入方式 | 秒/帧 | 请求数 | 生成 token | 21字段均值(P7,已有) | 帧全对率 |')
print('|---|---|---|---|---|---|')
print('| S1 切图 | %.3f | %d | %d | **0.950** | **0.298** |' % (res['S1'][0], res['S1'][1], res['S1'][2]))
print('| S4 整图 | %.3f | %d | %d | 0.874 | 0.060 |' % (res['S4'][0], res['S4'][1], res['S4'][2]))
print('\n整图比切图慢 %.2fx，且精度低 %.3f' % (res['S4'][0] / res['S1'][0], 0.950 - 0.874))
