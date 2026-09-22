#!/usr/bin/env python3
"""把单帧 5.4s 拆开：prefill 多少、decode 多少、哪些字段拖长了批。

关键机制：批量 generate 的墙钟由**批内最长的那条**决定，不是平均。
所以 31 个字段混在一批里，全部要陪 team_msgs 这种长输出等到底。
测这几档（都不重训、不改精度）：
  A 现状      未 merge, BATCH=16, 原序
  B merge     merge_and_unload 后同上
  C 单批      merge + 一次 31 条
  D 长度分组  merge + 按实测输出长度分成"短批/长批"
  E prefill   merge + 分组 + max_new_tokens=1（只跑 prefill，用来隔离两段成本）
用法: python train/bench_latency_breakdown.py <frame.jpg> [match_id] [轮数=5]
"""
import json, os, sys, time, torch, cv2, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('WEAPON_ONE_PROMPT', '1')
import tasks as T
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

MODEL = os.environ.get('BASE_MODEL', 'Qwen/Qwen3-VL-8B-Instruct')
ADAPTER = os.environ.get('ADAPTER', 'data/sft/v7/final')
FIELDS = [k for k in T.GRID_TASKS if k != 'supplies'] + list(T.SUP_KEYS)
path = sys.argv[1]; match = sys.argv[2] if len(sys.argv) > 2 else None
R = int(sys.argv[3]) if len(sys.argv) > 3 else 5
f = cv2.imread(path); assert f is not None

proc = AutoProcessor.from_pretrained(MODEL); tok = proc.tokenizer
proc.tokenizer.padding_side = 'left'
base = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to('cuda:0')
peft = PeftModel.from_pretrained(base, ADAPTER if ADAPTER.startswith('/') else T.R + '/' + ADAPTER).eval()

def prompt(text):
    return proc.apply_chat_template([{'role': 'system', 'content': T.SYSTEM},
                                     {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}],
                                    tokenize=False, add_generation_prompt=True)

IMS = {k: T.crop(f, k, match, scale=T.SCALE_INFER.get(k, T.SCALE)) for k in FIELDS}

def run(model, order, bs, mnt=192):
    """按 order 的顺序切 bs 大小的批，返回 (总 token, 每批的最长输出)"""
    tot = 0; maxes = []
    for i in range(0, len(order), bs):
        ks = order[i:i + bs]
        enc = proc(text=[prompt(T.PROMPTS[k]) for k in ks], images=[IMS[k] for k in ks],
                   padding=True, return_tensors='pt').to('cuda:0')
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=mnt, do_sample=False)
        n = int(o.shape[1] - enc['input_ids'].shape[1]); tot += n; maxes.append(n)
    return tot, maxes

def bench(name, fn):
    ts = []
    for _ in range(R):
        torch.cuda.synchronize(); t0 = time.time(); r = fn(); torch.cuda.synchronize(); ts.append(time.time() - t0)
    ts.sort()
    print('%-30s %6.3f s   %s   (%s)' % (name, ts[len(ts) // 2], r, ' '.join('%.2f' % x for x in ts)), flush=True)
    return ts[len(ts) // 2]

# ---- 逐字段实际输出长度（一次一条，拿到真实长度，用于分组）----
print('=== 逐字段实际输出 token 数（单条，无 padding 干扰）===', flush=True)
LEN = {}
for k in FIELDS:
    enc = proc(text=[prompt(T.PROMPTS[k])], images=[IMS[k]], return_tensors='pt').to('cuda:0')
    with torch.no_grad():
        o = peft.generate(**enc, max_new_tokens=192, do_sample=False)
    LEN[k] = int(o.shape[1] - enc['input_ids'].shape[1])
for k, v in sorted(LEN.items(), key=lambda x: -x[1]):
    print('  %-16s %3d' % (k, v), end='\n' if list(sorted(LEN.items(), key=lambda x: -x[1])).index((k, v)) % 1 == 0 else '')
print('  合计 %d，最长 %d（%s），中位 %d' % (sum(LEN.values()), max(LEN.values()),
      max(LEN, key=LEN.get), sorted(LEN.values())[len(LEN) // 2]), flush=True)

print('\n=== 分档墙钟（%d 轮取中位）===' % R, flush=True)
a = bench('A 现状 未merge BATCH=16', lambda: run(peft, FIELDS, 16))
merged = PeftModel.from_pretrained(
    AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to('cuda:0'),
    ADAPTER if ADAPTER.startswith('/') else T.R + '/' + ADAPTER).merge_and_unload().eval()
b = bench('B merge  BATCH=16', lambda: run(merged, FIELDS, 16))
c = bench('C merge  单批31', lambda: run(merged, FIELDS, 31))
byl = sorted(FIELDS, key=lambda k: LEN[k])
LONG = [k for k in FIELDS if LEN[k] > 2 * sorted(LEN.values())[len(LEN) // 2]]
SHORT = [k for k in FIELDS if k not in LONG]
print('  长输出字段(%d): %s' % (len(LONG), ','.join(LONG)), flush=True)
d = bench('D merge  长度分组', lambda: (run(merged, SHORT, len(SHORT)), run(merged, LONG, max(1, len(LONG)))))
e = bench('E merge  分组 prefill only', lambda: (run(merged, SHORT, len(SHORT), 1), run(merged, LONG, max(1, len(LONG)), 1)))

print('\n| 档 | 秒 | 相对现状 |')
print('|---|---|---|')
for n, v in [('A 现状', a), ('B merge', b), ('C 单批31', c), ('D 长度分组', d), ('E 仅prefill', e)]:
    print('| %s | %.3f | %.2fx |' % (n, v, a / v))
print('\nprefill 占 D 的 %.0f%%，decode 占 %.0f%%' % (100 * e / d, 100 * (d - e) / d))
