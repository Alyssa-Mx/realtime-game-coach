#!/usr/bin/env python3
"""多权重共识筛：给定的每个 tag 都预测成同一个值、但与 GT 不符 —— 候选清单。
用法: python train/consensus_n.py <out_tag> <tag>[:tier] <tag>[:tier] ...
      tag 可以是 data/eval_v3/<tag>（分档，用 :S1 之类指定档，默认 S1）或 data/zeroshot/<tag>（单档）。
产物: data/zeroshot/<out_tag>/s0.jsonl（喂 export_compare.py）+ eval/<out_tag>.json
⚠ 共享上游的权重之间，"都这么读"不是独立证据（v1→v2→v3 是同一条热启动链、同一批训练场次）。
   唯一没跟这条链共享权重的是零样本基座 `v1`，但它本身在多数字段上很弱。一律只当候选。"""
import json, os, sys, glob, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks as T

out_tag, specs = sys.argv[1], sys.argv[2:]
def load(spec):
    tag, _, tier = spec.partition(':'); tier = tier or 'S1'
    d = {}
    for f in sorted(glob.glob(f'{T.R}/data/eval_v3/{tag}/s*.jsonl')):
        for l in open(f, encoding='utf-8'):
            r = json.loads(l)
            if r.get('kind') == 'field' and r['tier'] == tier: d[(r['match'], r['game_t'], r['task'])] = r
    if not d:
        for f in sorted(glob.glob(f'{T.R}/data/zeroshot/{tag}/s*.jsonl')):
            for l in open(f, encoding='utf-8'):
                r = json.loads(l); d[(r['match'], r['game_t'], r['task'])] = r
    if not d: sys.exit('空: ' + spec)
    return tag, d

srcs = [load(s) for s in specs]
keys = set.intersection(*[set(d) for _, d in srcs])
canon = lambda r: json.dumps(T.parse_json(r['pred_raw']), sort_keys=True, ensure_ascii=False)
hits, seen = [], collections.Counter()
SKIP = {'compass'}          # GT(刻度带拟合) 比模型准，不让人判
for k in keys:
    if k[2] in SKIP: continue
    seen[k[2]] += 1
    cs = {canon(d[k]) for _, d in srcs}
    if len(cs) != 1: continue
    p = json.loads(cs.pop())
    if p is None: continue
    tgt = srcs[0][1][k]['target']
    if T.score(k[2], p, tgt).get('acc', 0) >= 1: continue
    hits.append({'match': k[0], 'game_t': k[1], 'task': k[2], 'target': tgt, 'pred': p,
                 'pred_raw': json.dumps(p, sort_keys=True, ensure_ascii=False), 'imputed': srcs[0][1][k]['imputed']})
hits.sort(key=lambda h: (h['task'], h['game_t']))
d = f'{T.R}/data/zeroshot/{out_tag}'; os.makedirs(d, exist_ok=True)
with open(d + '/s0.jsonl', 'w', encoding='utf-8') as fh:
    for h in hits: fh.write(json.dumps({k: h[k] for k in ('match', 'game_t', 'task', 'target', 'pred_raw', 'imputed')}, ensure_ascii=False) + '\n')
json.dump(hits, open(f'{T.R}/eval/{out_tag}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
per = collections.Counter(h['task'] for h in hits)
print('# %s 全部一致但与 GT 不符：%d 条 / %d 个共有(帧,字段) = %.2f%%\n' % (' ∩ '.join(specs), len(hits), len(keys), 100 * len(hits) / max(len(keys), 1)))
print('| 字段 | 命中 | 样本 | 占比 |'); print('|---|---|---|---|')
for k, n in per.most_common(): print('| %s | %d | %d | %.1f%% |' % (k, n, seen[k], 100 * n / seen[k]))
