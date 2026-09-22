"""汇总 eval_v3 输出 → 逐字段 × 档 / 帧级指标 / JSON 合法性 / 延迟 Pareto（TRAIN_PLAN_v3.md §八）。
用法: score_v3.py <tag_or_dir> [out.md]      跨档只比 21 个整框字段（S1 的物资单格不在此表）。"""
import json, os, sys, glob, collections, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks as T
d = sys.argv[1] if os.path.isdir(sys.argv[1]) else T.R + '/data/eval_v3/' + sys.argv[1]
FIELDS = list(T.GRID_TASKS)
CRITICAL = ['weapon_slot1', 'hp', 'ammo_mag', 'alive', 'zone_countdown', 'zone_dist_m', 'in_vehicle', 'stance']
acc = collections.defaultdict(list)                     # (tier, field) -> [acc]
frame = collections.defaultdict(dict)                   # (tier, match, game_t) -> field -> acc
blk = collections.defaultdict(lambda: collections.defaultdict(list))   # tier -> metric -> vals
for fn in sorted(glob.glob(d + '/s*.jsonl')):
    for l in open(fn, encoding='utf-8'):
        r = json.loads(l)
        if r['kind'] == 'block':
            b = blk[r['tier']]; b['json_ok'].append(r['json_ok']); b['missing'].append(len(r['missing'])); b['extra'].append(len(r['extra'])); b['n_tok'].append(r['n_tok']); b['nf'].append(len(r['fields']))
            continue
        if r['task'] not in FIELDS: continue
        s = T.score(r['task'], T.parse_json(r['pred_raw']), r['target'])
        a = s.get('acc', s.get('json_ok', 0.0)) if r['pred_raw'] is not None else 0.0
        acc[(r['tier'], r['task'])].append(a); frame[(r['tier'], r['match'], r['game_t'])][r['task']] = a
tiers = [t for t in ('S1', 'S2', 'S3', 'S4') if any(k[0] == t for k in acc)]
mean = lambda x: sum(x) / len(x) if x else float('nan')
L = ['# v3 配对评测 · %s' % os.path.basename(d.rstrip('/')), '', '## 逐字段 acc × 档（同一批帧）', '',
     '| 字段 | ' + ' | '.join(tiers) + ' | N |', '|---|' + '---|' * (len(tiers) + 1)]
for k in FIELDS:
    if not any((t, k) in acc for t in tiers): continue
    L.append('| %s | ' % k + ' | '.join('%.3f' % mean(acc[(t, k)]) for t in tiers) + ' | %d |' % len(acc[(tiers[0], k)]))
L.append('| **21 字段均值** | ' + ' | '.join('**%.3f**' % mean([mean(acc[(t, k)]) for k in FIELDS if (t, k) in acc]) for t in tiers) + ' | |')
L += ['', '## 帧级指标（一帧 = 该帧全部有目标字段拼成的 JSON）', '', '| 档 | 帧数 | 全对率 | 均错字段数 | 关键字段全对率 | 关键字段均错 |', '|---|---|---|---|---|---|']
for t in tiers:
    fr = [v for (tt, _, _), v in frame.items() if tt == t]
    if not fr: continue
    ex = mean([all(v.values()) for v in fr]); ne = mean([sum(1 - x for x in v.values()) for v in fr])
    cr = [{k: v[k] for k in CRITICAL if k in v} for v in fr]
    L.append('| %s | %d | %.3f | %.2f | %.3f | %.2f |' % (t, len(fr), ex, ne, mean([all(c.values()) for c in cr]), mean([sum(1 - x for x in c.values()) for c in cr])))
L += ['', '关键字段：' + ', '.join(CRITICAL), '', '## 块级 JSON 合法性', '', '| 档 | 块数 | 字段/块 | json_ok | 均缺键 | 均多键 | 均输出 token |', '|---|---|---|---|---|---|---|']
for t in tiers:
    b = blk[t]
    if b['json_ok']: L.append('| %s | %d | %.1f | %.3f | %.2f | %.2f | %.0f |' % (t, len(b['json_ok']), mean(b['nf']), mean(b['json_ok']), mean(b['missing']), mean(b['extra']), mean(b['n_tok'])))
lat = d + '/latency.jsonl'
if os.path.exists(lat):
    recs = [json.loads(l) for l in open(lat)]
    L += ['', '## 延迟（单卡 bf16 HF generate，逐帧墙钟中位数；"成批"=同帧块一次调用，"串行"=逐块调用）与 Pareto', '',
          '| 链路 | 块/帧 | 成批 s/帧 | 串行 s/帧 | 帧全对率 | 21 字段均值 |', '|---|---|---|---|---|---|']
    for t in tiers:
        rs = [r for r in recs if r['tier'] == t]
        if not rs: continue
        fr = [v for (tt, _, _), v in frame.items() if tt == t]
        L.append('| %s | %.1f | %.2f | %.2f | %.3f | %.3f |' % (t, mean([r['blocks'] for r in rs]), statistics.median([r['batched_s'] for r in rs]), statistics.median([r['sequential_s'] for r in rs]),
                 mean([all(v.values()) for v in fr]), mean([mean(acc[(t, k)]) for k in FIELDS if (t, k) in acc])))
    L.append('\n口径：这是**当前训练配额下的工程性能曲线**，不是各尺度的能力上限。')
md = '\n'.join(L); print(md)
if len(sys.argv) > 2:
    open(sys.argv[2], 'w', encoding='utf-8').write(md + '\n')
    summ = {t: {'mean21': mean([mean(acc[(t, k)]) for k in FIELDS if (t, k) in acc]),
                'frame_exact': mean([all(v.values()) for (tt, _, _), v in frame.items() if tt == t]),
                'json_ok': mean(blk[t]['json_ok']), 'n_frames': sum(1 for (tt, _, _) in frame if tt == t)} for t in tiers}
    json.dump(summ, open(sys.argv[2] + '.json', 'w'), indent=1)
