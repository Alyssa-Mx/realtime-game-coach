#!/usr/bin/env python3
"""结构约束检查：与模型完全无关的一路信号。两个用途 ——
  1) 训练侧：把违反约束的 (帧,字段) 从训练目标里剔掉（已知错的监督，不如不给）
  2) 核查侧：生成候选清单，优先级高于任何模型共识（不共享权重、不共享选场）

约束（每条都在 2026-09-04 的核查里见过真实违例）：
  alive        一局内不增；违例用最长不增子序列求最小矛盾集（127/6939 命中，两条独立路径交叉确认过）
  game_t       严格递增
  zone_countdown 同一轮缩圈内每秒 -1；跨轮重置不算违例
  supplies     单格是分段常数：孤立单帧尖峰（前后帧相等而中间不同）判为读数抖动
  supplies 值域 投掷物 <=5、消耗品 <=10，超出即离群（2026-09-04 定的规则包第 4 条）
  ammo_reserve 位数突变且前后帧稳定 → 粘连拼接（同规则包第 3 条）
用法: python train/constraints.py [P1 P2 ...]        默认全部精标场；打印逐场逐约束命中数并写 eval/constraint_hits.json
"""
import bisect, json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks as T

SUP_KEYS = ['frag', 'smoke', 'medkit', 'firstaid', 'stun', 'molotov', 'adrenaline', 'painkiller', 'emp', 'drink', 'bandage']
# 上限不拍脑袋定：从全库取值分布反推。2026-09-04 第一版我按"投掷物<=5、消耗品<=10"写死，
# 结果 bandage=15（占 27% 的帧，完全正常）和 frag=6 被误报了 700+ 条 —— 又是把常识当判据。
# 现在的规则：只有"既大（>=10）又罕见（<=0.3%）"才算离群，正常的大数值（bandage 15、drink 10）不受影响。
def sup_outliers(all_rows):
    import collections as _c
    cnt = {k: _c.Counter() for k in SUP_KEYS}
    for r in all_rows:
        s = r.get('supplies') or {}
        for k in SUP_KEYS:
            v = s.get(k)
            if isinstance(v, int): cnt[k][v] += 1
    out = {}
    for k in SUP_KEYS:
        tot = sum(cnt[k].values()) or 1
        out[k] = {v for v, n in cnt[k].items() if v >= 10 and n / tot <= 0.003}
    return out

def _lnis_bad(vals):
    """返回"要动的最少帧"的下标集合：总长 - 最长不增子序列。只做检出，不给真值。"""
    tails, back, idxs = [], [], []
    for i in range(len(vals) - 1, -1, -1):          # 反过来求最长不减
        x = vals[i]; j = bisect.bisect_right(tails, x)
        if j == len(tails): tails.append(x); idxs.append(i)
        else: tails[j] = x; idxs[j] = i
        back.append(None)
    keep = set()
    tails, idxs = [], []
    for i in range(len(vals) - 1, -1, -1):
        x = vals[i]; j = bisect.bisect_right(tails, x)
        if j == len(tails): tails.append(x); idxs.append(i)
        else: tails[j] = x; idxs[j] = i
    keep = set(idxs)
    return set(range(len(vals))) - keep

def check(rows, outliers=None):
    """rows: 按 game_t 排序的 gamestate 行。返回 ({(game_t, field): 原因}, {字段: 场级警告})"""
    bad, warn = {}, {}
    outliers = outliers if outliers is not None else sup_outliers(rows)
    # --- alive 单调不增
    seq = [(r['game_t'], r.get('alive')) for r in rows if isinstance(r.get('alive'), int)]
    for i in _lnis_bad([v for _, v in seq]):
        bad[(seq[i][0], 'alive')] = 'alive 破坏单调不增'
    # --- game_t 严格递增
    prev = None
    for r in rows:
        g = r.get('game_t')
        if prev is not None and isinstance(g, int) and g <= prev: bad[(g, 'game_t')] = 'game_t 未递增'
        if isinstance(g, int): prev = g
    # --- zone_countdown 每秒 -1（只查同一轮内，跨轮重置跳过）
    def secs(v):
        if not isinstance(v, str) or ':' not in v: return None
        try: m, s = v.split(':'); return int(m) * 60 + int(s)
        except Exception: return None
    seq = [(r['game_t'], secs(r.get('zone_countdown'))) for r in rows]
    seq = [(g, v) for g, v in seq if v is not None]
    for (g0, v0), (g1, v1) in zip(seq, seq[1:]):
        dt, dv = g1 - g0, v0 - v1
        if 0 < dt <= 8 and v1 < v0 + 5 and abs(dv - dt) > 2:      # 允许 ±2s 抖动；倒计时重置(v1>v0)跳过
            bad[(g1, 'zone_countdown')] = 'zone_countdown 与 %ds 的时间差对不上(掉了 %ds)' % (dt, dv)
    # --- supplies 值域 + 孤立尖峰
    sup = [(r['game_t'], r.get('supplies')) for r in rows]
    for g, s in sup:
        if not s: continue
        for k in SUP_KEYS:
            v = s.get(k)
            if isinstance(v, int) and v in outliers.get(k, ()):
                bad[(g, 'supplies#' + k)] = '%s=%s 在全库里既大又罕见（<=0.3%%）' % (k, v)
    n_frames = max(1, len([1 for _, s in sup if s]))
    for k in SUP_KEYS:
        seq = [(g, (s or {}).get(k)) for g, s in sup if s]
        spikes = [(g1, a, b, c) for (g0, a), (g1, b), (g2, c) in zip(seq, seq[1:], seq[2:])
                  if a is not None and a == c and b != a and g2 - g0 <= 8]
        if len(spikes) / n_frames > 0.2:      # 一整场都在跳 = 这个字段在这场整体不可信，不逐帧报
            warn['supplies#' + k] = '这一场有 %d/%d 帧是孤立尖峰，整列存疑（别逐帧核，先看这一列怎么产出的）' % (len(spikes), n_frames)
            continue
        for g1, a, b, c in spikes:
            bad.setdefault((g1, 'supplies#' + k), '%s 孤立尖峰 %s→%s→%s' % (k, a, b, c))
    # --- ammo_reserve 值域：物理上不可能 >500（弹匣最大 ~40，备弹上限几百）
    # 2026-09-04 标注侧指出：原来只有"位数突变"这条，它按孤立尖峰设计，
    # 抓不到**连续污染** —— P8 的 7120 连着 69 帧、前后都一样，规则看它"很稳定"，一条没报。
    # 全库 144 帧，形态一致（7120×69、715×31、7150×9…），是"弹匣末位粘进备弹"+"1↔7 混淆"两种错叠加。
    for r in rows:
        v = r.get('ammo_reserve')
        if isinstance(v, int) and v > 500:
            bad[(r['game_t'], 'ammo_mag')] = 'ammo_reserve=%d 物理不可能（>500）' % v
    # --- ammo_reserve 位数突变（前后稳定）
    seq = [(r['game_t'], r.get('ammo_reserve')) for r in rows]
    seq = [(g, v) for g, v in seq if isinstance(v, int)]
    for (g0, a), (g1, b), (g2, c) in zip(seq, seq[1:], seq[2:]):
        if a == c and b != a and len(str(b)) != len(str(a)) and g2 - g0 <= 8:
            bad[(g1, 'ammo_mag')] = 'ammo_reserve 位数突变 %s→%s→%s（疑粘连）' % (a, b, c)
    return bad, warn

def load(match):
    return sorted((json.loads(l) for l in open(T.R + '/data/frames/%s/gamestate.jsonl' % match, encoding='utf-8')),
                  key=lambda r: r['game_t'])

if __name__ == '__main__':
    names = sys.argv[1:] or list(T.PRECISE)
    out, tot = {}, collections.Counter()
    for p in names:
        m = T.PRECISE[p]; rows = load(m); bad, warn = check(rows)
        out[p] = {'hits': [{'game_t': g, 'field': f, 'why': w} for (g, f), w in sorted(bad.items())], 'warn': warn}
        c = collections.Counter(f for _, f in bad)
        tot.update(c)
        print('%s %d 帧 → 命中 %d 条  %s' % (p, len(rows), len(bad), dict(c)))
        for f, w in warn.items(): print('    ⚠ %s %s' % (f, w))
    print('\n合计', dict(tot))
    json.dump(out, open(T.R + '/eval/constraint_hits.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('写入 eval/constraint_hits.json')
