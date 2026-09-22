#!/usr/bin/env python3
"""v7 计划生成器：**按预算分配**，不再用"频次开方反比"。

为什么改（2026-09-05）：
  v4/v5 的权重是 1/sqrt(频次)，它惩罚的是"每帧都有的字段" —— 而 armor 恰恰每帧都有且很难，
  于是被判了最低权重，只拿到 3047 块（v3 多尺度时代有 11900 块），六个评测点纹丝不动。
  频次 ≠ 难度，这是 v4/v5 权重公式的根本错误。

现在的做法：
  1) 目标份额 ∝ max(1 − v5准确率, 0.03)，加 1.8% 地板（防遗忘）和 12% 上限（防一个字段吃掉五分之一）
  2) 粗标 54 场除武器外**新增盔/甲**，只取两路机器读数一致的帧；
     helmet 只取破损帧（它已 0.979 饱和，满耐久的不浪费预算），
     armor 破损全取、满耐久按 1/2 配比（只喂破损会让模型学成"永不报满耐久"）
  3) 抽字段时按**剩余需求**加权 —— 粗标已经给的会自动从精标那边扣掉，
     两个机制不重复加码（2026-09-05 发现的问题）

产物 data/sft/v6/{plan.jsonl, val.jsonl, pool.json, exposure.md}
env: SEED(0) STEPS(2400) K(8) COARSE_FRAC(0.35) VAL(P8,P4) VAL_FRAMES(300)
"""
import json, os, sys, random, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks as T, constraints as C

SEED = int(os.environ.get('SEED', '0')); STEPS = int(os.environ.get('STEPS', '4000'))
K = int(os.environ.get('K', '8')); COARSE_FRAC = float(os.environ.get('COARSE_FRAC', '0.35'))
VAL = [v for v in os.environ.get('VAL', 'P8,P4').split(',') if v]
VAL_FRAMES = int(os.environ.get('VAL_FRAMES', '300'))
WORLD, CF = 8, 2
OUT = T.R + '/data/sft/v7'; os.makedirs(OUT, exist_ok=True)
rng = random.Random(SEED)

# ---------- v7 目标份额：直接给定，不再从准确率反推 ----------
# 依据 v6 在 P8 全量（唯一三版都没训过的留出场，标签已精修）的结果 + 一条今天验证出来的判据：
#   多版一致率高（>=90%）的字段 = 标签问题，加权无效（ammo_mag/compass/stance 三版数字完全一样，
#   而它们的"错"修完标签后从 0.824 跳到 0.990）→ 退回地板
#   多版一致率低（<30%）= 模型问题，加权有效（armor 0.52→0.736）→ 保持高权重
# weapon_slot1：v6 把它从 16138 块砍到 5165，P8 上 0.959→0.777，砍过头了，补回 8%。
SHARE = {'armor': .12, 'weapon_slot2': .12, 'weapon_slot1': .08, 'team_panel': .05, 'energy': .03}
FLOOR_SHARE = .02          # 其余字段一律地板，防遗忘
w = {k: SHARE.get(k, FLOOR_SHARE) for k in T.TASKS}
s = sum(w.values()); W = {k: v / s for k, v in w.items()}

FINE = [p for p in T.PRECISE if p not in VAL]
print('精标训练场:', FINE, '  留出:', VAL)

# ---------- 结构约束：违规的 (帧,字段) 不进训练 ----------
banned = collections.defaultdict(set)
for p in T.PRECISE:
    m = T.PRECISE[p]; bad, _ = C.check(C.load(m))
    for (g, f), _ in bad.items():
        base = f.split('#')[0]
        banned[m].add((g, 'sup_' + f.split('#')[1] if '#' in f else base))
        if '#' in f: banned[m].add((g, 'supplies'))

# ---------- 精标池 ----------
def avail(row, m):
    out = []
    for k in T.TASKS:
        if (row['game_t'], k) in banned[m]: continue
        t = T.target(row, k)
        if t is None or t[1]: continue
        out.append(k)
    return out

pool, freq = [], collections.Counter()
for p in FINE:
    m = T.PRECISE[p]
    idx = {r['game_t'] for r in map(json.loads, open(T.R + '/data/frames/%s/index.jsonl' % m, encoding='utf-8'))}
    for l in open(T.R + '/data/frames/%s/gamestate.jsonl' % m, encoding='utf-8'):
        r = json.loads(l)
        if r['game_t'] not in idx: continue
        ks = avail(r, m)
        if ks: pool.append((m, r['game_t'], ks)); freq.update(ks)
print('精标池 %d 帧，(帧,字段)对 %d' % (len(pool), sum(freq.values())))

# ---------- 粗标池：武器 + 双源一致的盔/甲（helmet 只要破损，armor 破损全要、满耐久 1/2）----------
FH = {80: 1, 150: 2, 230: 3}; FA = {200: 1, 220: 2, 250: 3}
def agree(r, key):
    FT = FH if key == 'helmet' else FA
    f = r.get(key + '_full')
    return r.get(key) == (FT.get(f, 0) if f else 0)

coarse, cfreq = [], collections.Counter()
keep_full = random.Random(SEED + 7)
for m in T.COARSE:
    try:
        idx = {r['game_t'] for r in map(json.loads, open(T.R + '/data/frames/%s/index.jsonl' % m, encoding='utf-8'))}
        fh = open(T.R + '/data/frames/%s/gamestate.jsonl' % m, encoding='utf-8')
    except FileNotFoundError: continue
    for l in fh:
        r = json.loads(l)
        if r['game_t'] not in idx: continue
        ks = [k for k in ('weapon_slot1', 'weapon_slot2') if T.target(r, k) is not None]
        for key in ('helmet', 'armor'):
            if not agree(r, key): continue
            t = T.target(r, key)
            if not t or t[1]: continue
            dp = t[0]['dur_pct']
            broken = t[0][key] and dp is not None and dp < 0.999
            if broken: ks.append(key)
            elif key == 'armor' and keep_full.random() < 0.5: ks.append(key)   # 满耐久按 1/2 配比
        if ks: coarse.append((m, r['game_t'], ks)); cfreq.update(ks)
print('粗标池 %d 帧，可出字段:' % len(coarse), dict(cfreq))

# ---------- 排计划：按剩余需求抽字段 ----------
class Pool:
    def __init__(self, items, r): self.items, self.r, self.buf = items, r, []
    def draw(self, n):
        out = []
        while len(out) < n:
            if not self.buf: self.buf = list(range(len(self.items))); self.r.shuffle(self.buf)
            out.append(self.items[self.buf.pop()])
        return out

est_blocks = int(STEPS * (1 - COARSE_FRAC) * WORLD * K + STEPS * COARSE_FRAC * WORLD * CF * 1.6)
target = {k: W[k] * est_blocks for k in T.TASKS}
need = dict(target)
def pick(ks, n, r):
    take = []
    rest = list(ks)
    while len(take) < n and rest:
        ww = [max(need.get(k, 0), 1e-6) for k in rest]
        x = r.random() * sum(ww); acc_ = 0
        for i, v in enumerate(ww):
            acc_ += v
            if acc_ >= x: break
        k = rest.pop(i); take.append(k); need[k] = max(0.0, need[k] - 1)
    return take

pf, pc = Pool(pool, random.Random(SEED + 1)), Pool(coarse, random.Random(SEED + 2))
plan, expo = [], collections.Counter()
for st in range(STEPS):
    q = 'coarse' if rng.random() < COARSE_FRAC else 'fine'
    r = random.Random(SEED * 100003 + st)
    if q == 'fine':
        picks = pf.draw(WORLD); ids = [[m, g] for m, g, _ in picks]
        fields = [pick(ks, K, r) for _, _, ks in picks]
    else:
        picks = pc.draw(WORLD * CF); ids = [[m, g] for m, g, _ in picks]
        fields = [pick(ks, 2, r) for _, _, ks in picks]
    for fs in fields: expo.update(fs)
    plan.append({'step': st + 1, 'q': q, 's': 'S1', 'ids': ids, 'fields': fields, 'crop_seed': rng.randrange(1 << 30)})

val_ms = {T.PRECISE[v] for v in VAL}
assert not any(m in val_ms for p in plan for m, _ in p['ids']), '留出场混进计划了'
assert all(len(p['fields']) == len(p['ids']) for p in plan)

# ---------- 验证集：留出场全场均匀取，过约束筛 ----------
val_items = []
for v in VAL:
    m = T.PRECISE[v]
    idx = {r['game_t'] for r in map(json.loads, open(T.R + '/data/frames/%s/index.jsonl' % m, encoding='utf-8'))}
    rs = [json.loads(l) for l in open(T.R + '/data/frames/%s/gamestate.jsonl' % m, encoding='utf-8') if json.loads(l)['game_t'] in idx]
    per = VAL_FRAMES // len(VAL); step_ = max(1, len(rs) // per)
    for r in rs[::step_][:per]:
        for k in T.TASKS:
            if (r['game_t'], k) in banned[m]: continue
            t = T.target(r, k)
            if t is None or t[1]: continue
            val_items.append({'match': m, 'game_t': r['game_t'], 'task': k})
with open(OUT + '/val.jsonl', 'w', encoding='utf-8') as f:
    for it in val_items: f.write(json.dumps(it, ensure_ascii=False) + '\n')
with open(OUT + '/plan.jsonl', 'w', encoding='utf-8') as f:
    for p in plan: f.write(json.dumps(p, ensure_ascii=False) + '\n')
json.dump({'fine_frames': len(pool), 'coarse_frames': len(coarse), 'val': VAL,
           'weights': W, 'target': target}, open(OUT + '/pool.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

tot = sum(expo.values())
L = ['# v6 曝光表（按预算分配）', '',
     '步数 %d（fine %d / coarse %d）  合计 %d 块' % (STEPS, sum(1 for p in plan if p['q'] == 'fine'),
                                                sum(1 for p in plan if p['q'] == 'coarse'), tot), '',
     '| 字段 | 目标权重 | 目标块 | 实得块 | 达成 |', '|---|---|---|---|---|']
for k, n in expo.most_common():
    L.append('| %s | %.1f%% | %d | %d | %.0f%% |' % (k, 100 * W[k], int(target[k]), n, 100 * n / max(target[k], 1)))
open(OUT + '/exposure.md', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('\n'.join(L[:6]))
print('验证集 %d 条（%s，各 %d 帧）' % (len(val_items), ','.join(VAL), VAL_FRAMES // len(VAL)))
print('写入', OUT)
