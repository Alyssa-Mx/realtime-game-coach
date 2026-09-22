"""v3 多尺度：矩形生成 / 多字段 prompt / 答案拼装 / 反解打分。定义见 TRAIN_PLAN_v3.md §三。
一个"块"= (PIL 图, 字段列表, 目标 dict)；块与块之间是独立样本（sft.py 的 collate 拍平成 batch，没有共享上下文）。
S1 走 tasks.crop（×3）；S2/S3/S4 统一 ×2 最近邻，三档像素密度相同，唯一变量是范围。"""
import hashlib, random
import numpy as np
from PIL import Image
import tasks as T

FIELDS = list(T.GRID_TASKS)          # 21 个 canonical 字段
UP = 2                               # S2/S3/S4 放大倍数
MAXWH = {'S2': (720, 400), 'S3': (1280, 420), 'S4': (T.W, T.H)}   # S2 上限要装得下底部那簇（622×137）
PAD = 8

# 多字段 prompt 里每个字段一行口径（沿用 tasks.PROMPTS 的口径句，禁放具体示例值）。
# 答案值形态与单字段 target 一致：target 只有一个键 → 直接放值；多个键 → 保留 dict。
DESC = {
 'game_t': '右下角对局计时器（分:秒），换算成总秒数的整数',
 'alive': '左上角「剩余 N」的人数整数',
 'hp': '屏幕下方血条：填充部分占整条轨道的比例，0 到 1 两位小数；整条为空为 0',
 'signal': '血条下方信号值条：比例，口径同血条',
 'energy': '血条上方黄色能量细条：比例，口径同血条',
 'ammo_mag': '武器栏上方弹药数：{"ammo_mag": 斜杠左边大字弹匣内子弹整数, "ammo_reserve": 右边小字备弹整数}；未显示弹药（空手、开车）两项 null',
 'scope': '弹药数右侧倍镜文字："2X"|"3X"|"4X"|"6X"|"8X"；没有倍数文字（红点、全息、无镜）为 null',
 'helmet': '头盔图标：{"helmet": 等级整数 0-3（图标下方黄色短杠数，没有头盔为 0）, "dur_pct": 剩余耐久比例 0 到 1 两位小数（图标从下往上被红色覆盖的是已损耗）}；没有头盔 dur_pct 为 null',
 'armor': '护甲图标：{"armor": 等级整数 0-3, "dur_pct": 剩余耐久比例}，口径同头盔',
 'zone_countdown': '右上角毒圈倒计时，"分:秒" 两位数字冒号两位数字照抄；没有倒计时为 null',
 'zone_dist_m': '右上角到安全区的米数整数；已在圈内没有显示为 null',
 'in_vehicle': '屏幕下方是否出现 km/h 载具速度表：true|false',
 'stance': '武器栏左侧站姿小人："站"|"蹲"|"趴"；图标不可见为 null',
 'compass': '顶部罗盘中央指示标所指的方位角整数 0 到 359',
 'supplies': '右下角背包物资网格，11 个数量整数 {"frag": 手雷, "smoke": 烟雾弹, "stun": 闪光弹, "molotov": 燃烧瓶, "emp": 电磁脉冲, "medkit": 医疗箱, "firstaid": 急救包, "bandage": 绷带, "painkiller": 止痛药, "drink": 能量饮料, "adrenaline": 肾上腺素}，没有该物品为 0；整个网格不可见为 null',
 'team_panel': '左侧队友状态栏（从上到下 1-4 行含本人）：{"danger": [血条变红的行号], "downed": [倒地待救的行号]}；整行变灰或有叉号是已阵亡不算倒地；没有则空列表',
 'weapon_slot1': '武器栏左框当前手持武器的侧影图标：枪名|"方向盘"（开车）|null（空框）',
 'weapon_slot2': '武器栏右框收起的另一把枪的侧影图标（较暗较小）：枪名|null（空框）',
 'banner': '屏幕中央白色横幅：{"raw": 横幅完整文字一字不改照抄，没有横幅为 null, "kind": "self_kill"|"team_kill"|"item_use"|"other"|null}；主语是"你"为 self_kill、主语是队友为 team_kill、正在使用物品为 item_use',
 'killfeed': '左侧击杀记录：从上到下每行文字照抄组成的列表，武器小图标的位置用一个空格代替；没有记录为 []',
 'team_msgs': '右侧队友通讯：每条消息完整照抄组成的列表，每条以「我：」或「队友N：」开头，不以此开头的行是上一条的换行延续必须并回；没有消息为 []',
}
assert set(DESC) == set(FIELDS)
GUN_NOTE = '枪名口径：' + T.GUNS + '。只按图标轮廓判断。'

# §二 的区域（S3 的基本单元）；相邻对可随机合并
REGIONS = {
 'top': ['compass'], 'tl': ['alive', 'team_panel'], 'tr': ['zone_dist_m', 'zone_countdown'],
 'ml': ['killfeed'], 'mr': ['team_msgs'], 'mid': ['banner'],
 # 底部 HUD 只能是一个区域：rois_v2 里 in_vehicle 压着 hp/energy/ammo/weapon_slot1（见 clusters），
 # 而 energy 右端(897)×weapon_slot2 下沿(696) 的包围盒天然把 supplies(718–885,602–689) 和 game_t 圈在里面 —— 拆开必然重复问
 'bottom': ['energy', 'hp', 'signal', 'ammo_mag', 'scope', 'helmet', 'armor', 'in_vehicle', 'stance', 'weapon_slot1', 'weapon_slot2', 'supplies', 'game_t'],
}
ADJ = [('top', 'tl'), ('top', 'tr'), ('tl', 'ml'), ('tr', 'mr')]
assert sorted(k for v in REGIONS.values() for k in v) == sorted(FIELDS)

# ---------- 几何 ----------
def boxes(match): return {k: T.box(k, match) for k in FIELDS}
def _contains(R, b):
    rx, ry, rw, rh = R; x, y, w, h = b
    return x >= rx and y >= ry and x + w <= rx + rw and y + h <= ry + rh
def _inter(R, b):
    rx, ry, rw, rh = R; x, y, w, h = b
    return min(rx + rw, x + w) > max(rx, x) and min(ry + rh, y + h) > max(ry, y)
def _union(a, b):
    x0, y0 = min(a[0], b[0]), min(a[1], b[1]); x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return (x0, y0, x1 - x0, y1 - y0)
def _clip(R):
    x0, y0 = max(0, R[0]), max(0, R[1]); x1, y1 = min(T.W, R[0] + R[2]), min(T.H, R[1] + R[3])
    return (x0, y0, x1 - x0, y1 - y0)
def _exclude(R, b):
    """把框 b 切出矩形 R：四种切法里取面积最大的；切不出（R 太小）返回 None。"""
    rx, ry, rw, rh = R; x, y, w, h = b
    cands = [(rx, ry, x - rx, rh), (x + w, ry, rx + rw - (x + w), rh), (rx, ry, rw, y - ry), (rx, y + h, rw, ry + rh - (y + h))]
    cands = [c for c in cands if c[2] >= 24 and c[3] >= 24]
    return max(cands, key=lambda c: c[2] * c[3]) if cands else None

_CL = {}
def clusters(match):
    """互相重叠的框并成一簇（连通分量），进/出矩形按簇判断 —— rois_v2 里 in_vehicle 压着 hp/energy/ammo/weapon_slot1，
    ammo_mag 压着 scope，zone 两框互压；按单框"要么全进要么全出"在这些地方无解。"""
    if match in _CL: return _CL[match]
    bx = boxes(match); ks = list(bx); par = {k: k for k in ks}
    def f(k):
        while par[k] != k: par[k] = par[par[k]]; k = par[k]
        return k
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            if _inter(bx[a], bx[b]): par[f(a)] = f(b)
    g = {}
    for k in ks: g.setdefault(f(k), []).append(k)
    out = []
    for mem in g.values():
        U = None
        for k in mem: U = bx[k] if U is None else _union(U, bx[k])
        out.append((mem, U))
    _CL[match] = out
    return out

def snap(R, match, maxwh, want=()):
    """裁边不穿框：碰到的簇先外扩包住（不超过档位上限），装不下就内缩排除。迭代到稳定；收敛不了返回 None。
    want：这一块本来要问的字段；不含 want 字段的簇优先切掉（切掉后 want 仍全在矩形里才切），
    否则 S3 的区域块会因抖动把邻区整簇吞进来（底部条 / 毒圈 / 物资几乎每帧都重复问）。"""
    R = _clip(R)
    want = set(want); bx = boxes(match) if want else {}
    for _ in range(12):
        changed = False
        for mem, U0 in clusters(match):
            if not _inter(R, U0) or _contains(R, U0): continue
            U = _union(R, U0)
            fits = U[2] <= maxwh[0] and U[3] <= maxwh[1]
            if want and not (want & set(mem)):
                R2 = _exclude(R, U0)
                if R2 is not None and all(_contains(R2, bx[k]) for k in want): R = R2; changed = True; continue
            if fits: R = _clip(U)
            else:
                R2 = _exclude(R, U0)
                if R2 is None: return None
                R = R2
            changed = True
        if not changed: break
    else: return None
    return R

def contained(R, bx): return [k for k, b in bx.items() if _contains(R, b)]

def order(fields, bx):
    """答案键顺序：从上到下、从左到右（按框中心，行内 y 相差 <30 px 视为同一行）。"""
    cs = sorted(fields, key=lambda k: (bx[k][1] + bx[k][3] / 2, bx[k][0]))
    rows, cur = [], []
    for k in cs:
        if cur and abs((bx[k][1] + bx[k][3] / 2) - (bx[cur[0]][1] + bx[cur[0]][3] / 2)) > 30: rows.append(cur); cur = []
        cur.append(k)
    if cur: rows.append(cur)
    return [k for r in rows for k in sorted(r, key=lambda k: bx[k][0])]

def cut(frame_bgr, R, up=UP):
    x, y, w, h = R
    im = Image.fromarray(frame_bgr[y:y + h, x:x + w, ::-1])
    return im.resize((im.width * up, im.height * up), Image.NEAREST)

# ---------- prompt / 答案 ----------
def prompt(fields, whole):
    head = '这是《和平精英》对局画面的' + ('整帧截图' if whole else '一块局部裁剪图') + '。'
    body = ('只读下列 %d 个字段，输出一个 JSON，键名和顺序与清单一致，不要多出或遗漏键：\n' % len(fields) +
            '\n'.join('- "%s": %s' % (k, DESC[k]) for k in fields))
    tail = '\n只记录画面里直接可见的事实，看不清或不存在的用 null，禁止猜测。'
    if any(k.startswith('weapon_') for k in fields): tail += '\n' + GUN_NOTE
    return head + '\n' + body + tail

def val(tgt): return next(iter(tgt.values())) if len(tgt) == 1 else tgt
def unval(v, tgt):
    """把多字段答案里某个字段的值反解成 tasks.score 能吃的 dict。"""
    if len(tgt) == 1: return {next(iter(tgt)): v}
    return v if isinstance(v, dict) else {}

def avail(row, skip_imputed=True):
    """该帧有目标的字段 → target dict（None 目标 / 拟合值 的字段不进清单也不进答案）。"""
    out = {}
    for k in FIELDS:
        t = T.target(row, k)
        if t is None or (skip_imputed and t[1]): continue
        out[k] = t[0]
    return out

def answer(fields, tg): return {k: val(tg[k]) for k in fields}

# ---------- 各档出块：返回 [(PIL, prompt_text, answer_dict, meta)] ----------
def _s2_blocks(frame, match, tg, rng, jitter):
    """贪心铺满：随机挑一个未覆盖字段当锚，随机外扩后 snap；块内所有有目标的字段都问（同一字段可能被两块问到，
    曝光表会计到；不做"问过就不再问"，那会让块里出现看得见却不问的字段）。"""
    bx = boxes(match); left = set(tg); blocks = []
    while left and len(blocks) < 10:
        a = rng.choice(sorted(left)); b = bx[a]; R = None
        for _ in range(4):
            e = [rng.randint(20, 200) for _ in range(4)]
            R = snap((b[0] - e[0], b[1] - e[1], b[2] + e[0] + e[2], b[3] + e[1] + e[3]), match, MAXWH['S2'])
            if R and _contains(R, b): break
            R = None
        if R is None:      # 铺不进去的字段补 S1 裁块
            blocks.append((T.crop(frame, a, match, jitter=jitter), [a], True)); left.discard(a); continue
        fs = [k for k in contained(R, bx) if k in tg]
        blocks.append((cut(frame, R), order(fs, bx), False)); left -= set(fs)
    for a in sorted(left):                                  # 块数到上限还没铺到的字段补 S1 裁块
        blocks.append((T.crop(frame, a, match, jitter=jitter), [a], True))
    return blocks

def _s3_blocks(frame, match, tg, rng):
    bx = boxes(match)
    regs = {n: [k for k in ks] for n, ks in REGIONS.items()}
    if rng.random() < 0.5:                                  # 随机合并一对相邻区域
        a, b = rng.choice(ADJ); regs[a] = regs[a] + regs.pop(b)
    blocks, covered = [], set()
    for n, ks in regs.items():
        R = None
        for k in ks: R = bx[k] if R is None else _union(R, bx[k])
        e = [rng.randint(PAD, PAD + 30) for _ in range(4)]
        R = snap((R[0] - e[0], R[1] - e[1], R[2] + e[0] + e[2], R[3] + e[1] + e[3]), match, MAXWH['S3'], want=ks)
        if R is None: continue
        fs = [k for k in contained(R, bx) if k in tg]
        if not fs: continue
        blocks.append((cut(frame, R), order(fs, bx), False)); covered |= set(fs)
    for k in tg:                                            # 漏掉的补 S1
        if k not in covered: blocks.append((T.crop(frame, k, match), [k], True))
    return blocks

def make_item(frame, match, row, tier, seed, jitter=3, skip_imputed=True):
    """精标一帧 → 该档的全部块。tier ∈ S1/S2/S3/S4。返回 [(PIL, prompt, answer, meta)]，meta={'tier','fields','rect'}。"""
    rng = random.Random(seed); np.random.seed(seed % (2 ** 31))
    tg = avail(row, skip_imputed)
    out = []
    if tier == 'S1':
        for k in T.TASKS:
            t = T.target(row, k)
            if t is None or (skip_imputed and t[1]): continue
            out.append((T.crop(frame, k, match, jitter=jitter), T.PROMPTS[k], t[0], {'tier': 'S1', 'fields': [k]}))
        return out
    if tier == 'S4':
        fs = order(list(tg), boxes(match))
        return [(cut(frame, (0, 0, T.W, T.H)), prompt(fs, True), answer(fs, tg), {'tier': 'S4', 'fields': fs})]
    blocks = _s2_blocks(frame, match, tg, rng, jitter) if tier == 'S2' else _s3_blocks(frame, match, tg, rng)
    for im, fs, single in blocks:
        if single: out.append((im, T.PROMPTS[fs[0]], tg[fs[0]], {'tier': tier + '/S1', 'fields': fs}))
        else: out.append((im, prompt(fs, False), answer(fs, tg), {'tier': tier, 'fields': fs}))
    return out

def make_coarse(frame, match, row, tier, seed, jitter=3):
    """粗标一帧 → 只出武器块。S1: 左/右框各一块（tasks.crop）；S2: 一个包住武器双框的随机矩形，只问武器。"""
    rng = random.Random(seed); np.random.seed(seed % (2 ** 31))
    tg = {k: T.target(row, k)[0] for k in ('weapon_slot1', 'weapon_slot2') if T.target(row, k) is not None}
    if not tg: return []
    if tier == 'S1':
        return [(T.crop(frame, k, match, jitter=jitter), T.PROMPTS[k], tg[k], {'tier': 'S1', 'fields': [k]}) for k in tg]
    bx = boxes(match); R = _union(bx['weapon_slot1'], bx['weapon_slot2'])
    e = [rng.randint(20, 200) for _ in range(4)]
    R2 = snap((R[0] - e[0], R[1] - e[1], R[2] + e[0] + e[2], R[3] + e[1] + e[3]), match, MAXWH['S2'])
    if R2 is None or not all(_contains(R2, bx[k]) for k in tg): R2 = _clip((R[0] - PAD, R[1] - PAD, R[2] + 2 * PAD, R[3] + 2 * PAD))
    fs = order(list(tg), bx)
    return [(cut(frame, R2), prompt(fs, False), answer(fs, tg), {'tier': 'S2', 'fields': fs})]

def frame_hash(match, t_video, seed):
    return int(hashlib.sha1(('%s|%s|%d' % (match, t_video, seed)).encode()).hexdigest(), 16)
