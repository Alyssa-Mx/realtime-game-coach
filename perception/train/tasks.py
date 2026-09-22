"""21 个裁剪任务的唯一定义：框 / prompt / 目标 / 打分。推理(zeroshot.py)和训练(sft.py)都只从这里取。
框来自 perception/rois_v2.json（+ 各场 rois.json 覆盖），目标来自 data/frames/<m>/gamestate.jsonl。
口径见 TRAIN_PLAN.md §二/§三/§六。prompt 里禁放具体示例值（打标阶段三次实锤会变成默认答案）。"""
import json, os, re, glob
import numpy as np
from PIL import Image

# 数据根目录：逐帧读数 data/frames/<场次>/gamestate.jsonl + 抽帧。录像与读数不随仓库发布。
R = os.environ.get('AICOACH_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'private_data'))
PAD, SCALE = 8, 3
W, H = 1280, 720

# 8 场人审对局 P1–P8 → 场次目录名。原始场次 ID 属于内部数据，不随仓库发布，放在 R/data/matches.json。
PRECISE = json.load(open(os.path.join(R, 'data', 'matches.json'), encoding='utf-8')) \
    if os.path.isfile(os.path.join(R, 'data', 'matches.json')) else {f'P{i}': f'P{i}' for i in range(1, 9)}
def _coarse():
    ms = [os.path.basename(os.path.dirname(p)) for p in sorted(glob.glob(R + '/data/frames/*/gamestate.jsonl'))]
    return [m for m in ms if m not in set(PRECISE.values())]
COARSE = _coarse()      # 54 场粗标：机器粗标，补枪谱用的就是这批

SPLIT = {'train': ['P2', 'P3', 'P4', 'P6', 'P8'], 'val': ['P7'], 'test': ['P1', 'P5']}   # smoke test 固定划分

# ---------- 框 ----------
_BASE = {f['key']: f['box'] for f in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rois_v2.json'), encoding='utf-8'))['fields']}
_OVR = {}
def box(key, match):
    if match not in _OVR:
        p = R + '/data/frames/%s/rois.json' % match
        o = json.load(open(p, encoding='utf-8')) if os.path.isfile(p) else {}
        _OVR[match] = {f['key']: f['box'] for f in o['fields']} if 'fields' in o else o
    b = _OVR[match].get(key, _BASE[key])
    return b['x'], b['y'], b['w'], b['h']

# 物资网格 3 行 4 列，第 3 行第 2 列恒空（口径同 pipeline/api_readers.py 的 KEYS）
SUP_ROWS = [['frag', 'smoke', 'medkit', 'firstaid'],
            ['stun', 'molotov', 'adrenaline', 'painkiller'],
            ['emp', None, 'drink', 'bandage']]
SUP_POS = {k: (r, c) for r, row in enumerate(SUP_ROWS) for c, k in enumerate(row) if k}
SUP_KEYS = ['sup_' + k for k in SUP_POS]

def _sup_cell_box(item, match):
    """单格框 = supplies 大框按 3x4 等分。整格给出（图标 + 它自己的数字），不额外标注。
    2026-09-03 实测：整格网格一次读 11 个数时，模型会把数字配到左邻格的图标上
    （非首列错例 95% 等于左邻格 GT），单格裁掉了这个歧义。"""
    x, y, w, h = box('supplies', match)
    r, c = SUP_POS[item]
    cw, ch = w / 4.0, h / 3.0
    return int(round(x + c * cw)), int(round(y + r * ch)), int(round(cw)), int(round(ch))

# 逐字段放大倍数（2026-09-07 实测：v6 按 x3 训的，多数字段在 x1 下不掉分，视觉 token 降 3.9 倍）。
# 只在推理时生效（训练一律 x3，改了要重训）。P8+P4 两个留出场各 120 帧交叉验证，
# 判据：两场都不掉超过 0.02 才降。helmet 在 P4 上 x1 掉到 0.592（-0.40）—— 单场测会漏掉，
# 这也是为什么必须两场都验。
SCALE_INFER = {}
try:
    import json as _j
    SCALE_INFER = _j.load(open(R + '/eval/scale_by_field.json', encoding='utf-8'))
except Exception:
    pass

def crop(frame_bgr, key, match, jitter=0, scale=None):
    """裁框 + 四周 8px + 最近邻 ×3。jitter>0 时随机平移（训练增强）。返回 PIL(RGB)。"""
    cell = key.startswith('sup_')
    x, y, w, h = _sup_cell_box(key[4:], match) if cell else box(key, match)
    pad = 2 if cell else PAD          # 单格只留 2px：8px 会把左右邻格的数字带进来，等于没切
    if cell: jitter = min(jitter, 1)  # 42x29 的格子抖 3px 会把数字切掉

    if jitter:
        x += np.random.randint(-jitter, jitter + 1); y += np.random.randint(-jitter, jitter + 1)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    im = Image.fromarray(frame_bgr[y0:y1, x0:x1, ::-1])
    sc = SCALE if scale is None else scale
    return im if sc == 1 else im.resize((im.width * sc, im.height * sc), Image.NEAREST)

# ---------- prompt ----------
# 2026-09-05 实验开关：WEAPON_ONE_PROMPT=1 时，两个武器框共用同一条 prompt。
# 起因是审读时发现：两个框是分开裁的两张图、要读的都是"这把枪叫什么"，
# 写成"左框/当前手持"和"右框/收起的另一把（较暗较小）"等于把同一种能力拆成两个任务，
# 数据没有翻倍反而被劈开 —— 而 slot2 的准确率在 v1~v4 四个版本里一直比 slot1 低 0.19~0.52。
_ONE_WEAPON_PROMPT = os.environ.get('WEAPON_ONE_PROMPT', '0') == '1'
# 2026-09-05 的第二个实验：武器框**不给 prompt、不要 JSON**，只给图，直接输出枪名。
# 想验证的是"额外的格式和描述是不是反而在干扰"。BARE_WEAPON=1 时：
#   prompt = 空串；答案 = 纯文本枪名（没有枪写「无」，开车写「方向盘」）
BARE_WEAPON = os.environ.get('BARE_WEAPON', '0') == '1'
WEAPON_TASKS = ('weapon_slot1', 'weapon_slot2')

def ans_text(key, tgt):
    """训练时喂给模型的答案文本。默认是 JSON；BARE_WEAPON 下武器任务是纯枪名。"""
    if BARE_WEAPON and key in WEAPON_TASKS:
        v = tgt.get('weapon')
        return '无' if v is None else str(v)
    return json.dumps(tgt, ensure_ascii=False)

def parse_pred(key, txt):
    """把模型原文解析成 score() 认的 dict。"""
    if BARE_WEAPON and key in WEAPON_TASKS:
        t = (txt or '').strip().split('\n')[0].strip()
        return {'weapon': None if t in ('', '无', 'null', 'None') else t}
    return parse_json(txt)
SYSTEM = '你是标注员。只输出 JSON。只记录画面里直接可见的事实，看不清或不存在用 null，禁止猜测。'
# BARE_WEAPON 下武器任务的答案是裸枪名，system 里那句"只输出 JSON"就成了自相矛盾的指令
# （2026-09-05 核对"prompt 什么都没给吗"时发现的：user 文本确实是空的，但 system 还在要求 JSON）。
# 其余任务各自的 prompt 里都写了"输出 JSON：{...}"，所以去掉这句对它们没有影响。
SYSTEM_BARE = '你是标注员。只记录画面里直接可见的事实，看不清或不存在用 null，禁止猜测。'

def system_for(key):
    return SYSTEM_BARE if (BARE_WEAPON and key in WEAPON_TASKS) else SYSTEM
GUNS = ('M416、M762、AKM、SCAR-L、ACE32、ARX、AUG、GROZA、M16A4、QBZ95、G36C、蜜獾、MG-36、'
        'UMP45、Vector、P90、UZI、野牛、汤姆逊、Mini14、SLR、SKS、Mk14、QBU、M1加兰德、'
        'M24、Kar98k、AWM、M200、AMR、M338、S12K、S686、S1897、DBS、DP-28、M249、PKM、MG3、'
        'P1911、P92、迫击炮、爆炸猎弓、召回信号枪、十字弩、平底锅')
_ratio = '值是填充部分占整条轨道长度的比例，0 到 1 之间的小数，保留两位。整条为空输出 0，整条被遮挡或不可见输出 null。'
PROMPTS = {
 'game_t': '这是《和平精英》右下角对局计时器的裁剪图，形如 分:秒。\n输出 JSON：{"game_t": <换算成总秒数的整数>}\n看不清输出 {"game_t": null}。',
 'alive': '这是《和平精英》左上角剩余人数的裁剪图，形如「剩余 N」。\n输出 JSON：{"alive": <剩余人数整数>}\n看不清输出 {"alive": null}。',
 'hp': '这是《和平精英》屏幕下方血条整条轨道的裁剪图。\n输出 JSON：{"hp": <比例>}\n' + _ratio,
 'signal': '这是《和平精英》血条下方信号值条整条轨道的裁剪图。\n输出 JSON：{"signal": <比例>}\n' + _ratio,
 'energy': '这是《和平精英》血条上方能量条（黄色细条）整条轨道的裁剪图。\n输出 JSON：{"energy": <比例>}\n' + _ratio,
 'ammo_mag': '这是《和平精英》武器栏上方弹药数的裁剪图：斜杠左边的大字是弹匣内子弹数，右边的小字是备弹数。\n'
             '输出 JSON：{"ammo_mag": <整数或null>, "ammo_reserve": <整数或null>}\n没有显示弹药（如空手、开车）两项都为 null。',
 'scope': '这是《和平精英》弹药数右侧倍镜标识的裁剪图。\n输出 JSON：{"scope": "2X"|"3X"|"4X"|"6X"|"8X"|null}\n只按文字读；没有倍数文字（红点、全息、无镜）输出 null。',
 'helmet': '这是《和平精英》头盔图标的裁剪图。图标下方的黄色短杠数量是等级（0-3，没有头盔时没有图标）；图标从下往上被红色覆盖的部分代表已损耗的耐久。\n'
           '输出 JSON：{"helmet": <等级整数>, "dur_pct": <剩余耐久比例，0 到 1 小数，两位>}\n没有头盔输出 {"helmet": 0, "dur_pct": null}。',
 'armor': '这是《和平精英》护甲图标的裁剪图。图标下方的黄色短杠数量是等级（0-3，没有护甲时没有图标）；图标从下往上被红色覆盖的部分代表已损耗的耐久。\n'
          '输出 JSON：{"armor": <等级整数>, "dur_pct": <剩余耐久比例，0 到 1 小数，两位>}\n没有护甲输出 {"armor": 0, "dur_pct": null}。',
 'zone_countdown': '这是《和平精英》右上角毒圈倒计时的裁剪图，形如 分:秒。\n输出 JSON：{"zone_countdown": "<分:秒，两位数字冒号两位数字，照抄>"}\n没有倒计时输出 {"zone_countdown": null}。',
 'zone_dist_m': '这是《和平精英》右上角安全区距离的裁剪图，显示到安全区的米数。\n输出 JSON：{"zone_dist_m": <整数米数>}\n没有显示距离（已在圈内）输出 {"zone_dist_m": null}。',
 'in_vehicle': '这是《和平精英》屏幕下方载具速度表位置的裁剪图。\n输出 JSON：{"in_vehicle": true|false}\n出现 km/h 速度表为 true，否则 false。',
 'stance': '这是《和平精英》武器栏左侧站姿小人图标的裁剪图。\n输出 JSON：{"stance": "站"|"蹲"|"趴"|null}\n按小人的姿态判断；图标不可见输出 null。',
 'compass': '这是《和平精英》屏幕顶部罗盘刻度带的裁剪图，中央有一个指示标，刻度带上有方位数字和方向字母。\n'
            '输出 JSON：{"compass": <中央指示标所指的方位角整数，0 到 359>}\n看不清输出 {"compass": null}。',
 'supplies': '这是《和平精英》右下角背包物资网格的裁剪图，每格是一种物品图标加右下角数量。\n输出 JSON（11 个键，每个是数量整数，没有该物品为 0）：\n'
             '{"frag": 手雷, "smoke": 烟雾弹, "stun": 闪光弹, "molotov": 燃烧瓶, "emp": 电磁脉冲, "medkit": 医疗箱, "firstaid": 急救包, '
             '"bandage": 绷带, "painkiller": 止痛药, "drink": 能量饮料, "adrenaline": 肾上腺素}\n值只写整数，不写中文。整个网格不可见输出 null。',
 'team_panel': '这是《和平精英》左侧队友状态栏的裁剪图，从上到下 1-4 行（含本人），每行有名字和血条。\n'
               '输出 JSON：{"danger": [<血条变红的行号>], "downed": [<处于倒地待救状态的行号>]}\n没有则为空列表。整行变灰或有叉号的是已阵亡，不算倒地。',
 'weapon_slot1': '这是《和平精英》武器栏左框的裁剪图，显示当前手持武器的侧影图标；开车时这里是方向盘。\n'
                 '输出 JSON：{"weapon": "<枪名>"|"方向盘"|null}\n枪名口径：' + GUNS + '。\n只按图标轮廓判断；空框输出 null。',
 'weapon_slot2': '这是《和平精英》武器栏右框的裁剪图，显示收起的另一把枪的侧影图标（较暗、较小）。\n'
                 '输出 JSON：{"weapon": "<枪名>"|null}\n枪名口径：' + GUNS + '。\n只按图标轮廓判断；空框输出 null。',
}

if BARE_WEAPON:
    PROMPTS['weapon_slot1'] = PROMPTS['weapon_slot2'] = ''
if _ONE_WEAPON_PROMPT:
    # 2026-09-05 定：不要任何导向性描述 —— 不说左框右框、不说主枪副枪、
    # 不说"手持的更亮更大/收起的更暗更小"。只说"这里有一件武器，认出它是什么"。
    # 名称清单保留：它定义的是输出口径（严格匹配打分需要），不含"哪个框更可能是哪把"的信息。
    _W = ('这是《和平精英》武器栏里一个框的裁剪图，框里是一件武器的侧影图标。认出它是什么。\n'
          '输出 JSON：{"weapon": "<名称>"|null}\n'
          '名称口径：' + GUNS + '、方向盘。\n看不清或框里没有东西输出 {"weapon": null}。')
    PROMPTS['weapon_slot1'] = PROMPTS['weapon_slot2'] = _W
for k in ('banner', 'killfeed', 'team_msgs'):
    PROMPTS[k] = open(R + '/prompts/%s.txt' % k, encoding='utf-8').read().strip()
_SUP_CELL_P = ('这是《和平精英》右下角背包物资网格中**单独一格**的裁剪图：左边是物品图标，右边紧跟这一格自己的数量数字。\n'
               '输出 JSON：{"count": <整数>}\n数量为 0 时数字画成暗色，仍要如实输出 0；这一格永远有数字，不要输出 null。\n'
               '只读这一格里的数字，不要参考格子外的任何数字。')
PROMPTS.update({k: _SUP_CELL_P for k in SUP_KEYS})

TASKS = list(PROMPTS)                      # 21 个整框任务 + 11 个物资单格任务
GRID_TASKS = [k for k in TASKS if not k.startswith('sup_')]
assert set(GRID_TASKS) == set(_BASE), set(GRID_TASKS) ^ set(_BASE)

# ---------- 目标 ----------
THROWABLE = {'破片手榴弹', '烟雾弹', '电磁脉冲手雷', '燃烧瓶', '闪光弹'}
def _gun(s):
    if s is None: return None
    s = str(s)
    if s.isdigit(): return 'unknown'           # 纯数字 ID，映射未建
    return 'Kar98k' if s.lower() == 'kar98k' else s
def _r2(v): return None if v is None else round(float(v), 2)

def target(row, key):
    """返回 (目标dict, imputed:bool)；None = 该帧此任务不构成样本（丢弃）。"""
    r = row
    if key in ('game_t', 'alive', 'zone_countdown', 'zone_dist_m', 'in_vehicle', 'stance', 'compass'):
        v = r[key]
        if key == 'compass' and v is not None: v = int(v) % 360
        return {key: v}, False
    if key in ('hp', 'signal'): return {key: _r2(r[key])}, False
    if key == 'energy':
        if r['energy'] is not None: return {'energy': _r2(r['energy'])}, False
        f = r.get('energy_fit')
        if f is None: return {'energy': None}, False
        return ({'energy': 0.0}, False) if f < 0.05 else ({'energy': _r2(f)}, True)   # 空条=0；否则是拟合填的
    if key == 'ammo_mag': return {'ammo_mag': r['ammo_mag'], 'ammo_reserve': r.get('ammo_reserve')}, False
    if key == 'scope': return {'scope': r['scope']}, False
    if key in ('helmet', 'armor'):
        lv, d, f = r[key], r.get(key + '_dur'), r.get(key + '_full')
        pct = _r2(d / f) if (lv and d is not None and f) else None
        return {key: lv, 'dur_pct': pct}, False
    if key == 'supplies': return ({'supplies': None} if r['supplies'] is None else dict(r['supplies'])), False
    if key.startswith('sup_'):
        if r['supplies'] is None: return None      # 网格不可见：可见性交给整框 supplies 任务，单格不构成样本
        return {'count': r['supplies'].get(key[4:])}, False
    if key == 'banner': return {'raw': r['banner']['raw'], 'kind': r['banner']['kind']}, False
    if key in ('killfeed', 'team_msgs'): return {'lines': list(r[key])}, False
    if key == 'team_panel':
        tp = r['team_panel']
        return {'danger': sorted(tp.get('danger') or []), 'downed': sorted(tp.get('downed') or [])}, False
    if key == 'weapon_slot1':
        m = r['weapon_main']
        if m is None or m in THROWABLE: return None
        return {'weapon': _gun(m)}, False
    if key == 'weapon_slot2':
        m = r['weapon_main']
        if m is None or m in THROWABLE or m == '方向盘' or r.get('seat') == '驾驶': return None
        slots = {_gun(r.get('weapon_slot1')), _gun(r.get('weapon_slot2'))} - {None, _gun(m)}
        if 'unknown' in slots or len(slots) > 1: return None
        return {'weapon': slots.pop() if slots else None}, False
    raise KeyError(key)

def iter_samples(matches, keys=TASKS, stride=1):
    """(match, frame_file, game_t, key, target, imputed) 生成器；stride>1 隔帧抽（smoke test 用）。"""
    for m in matches:
        idx = {r['game_t']: r['frame'] for r in map(json.loads, open(R + '/data/frames/%s/index.jsonl' % m, encoding='utf-8'))}
        for i, l in enumerate(open(R + '/data/frames/%s/gamestate.jsonl' % m, encoding='utf-8')):
            if i % stride: continue
            r = json.loads(l)
            if r['game_t'] not in idx: continue
            for k in keys:
                t = target(r, k)
                if t is None: continue
                yield m, idx[r['game_t']], r['game_t'], k, t[0], t[1]

# ---------- 解析与打分 ----------
def parse_json(text):
    if text is None: return None
    s = text.strip()
    s = re.sub(r'^```(?:json)?\s*|\s*```$', '', s)
    i, j = s.find('{'), s.rfind('}')
    if i < 0 or j < 0: return None
    try: return json.loads(s[i:j + 1])
    except Exception: return None

def _num(v):
    try: return None if v is None else float(v)
    except (TypeError, ValueError): return None
def _tol(p, t, tol):
    p, t = _num(p), _num(t)
    if t is None or p is None: return float(p is None and t is None)
    return float(abs(p - t) <= tol)
_QUOTES = "\'’‘′`\"“”「」『』"
_DROP = "，,、：:；;。.．！!？?（）()【】[]{}－-—_~·"
def _norm_line(x):
    """文本行归一：NFKC → 去引号/撇号 → 去空白 → 去标点。
    2026-09-04 人工核对：team_msgs/banner 逐字读对却判错，全部是 GT 侧不稳造成的：
    同一句「166'」和「166」两版并存、「'166'」三版并存、「淘汰了,玩家85」与「淘汰了 玩家85」。
    这些差异不反映模型读得对不对，所以从比较里去掉；真正的字符错（队友3→队友4、标记→你记、
    AKM→A）不受影响，仍然判错。"""
    import unicodedata
    s = unicodedata.normalize('NFKC', str(x))
    s = ''.join(c for c in s if c not in _QUOTES)
    s = re.sub(r'\s+', '', s)
    return ''.join(c for c in s if c not in _DROP)

def _eq_text(a, b, tail=2):
    """归一后相等；或一方是另一方的前缀且尾巴 <= tail 个字符。
    尾巴规则治的是 GT 侧的截断：裁剪框边缘把「(244 m」截成「(244」，模型读全了反而判错。"""
    a, b = _norm_line(a or ''), _norm_line(b or '')
    if a == b: return True
    lo, hi = (a, b) if len(a) < len(b) else (b, a)
    return bool(lo) and hi.startswith(lo) and len(hi) - len(lo) <= tail

def _line_f1(pred, tgt, tol=0.15):
    """按行软匹配：归一后 CER<=tol 算同一行（贪心一对一），返回 F1。
    2026-09-03 人工核对：killfeed/team_msgs 的"错"大半是 GT 自己的错字/漏行/撇号，
    整条严格相等会把模型读对的判成错，所以严格 acc 只当下界，这个 F1 才是可比的数。"""
    P = [_norm_line(x) for x in (pred or [])]
    T_ = [_norm_line(x) for x in (tgt or [])]
    if not P and not T_: return 1.0
    used, hit = set(), 0
    for a in T_:
        best, bi = 1e9, None
        for i, b in enumerate(P):
            if i in used: continue
            d = _cer([b], [a]) if a else (0.0 if not b else 1.0)
            if d < best: best, bi = d, i
        if bi is not None and best <= tol: used.add(bi); hit += 1
    if not hit: return 0.0
    prec, rec = hit / len(P) if P else 0.0, hit / len(T_) if T_ else 0.0
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)

def _setf1(p, t):
    p, t = set(map(str, p or [])), set(map(str, t or []))
    if not p and not t: return 1.0
    if not p or not t: return 0.0
    i = len(p & t); return 2 * i / (len(p) + len(t))
def _cer(p, t):
    p, t = '\n'.join(map(str, p or [])), '\n'.join(map(str, t or []))
    if not t: return float(bool(p))
    d = list(range(len(t) + 1))
    for i, a in enumerate(p, 1):
        nd = [i]
        for j, b in enumerate(t, 1): nd.append(min(d[j] + 1, nd[j - 1] + 1, d[j - 1] + (a != b)))
        d = nd
    return min(1.0, d[-1] / len(t))

def score(key, pred, tgt):
    """返回该样本的指标 dict（每个值 0/1 或 [0,1]），按字段类型定义；pred=None 表示 JSON 不合法。"""
    out = {'json_ok': float(pred is not None)}
    p = pred or {}
    if key in ('game_t', 'alive', 'zone_dist_m', 'zone_countdown', 'scope', 'stance', 'in_vehicle'):
        pv, tv = p.get(key), tgt[key]
        out['acc'] = float(_num(pv) == _num(tv)) if key in ('game_t', 'alive', 'zone_dist_m') else float(pv == tv)
        out['null_tp'] = float(pv is None and tv is None); out['null_p'] = float(pv is None); out['null_t'] = float(tv is None)
    elif key in ('hp', 'signal', 'energy'):
        out['acc'] = _tol(p.get(key), tgt[key], 0.05)
        if _num(p.get(key)) is not None and _num(tgt[key]) is not None: out['mae'] = abs(_num(p.get(key)) - _num(tgt[key]))
    elif key == 'compass':
        pv, tv = _num(p.get('compass')), _num(tgt['compass'])
        if pv is None or tv is None: out['acc'] = float(pv is None and tv is None)
        else:
            d = abs(pv - tv) % 360; d = min(d, 360 - d); out['acc'] = float(d <= 5); out['mae'] = d
    elif key == 'ammo_mag':
        out['acc_mag'] = float(_num(p.get('ammo_mag')) == _num(tgt['ammo_mag']))
        out['acc_reserve'] = float(_num(p.get('ammo_reserve')) == _num(tgt['ammo_reserve']))
        out['acc'] = out['acc_mag'] * out['acc_reserve']
    elif key in ('helmet', 'armor'):
        out['acc_level'] = float(_num(p.get(key)) == _num(tgt[key]))
        out['acc_dur'] = _tol(p.get('dur_pct'), tgt['dur_pct'], 0.05)
        out['acc'] = out['acc_level'] * out['acc_dur']
    elif key == 'supplies':
        if 'supplies' in tgt: out['acc'] = float(pred is None or not p or p.get('supplies', 0) is None)   # 目标=网格不可见
        else:
            per = [float(_num(p.get(k)) == float(v)) for k, v in tgt.items()]
            out['acc_item'] = sum(per) / len(per); out['acc'] = float(all(per))
    elif key.startswith('sup_'):
        out['acc'] = float(_num(p.get('count')) == _num(tgt['count']))
    elif key == 'banner':
        out['acc_kind'] = float(p.get('kind') == tgt['kind'])
        out['acc_raw'] = float(_eq_text(p.get('raw'), tgt['raw']))
        out['acc_strict'] = float(p.get('raw') == tgt['raw'] and p.get('kind') == tgt['kind'])
        out['acc'] = out['acc_kind'] * out['acc_raw']; out['cer'] = _cer([p.get('raw') or ''], [tgt['raw'] or ''])
        out['null_tp'] = float(p.get('raw') is None and tgt['raw'] is None); out['null_p'] = float(p.get('raw') is None); out['null_t'] = float(tgt['raw'] is None)
    elif key in ('killfeed', 'team_msgs'):
        pl, tl = list(p.get('lines') or []), tgt['lines']
        out['acc_strict'] = float(pl == tl)
        out['acc'] = float(len(pl) == len(tl) and all(_eq_text(a, b) for a, b in zip(pl, tl)))   # 逐行归一后相等（含截断容忍）
        out['f1_line'] = _line_f1(pl, tl)
        out['cer'] = _cer(pl, tl); out['acc_n'] = float(len(pl) == len(tl))
    elif key == 'team_panel':
        out['f1_danger'] = _setf1(p.get('danger'), tgt['danger']); out['f1_downed'] = _setf1(p.get('downed'), tgt['downed'])
        out['acc'] = float(out['f1_danger'] == 1 and out['f1_downed'] == 1)
    elif key in ('weapon_slot1', 'weapon_slot2'):
        pv, tv = p.get('weapon'), tgt['weapon']
        pv = _gun(pv) if isinstance(pv, str) else pv
        out['acc'] = float(pv == tv)
        out['null_tp'] = float(pv is None and tv is None); out['null_p'] = float(pv is None); out['null_t'] = float(tv is None)
    return out

if __name__ == '__main__':
    import collections, sys
    ms = [PRECISE[v] for v in sys.argv[1:]] or list(PRECISE.values())
    n = collections.Counter(); imp = collections.Counter()
    for m, fr, gt, k, t, im in iter_samples(ms):
        n[k] += 1; imp[k] += im
    for k in TASKS: print('%-15s %6d  imputed %d' % (k, n[k], imp[k]))
    print('total', sum(n.values()))
