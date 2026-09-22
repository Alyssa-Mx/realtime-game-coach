# -*- coding: utf-8 -*-
"""Observer = 事件检测器：把逐帧 gamestate 变成带时间的事件。**纯规则，不调模型。**

输入是状态序列（每行一个 dict，字段见 docs/06 与 perception/rois_v2.json），内部维护最近 WINDOW_S 秒
的状态窗；每 update() 一行，返回这一行新产生的事件列表，并更新一组上下文旗标（ctx），
Director 拿 ctx 判断"现在适不适合治疗/换弹"这类行动安全性问题。

事件分两种：
    edge     状态跳变那一刻发一次（队友倒地、上下车、击杀横幅、缩圈开始…）
    persist  条件持续成立时，第一次发，之后每 reemit_s 秒再发一次（圈外远、血低、弹匣空…）
             —— 真正说不说、多久说一次由 Director 的冷却决定，这里只负责"事实仍然成立"。

阈值全是**教练参数**不是游戏规则（见 docs/PLAYBOOK.md），2026-09-02 在 P1–P8 上标定：
    自己掉血 ≥0.05/步 的 8 场只有 38 次（都是强队，末尾满血），所以 SELF_HP_DROP 阈值取 0.08；
    队友血条 hp_pct 每步抖动大（≥0.05 的 1574 次），队友状态改用人审过的 danger/downed 列表；
    信号值 <1 的帧 85% 在圈内（回圈后慢慢恢复），所以判"正在掉信号"用相邻两行的下降，不用绝对值。
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

from .minimap import compass_name, rel_dir
from .state_provider import parse_countdown

WINDOW_S = 12            # 状态窗长度（对局秒）
# ---- 以下口径来自我整理的《和平精英知识库》（README 3.2/3.3、安全区篇 5/6/7、计算器 BUDGET/SPEED/BOOST）----
RUN_SPEED = 6.3          # 收枪疾跑 m/s
DRIVE_SPEED = 18.0       # 载具越野实际均速 m/s（标称极速的一半）
CAR_PICKUP_S = 15        # 取车隐形成本（15–30s，取下限）
BOOST = [(0.91, 1.062), (0.61, 1.025)]                 # 能量档移速加成
BUDGET = {1: 200, 2: 160, 3: 120, 4: 100, 5: 70, 6: 50, 7: 35, 8: 25}   # 圈外可扛总时长（信号满→0），按圈序
PHASE_START = [300, 800, 1090, 1300, 1480, 1540, 1670, 1760]             # 各圈开始收缩时刻（8×8 图，约）
# ⚠ 圈序只在内部算预算用，**绝不输出给玩家**（知识库护栏：第几个圈是画面外推断）
CAR_THRESH = [(2000, "leave_early"), (1200, "must_car"), (600, "prefer_car"), (200, "car_if_handy"), (0, "run")]


def phase_at(t: int) -> int:
    return max(1, min(8, sum(1 for s in PHASE_START if t >= s) or 1))


def run_speed(energy) -> float:
    for th, k in BOOST:
        if energy is not None and energy >= th:
            return RUN_SPEED * k
    return RUN_SPEED


def car_advice(dist: int) -> str:
    for th, name in CAR_THRESH:
        if dist >= th:
            return name
    return "run"

HEAL_ITEMS = ("medkit", "firstaid", "bandage")
BOOST_ITEMS = ("drink", "painkiller", "adrenaline")
# 弹容量先验（含常见扩容）；实际以本场观测到的最大值为准（自适应，见 _mag_cap）
MAG_PRIOR = {"M416": 40, "SCAR-L": 40, "AKM": 40, "AUG": 40, "GROZA": 40, "M762": 40, "ACE32": 40,
             "ARX": 40, "QBZ": 40, "G36C": 40, "M16A4": 40, "MK47": 30, "UMP45": 35, "P90": 50,
             "汤姆逊": 50, "野牛": 64, "蜜獾": 40, "PKM": 100, "MG3": 75, "M249": 150, "DP-28": 47,
             "Mk14": 20, "SKS": 20, "SLR": 20, "M1加兰德": 8, "AMR": 5, "AWM": 7, "M24": 7,
             "Kar98k": 5, "M200": 5, "S1897": 5, "S686": 2, "P1911": 10}
NOT_A_GUN = ("方向盘", "手榴弹", "烟雾弹", "燃烧瓶", "震爆弹", "手雷", "信号枪", "迫击炮", "猎弓", "复合弓")


@dataclass
class Event:
    type: str
    t: int                       # game_t
    t_video: float
    kind: str = "edge"           # edge | persist
    data: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"type": self.type, "t": self.t, "t_video": self.t_video, "kind": self.kind, **self.data}


_KILL = re.compile(r"(击倒|淘汰)了?\s*玩家(\d+)")
_TEAM_MSG_KIND = [
    ("mark_vehicle", re.compile(r"标记了一辆载具")),
    ("mark_defend", re.compile(r"防守标记点")),
    ("mark_door", re.compile(r"关闭的房门")),
    ("enemy", re.compile(r"有人来过|发现敌人|敌人|有人")),
    ("gather", re.compile(r"集合")),
    ("mark", re.compile(r"标?记了\s*一处|标记了")),
    ("supply", re.compile(r"这里有|背包里有")),
    ("supply_ask", re.compile(r"我要了|需要")),
    ("airdrop", re.compile(r"补给仓|空投")),
]


_NORM = re.compile(r"[^\u4e00-\u9fffA-Za-z]")


def _norm_msg(m: str) -> str:
    return _NORM.sub("", m)


def _sim(a: str, b: str) -> float:
    import difflib
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _msg_kind(m: str) -> str:
    for k, pat in _TEAM_MSG_KIND:
        if pat.search(m):
            return k
    return "other"


class EventDetector:
    def __init__(self):
        self.win: deque[dict] = deque()
        self.prev: dict | None = None
        self.t0: int | None = None
        # 记忆
        self.last_damage_t = -999
        self.last_fire_t = -999
        self.last_kill_t = -999
        self.last_heal_t = -999
        self.last_reemit: dict[str, int] = {}
        self._big_drop_t = -999          # 最近一次一下掉 ≥30% 血的时刻（用于"顶住了"的信心提振）
        self._burst_praised_t = -999
        self._prev_dist_in_car = None    # 上一行的圈距（仅在车上时记录，用于"快到了提前想停哪"）
        self._team_banners: list = []    # (t, 队友编号或 None, 原文)，判队友倒地原因用
        self.seen_msgs: dict[tuple, list] = {}    # (说话人, 类别) → [(game_t, 规范化文本)]
        self.outside_run = 0                      # 连续几行在圈外 / 圈内（去抖）
        self.inside_run = 0
        self.dist_hist: deque[tuple[int, int]] = deque()   # (game_t, dist) 圈距历史，取中位数抗离群
        self.last_banner: tuple | None = None     # (kind, raw) 去重
        self.mag_cap_seen: dict[str, int] = {}
        self.alive_stages_done: set = set()
        self.alive_hist: deque = deque(maxlen=5)          # 剩余人数 OCR 会把 11 读成 1、13 读成 1228：取中位数
        self.mag_hist: deque = deque(maxlen=3)            # 弹匣/备弹同样掉位（146→6、169→7169）：3 行中位数 + 连续 3 行才算
        self.res_hist: deque = deque(maxlen=3)
        self.mag_low_run = 0
        self.alive_smooth = None
        self.last_row = None
        self.tally = {"self_kill": 0, "self_knock": 0, "team_kill": 0, "signal_drops": 0, "self_downed": 0,
                      "team_downed": 0, "zone_entries": 0, "hp_low_episodes": 0}
        self.was_outside = False
        self.downed = False
        self.ctx: dict = {}

    # ---------- 工具 ----------
    def _mag_cap(self, weapon: str | None) -> int | None:
        if not weapon or any(k in weapon for k in NOT_A_GUN):
            return None
        return max(self.mag_cap_seen.get(weapon, 0), MAG_PRIOR.get(weapon, 30))

    def _persist(self, name: str, t: int, reemit_s: int) -> bool:
        """persist 型事件：第一次 True，之后每 reemit_s 秒 True 一次。"""
        last = self.last_reemit.get(name)
        if last is None or t - last >= reemit_s:
            self.last_reemit[name] = t
            return True
        return False

    def _clear(self, name: str) -> None:
        self.last_reemit.pop(name, None)

    # ---------- 主入口 ----------
    def update(self, s: dict) -> list[Event]:
        t, tv = s["game_t"], s["t_video"]
        if self.t0 is None:
            self.t0 = t
        p = self.prev
        ev: list[Event] = []

        def E(type_, kind="edge", **data):
            ev.append(Event(type_, t, tv, kind, data))

        # ---- 窗口 ----
        self.win.append(s)
        while self.win and t - self.win[0]["game_t"] > WINDOW_S:
            self.win.popleft()

        hp, sig = s.get("hp"), s.get("signal")
        sup = s.get("supplies") or {}
        tp = s.get("team_panel") or {}
        weapon = s.get("weapon_main")
        mag = s.get("ammo_mag")
        in_vehicle = bool(s.get("in_vehicle"))
        cd = parse_countdown(s.get("zone_countdown"))
        raw_dist = s.get("zone_dist_m")
        shrinking = cd == 0
        # 去抖：zone_dist_m 单行的 null/非 null 跳变很多（P2 里大量 1 行的孤立值），连续 2 行才认
        if raw_dist is not None:
            self.outside_run += 1; self.inside_run = 0
            self.dist_hist.append((t, raw_dist))
        else:
            self.inside_run += 1; self.outside_run = 0
        while self.dist_hist and t - self.dist_hist[0][0] > 8:
            self.dist_hist.popleft()
        outside = self.outside_run >= 2 or (raw_dist is not None and self.was_outside)
        dist = None
        if outside and self.dist_hist:
            vals = sorted(d for _, d in self.dist_hist)
            dist = vals[len(vals) // 2]                     # 中位数：P5 里 962→17→952 这种离群单点
        # 圈距变化率（m/s，负数=在接近）
        closing = 0.0
        if len(self.dist_hist) >= 2 and self.dist_hist[-1][0] > self.dist_hist[0][0]:
            closing = (self.dist_hist[-1][1] - self.dist_hist[0][1]) / (self.dist_hist[-1][0] - self.dist_hist[0][0])

        # ---- 横幅（击杀 / 用药）----
        b = s.get("banner") or {}
        raw, kind = b.get("raw"), b.get("kind")
        if raw and (kind, raw) != self.last_banner:
            self.last_banner = (kind, raw)
            if kind in ("self_kill", "team_kill"):
                m = _KILL.search(raw.replace(" ", ""))
                act = m.group(1) if m else "淘汰"
                pid = m.group(2) if m else None
                self.kill_seen = getattr(self, "kill_seen", set())      # 横幅 OCR 抖字会把同一条算成多条：按 (类型, 玩家号) 去重计数
                key = (kind, act, pid)
                if kind == "self_kill":
                    self.last_kill_t = t
                    if pid and key not in self.kill_seen:
                        self.kill_seen.add(key); self.tally["self_knock" if act == "击倒" else "self_kill"] += 1
                    E("SELF_KNOCK" if act == "击倒" else "SELF_KILL", raw=raw, player=pid)
                else:
                    if pid and act == "淘汰" and key not in self.kill_seen:     # 只数淘汰，击倒+淘汰同一人不重复
                        self.kill_seen.add(key); self.tally["team_kill"] += 1
                    mw = re.search(r"队友\s*(\d)", raw.replace(" ", ""))       # 横幅原文里的队友编号（P1O4 09:33 审片标注）
                    who = int(mw.group(1)) if mw else None
                    if who is not None and not (1 <= who <= 4 and who != s.get("self_slot")):
                        who = None
                    self._team_banners.append((t, who, raw))
                    self._team_banners = [b for b in self._team_banners if t - b[0] <= 30]
                    E("TEAM_KILL", raw=raw, action=act, who=who, weapon=(re.search(r"使用([\u4e00-\u9fffA-Za-z0-9\-]+?)(?:突击步枪|步枪|冲锋枪|狙击枪|轻机枪|霰弹枪|射手步枪)", raw.replace(" ", "")) or [None, None])[1] if "使用" in raw else None)
            elif kind == "item_use":
                if any(k in raw for k in ("急救", "绷带", "医疗")):
                    self.last_heal_t = t
                    E("HEAL_IN_PROGRESS", raw=raw)
                elif any(k in raw for k in ("饮料", "止痛", "肾上腺", "梨")):
                    self.last_heal_t = t
                    E("BOOST_IN_PROGRESS", raw=raw)
        elif not raw:
            self.last_banner = None

        # ---- 开火（弹匣减少）与弹容量 ----
        if weapon and mag is not None and self._mag_cap(weapon) is not None:
            self.mag_cap_seen[weapon] = max(self.mag_cap_seen.get(weapon, 0), mag)
        if p and mag is not None and p.get("ammo_mag") is not None and weapon == p.get("weapon_main"):
            if mag < p["ammo_mag"]:
                self.last_fire_t = t

        # ---- 倒地 / 被扶起：以队伍面板 downed 列表里"自己那一行"为准（人审字段），且连续 2 行；
        # 血条读到 0.0 不算倒地（P1 23:15 血条 0.0 两行但人站着、面板没标倒地，审片标注实锤），只当"血极低"
        me_slot = s.get("self_slot")
        in_down = me_slot is not None and me_slot in (tp.get("downed") or [])
        self.self_down_run = (getattr(self, "self_down_run", 0) + 1) if in_down else 0
        downed = self.self_down_run >= 2 or (self.downed and in_down)
        if hp is not None and hp <= 0.0 and not downed:
            hp = 0.01                                     # 血条读成 0 但没倒：按极低血处理
        if downed and not self.downed:
            self.downed = True
            self.last_damage_t = t
            self.tally["self_downed"] += 1
            E("SELF_DOWNED", frm=(p or {}).get("hp"))
        elif self.downed and not in_down and hp is not None and hp > 0.05:
            self.downed = False
            E("SELF_REVIVED", hp=hp)

        # ---- 自己掉血 ----
        if p and hp is not None and p.get("hp") is not None and not downed:
            drop = p["hp"] - hp
            if drop >= 0.08:
                self.last_damage_t = t
                if drop >= 0.3:
                    self._big_drop_t = t
                E("SELF_HP_DROP", frm=p["hp"], to=hp, drop=round(drop, 2))
        # ---- 顶住了一波：一下掉过 ≥30% 血，之后 20 秒没再挨枪、没倒、血没归零 ----
        # （参考的一份教练 Skill 设计里的"信心提振"子模块：失误/险情之后给一句肯定，我们原来只有倒地安慰）
        if (self._big_drop_t > 0 and 20 <= t - self._big_drop_t <= 24 and t - self.last_damage_t >= 20
                and not downed and (hp or 0) > 0.05 and t - self._burst_praised_t > 180):
            self._burst_praised_t = t
            E("SURVIVED_BURST", since_s=t - self._big_drop_t, hp=hp)
        in_combat = (t - self.last_damage_t <= 10) or (t - self.last_fire_t <= 8) or (t - self.last_kill_t <= 8)
        healing = t - self.last_heal_t <= 6

        # ---- 血低（持续条件）----
        has_heal = any((sup.get(k) or 0) > 0 for k in HEAL_ITEMS)
        if hp is not None and hp < 0.5 and not healing and not downed:
            name = "SELF_HP_LOW"
            if name not in self.last_reemit:
                self.tally["hp_low_episodes"] += 1
            if self._persist(name, t, 20):
                E(name, "persist", hp=hp, in_combat=in_combat, has_heal=has_heal,
                  items={k: sup.get(k) for k in HEAL_ITEMS if sup.get(k)})
        else:
            self._clear("SELF_HP_LOW")

        # ---- 信号值下降（在圈外掉信号）----
        if p and sig is not None and p.get("signal") is not None and p["signal"] - sig >= 0.01:
            if self._persist("SIGNAL_DROPPING", t, 10):
                self.tally["signal_drops"] += 1
                E("SIGNAL_DROPPING", "persist", frm=p["signal"], to=sig, dist=raw_dist,
                  bearing=s.get("zone_bearing"), facing=s.get("compass"),
                  rel=(rel_dir(s["zone_bearing"], s["compass"]) if (s.get("zone_bearing") is not None and s.get("compass") is not None) else None),
                  painkiller=sup.get("painkiller") or 0, drink=sup.get("drink") or 0, adrenaline=sup.get("adrenaline") or 0,
                  budget_left=round(BUDGET[phase_at(t)] * sig))
        elif t - self.last_reemit.get("SIGNAL_DROPPING", -999) > 10:
            self._clear("SIGNAL_DROPPING")

        # ---- 毒圈 / 安全区 ----
        if p and shrinking and parse_countdown(p.get("zone_countdown")) not in (0, None):
            E("ZONE_SHRINK_START", outside=outside, dist=dist)
        # 方位（小地图虚线，state_provider 已做中位平滑；None = 读不到 → 不说方向）
        bearing = s.get("zone_bearing"); facing = s.get("compass")
        rel = rel_dir(bearing, facing) if (bearing is not None and facing is not None) else None
        if outside and dist is not None and dist < 60:
            # <60 米不可信：标记点距离小圆标会被当成圈距（P1 10:11 "24 米"，审片标注实锤），且真在圈边时也没什么可说
            for lv in ("URGENT", "TIGHT", "PLAN"):
                self._clear(f"ZONE_OUTSIDE_{lv}")
        elif outside and dist is not None:
            # ---- 知识库计算器逻辑：耗时 ÷ 可扛时长 ----
            v_run = run_speed(s.get("energy_fit"))
            t_run = dist / v_run
            t_car = (dist / DRIVE_SPEED) if in_vehicle else (CAR_PICKUP_S + dist / DRIVE_SPEED)
            advice = car_advice(dist)
            t_need = t_car if (in_vehicle or advice in ("prefer_car", "must_car", "leave_early")) else t_run
            budget = BUDGET[phase_at(t)] * (sig if sig is not None else 1.0)
            available = (cd if (cd is not None and cd > 0) else 0) + budget
            ratio = t_need / available if available > 0 else 9.9
            if ratio >= 1.0:
                level = "urgent"
            elif ratio >= 0.6:
                level = "tight"
            elif dist >= 200 or (cd is not None and cd <= 60):
                level = "plan"
            else:
                level = None
            if shrinking and dist >= 150 and level == "plan":
                level = "tight"                              # 正在缩圈且不在圈边：至少 tight
            if shrinking and sig is not None and sig < 0.75 and level != "urgent":
                level = "urgent"                             # 已经在掉信号且 <75%（挨打增伤）
            approaching = in_vehicle and closing <= -5.0
            if approaching and level in ("urgent", "tight"):        # 已在驾车接近：降一级
                level = {"urgent": "tight", "tight": "plan"}[level]
            if level:
                name = f"ZONE_OUTSIDE_{level.upper()}"
                reemit = {"urgent": 15, "tight": 25, "plan": 45}[level]
                if self._persist(name, t, reemit):
                    E(name, "persist", dist=dist, countdown=cd, shrinking=shrinking, in_vehicle=in_vehicle,
                      approaching=approaching, closing_mps=round(closing, 1),
                      t_run=round(t_run), t_car=round(t_car), t_need=round(t_need), budget=round(budget),
                      available=round(available), ratio=round(ratio, 2), car=advice,
                      signal=sig, bearing=bearing, facing=facing, rel=rel,
                      compass_name=(compass_name(bearing) if bearing is not None else None),
                      painkiller=sup.get("painkiller") or 0, drink=sup.get("drink") or 0,
                      adrenaline=sup.get("adrenaline") or 0)
            for lv in ("URGENT", "TIGHT", "PLAN"):
                if level is None or lv != level.upper():
                    self._clear(f"ZONE_OUTSIDE_{lv}")
        elif not outside:
            for lv in ("URGENT", "TIGHT", "PLAN"):
                self._clear(f"ZONE_OUTSIDE_{lv}")
            if self.was_outside and self.inside_run >= 3:
                self.tally["zone_entries"] += 1
                E("ZONE_ENTERED")
                self.was_outside = False
        if outside:
            self.was_outside = True

        # ---- 弹药（OCR 掉位：146→6、169→7169、31→1；用 3 行中位数，且连续 3 行低才算）----
        cap = self._mag_cap(weapon)
        if p and p.get("weapon_main") != weapon:
            self.mag_hist.clear(); self.res_hist.clear(); self.mag_low_run = 0
        if cap and mag is not None and 0 <= mag <= cap + 10:
            self.mag_hist.append(mag)
        mag_med = sorted(self.mag_hist)[len(self.mag_hist) // 2] if len(self.mag_hist) == 3 else None
        if cap and mag_med is not None:
            low = mag_med <= max(3, int(cap * 0.2))
            self.mag_low_run = self.mag_low_run + 1 if low else 0
            if low and self.mag_low_run >= 3:
                name = "AMMO_MAG_LOW"
                if self._persist(name, t, 15):
                    E(name, "persist", mag=mag_med, cap=cap, weapon=weapon, in_combat=in_combat,
                      firing=(t - self.last_fire_t <= 4))
            elif not low:
                self._clear("AMMO_MAG_LOW")
        else:
            self._clear("AMMO_MAG_LOW")
        res = s.get("ammo_reserve")
        if res is not None and 0 <= res <= 999:
            self.res_hist.append(res)
        res_med = sorted(self.res_hist)[len(self.res_hist) // 2] if len(self.res_hist) == 3 else None
        res_th = min(cap, 40) if cap else None
        if cap and mag is not None and res_med is not None and res_med < res_th and not in_vehicle and not in_combat:
            if self._persist("AMMO_RESERVE_LOW", t, 120):
                E("AMMO_RESERVE_LOW", "persist", reserve=res_med, cap=cap, weapon=weapon)
        elif res_med is not None and res_th and res_med >= res_th:
            self._clear("AMMO_RESERVE_LOW")

        # ---- 护具 ----
        hd, ad = s.get("helmet_dur_pct"), s.get("armor_dur_pct")
        broken = [n for n, v in (("头盔", hd), ("护甲", ad)) if v is not None and v < 0.3]
        if broken and not in_combat:
            if self._persist("GEAR_DAMAGED", t, 90):
                E("GEAR_DAMAGED", "persist", parts=broken, helmet_pct=hd, armor_pct=ad)
        elif not broken:
            self._clear("GEAR_DAMAGED")
        missing = [n for n, v in (("头盔", s.get("helmet")), ("护甲", s.get("armor"))) if v == 0]
        if missing and t - self.t0 >= 120 and (s.get("alive") or 100) < 100 and not in_vehicle and not in_combat:
            if self._persist("GEAR_MISSING", t, 180):
                E("GEAR_MISSING", "persist", parts=missing)
        elif not missing:
            self._clear("GEAR_MISSING")

        # ---- 队友 ----
        ptp = (p or {}).get("team_panel") or {}
        me = s.get("self_slot")                       # 面板里自己那一行：不是队友，剔除（P6 22:01 的错）
        # 面板的 danger/downed 单行会闪（P1 02:34 一行 danger=[1,2,3] 下一行全好，审片标注"这句之后才开始掉血"）：
        # 要连续 2 行都在列表里才算事件；离开列表 2 行才算这一段结束
        self.dan_run = getattr(self, "dan_run", {}); self.dan_off = getattr(self, "dan_off", {})
        self.down_run = getattr(self, "down_run", {}); self.down_off = getattr(self, "down_off", {})
        cur_down = set(tp.get("downed") or []); cur_dan = set(tp.get("danger") or [])
        for i in range(1, 5):
            if i == me:
                continue
            # downed
            if i in cur_down:
                self.down_run[i] = self.down_run.get(i, 0) + 1; self.down_off[i] = 0
                if self.down_run[i] == 2:
                    self.tally["team_downed"] += 1
                    # 倒地原因：横幅里能看出来 —— "由于在…外时间过长倒地了"=被毒倒，"误伤…自己"=自伤，
                    # 其它情况默认是被敌人打倒（我们的击杀横幅只有己方视角，敌人打倒队友不会单独出一条）。
                    # 封烟规则：周围有人才封烟 —— 被毒倒/自伤时周围不一定有人，别浪费烟。
                    cause = "enemy"
                    for bt, who, raw in self._team_banners:
                        if abs(t - bt) <= 8 and (who == i or who is None):
                            if "外时间过长" in raw or "信号" in raw: cause = "zone"; break
                            if "误伤" in raw or "自己" in raw: cause = "self"; break
                    E("TEAMMATE_DOWNED", member=i, smoke=(sup.get("smoke") or 0), cause=cause)
            else:
                self.down_off[i] = self.down_off.get(i, 0) + 1
                if self.down_off[i] >= 2:
                    self.down_run[i] = 0
            # danger（倒地的不重复报）
            if i in cur_dan and i not in cur_down:
                self.dan_run[i] = self.dan_run.get(i, 0) + 1; self.dan_off[i] = 0
                if self.dan_run[i] == 2:
                    E("TEAMMATE_DANGER", member=i, in_combat=in_combat)
            else:
                self.dan_off[i] = self.dan_off.get(i, 0) + 1
                if self.dan_off[i] >= 2:
                    self.dan_run[i] = 0
        # online_count 不用：P2/P3/P4 上它在 4→1→4 之间乱跳（读数噪声），"队友被淘汰"无法从它判。

        # ---- 队友消息（新出现的一条才算）----
        # 同一条消息在屏幕上停留几分钟，OCR 每帧抖字（"AC-E32"/"AC E32"/"G ROZA"），
        # 所以不能按原文去重：按 (说话人, 类别) 90s 内只算一次，再拿规范化文本和 300s 内的同类消息模糊比对。
        for m in s.get("team_msgs") or []:
            if m.startswith("我：") or "：" not in m:
                continue
            who, body = m.split("：", 1)
            # 粘行：一条消息里同时出现"有X"和"标记"，说明 OCR 把两条粘一起了 → 说话人不可信，只保留方位不报编号
            glued = bool(re.search(r"(我这里有|我背包里有|集合|我要了)", body)) and bool(re.search(r"标\s*记", body))
            kind_m = _msg_kind(body)
            # 说话人编号校验：只能是 1–4 且不是自己那一行；OCR 会读出 队友7/队友0/队友5（P1v6 13:47 审片标注）
            mnum = re.fullmatch(r"队友(\d)", who.strip())
            num = int(mnum.group(1)) if mnum else None
            if glued or num is None or not (1 <= num <= 4) or num == s.get("self_slot"):
                who = "队友"                                   # 编号不可信（粘行/读错/是自己）：不报几号
            # 标记消息里的 "115'" 是标记点的罗盘方位（占 76% 的消息），可以给出有依据的方向
            mb = re.search(r"(\d{1,3})'", body)
            mark_bearing = int(mb.group(1)) if mb and int(mb.group(1)) <= 360 else None
            mark_rel = rel_dir(mark_bearing, facing) if (mark_bearing is not None and facing is not None) else None
            key = (who, kind_m)
            norm = _norm_msg(body)
            recent = [(tt, nn) for (tt, nn) in self.seen_msgs.get(key, []) if t - tt <= 300]
            dup = any(t - tt <= 90 for tt, _ in recent) or any(_sim(norm, nn) >= 0.6 for _, nn in recent)
            recent.append((t, norm))
            self.seen_msgs[key] = recent
            if dup:
                continue
            E("TEAM_MSG", msg=m, who=who, msg_kind=kind_m, mark_bearing=mark_bearing, mark_rel=mark_rel,
              mark_compass=(compass_name(mark_bearing) if mark_bearing is not None else None))

        # ---- 载具 ----
        if p is not None and in_vehicle != bool(p.get("in_vehicle")):
            E("ENTER_VEHICLE" if in_vehicle else "EXIT_VEHICLE", seat=s.get("seat"), outside=outside, dist=dist)
        # ---- 开车快到圈了：提前想停哪（参考的教练 Skill 设计：载具使用/"应该将载具停在哪里"，我们原来只在下车后才说）----
        if in_vehicle and dist is not None:
            if self._prev_dist_in_car is not None and self._prev_dist_in_car >= 250 > dist:
                E("APPROACHING_DEST", dist=dist)
            self._prev_dist_in_car = dist
        else:
            self._prev_dist_in_car = None

        # ---- 对局阶段（用平滑后的剩余人数；吃鸡只在对局结束时判，见 finish()）----
        raw_alive = s.get("alive")
        # 合法性：不小于队伍存活数（自己+面板里血>0 的队友），不大于上一个平滑值+2（人数只会减少）
        # 队伍存活数：面板里有血条的行（倒地 hp=0 也算活着，剩余人数里倒地的还计数；P1 29:01 审片标注）；
        # 面板会整行漏读（None），取最近 5 行的最大值
        n_rows = sum(1 for h in (tp.get("hp_pct") or []) if h is not None)
        self.team_rows_hist = getattr(self, "team_rows_hist", deque(maxlen=5)); self.team_rows_hist.append(n_rows)
        team_alive = max(1, max(self.team_rows_hist))
        # 合理性：2 秒内最多少 5 人；OCR 掉位（60→6 连续四五行）会被这条挡住；连续 8 行都"不合理"才认为是真变了，重置
        if raw_alive is not None and team_alive <= raw_alive <= 100:
            if self.alive_smooth is None or (self.alive_smooth - 5 <= raw_alive <= self.alive_smooth + 2):
                self.alive_hist.append(raw_alive); self.alive_reject = 0
            else:
                self.alive_reject = getattr(self, "alive_reject", 0) + 1
                if self.alive_reject >= 8:
                    self.alive_hist.clear(); self.alive_hist.append(raw_alive); self.alive_reject = 0
        prev_smooth = self.alive_smooth
        if self.alive_hist:
            self.alive_smooth = sorted(self.alive_hist)[len(self.alive_hist) // 2]
        alive = self.alive_smooth
        if alive is not None and prev_smooth is not None:
            for th in (50, 20, 10, 5):
                if prev_smooth > th >= alive and th not in self.alive_stages_done:
                    self.alive_stages_done.add(th)
                    E("ALIVE_STAGE", threshold=th, alive=alive)
        # 吃鸡时刻：剩余人数 = 自己队伍的存活人数，且连续 3 行（P1v7 29:17 吃鸡后还在喊"支援队友"，审片标注）
        if alive is not None and team_alive >= 2 and alive <= team_alive:
            self.win_run = getattr(self, "win_run", 0) + 1
        else:
            self.win_run = 0
        if self.win_run >= 3 and "win" not in self.alive_stages_done:
            self.alive_stages_done.add("win")
            E("MATCH_WON", alive=alive, tally=dict(self.tally))
        self.last_row = s

        # ---- 能量（知识库v2 药品篇 §6：单为移速打能量数学上是亏的，先问安全窗口再问血线）----
        # 只在"血卡在 75 附近、绷带/急救包推不上去、且此刻确实安全"时提；纯粹能量为 0 不再提。
        en = s.get("energy_fit")
        has_boost = any((sup.get(k) or 0) > 0 for k in BOOST_ITEMS)
        # 安全窗口：≥12 秒没掉血、≥12 秒没开火、不在圈外掉信号、没在车上
        safe_window = (t - self.last_damage_t > 12) and (t - self.last_fire_t > 12) and not in_vehicle and not outside
        capped75 = hp is not None and 0.70 <= hp <= 0.86        # 只有能量能把最后这段推上去
        if en is not None and en < 0.61 and has_boost and safe_window and capped75 and not healing:
            if "energy_said" not in self.alive_stages_done and self._persist("ENERGY_TOPUP", t, 300):
                self.alive_stages_done.add("energy_said")
                E("ENERGY_TOPUP", "persist", hp=hp, energy=en,
                  items={k: sup.get(k) for k in BOOST_ITEMS if sup.get(k)}, quiet_s=min(t - self.last_damage_t, t - self.last_fire_t))
        elif not capped75 or not safe_window:
            self._clear("ENERGY_TOPUP")

        self.ctx = {
            "in_combat": in_combat, "healing": healing, "downed": downed, "in_vehicle": in_vehicle, "seat": s.get("seat"),
            "outside": outside, "shrinking": shrinking, "dist": dist, "countdown": cd,
            "hp": hp, "alive": alive, "alive_raw": raw_alive, "stance": s.get("stance"), "weapon": weapon,
            "mag": (mag_med if mag_med is not None else mag), "mag_cap": cap,
            "has_heal": has_heal, "smoke": sup.get("smoke") or 0, "has_boost": has_boost,
            "signal": sig, "energy": s.get("energy_fit"), "zone_bearing": bearing, "facing": facing, "zone_rel": rel,
            "zone_compass": (compass_name(bearing) if bearing is not None else None),
            "supplies": {k: v for k, v in sup.items() if v}, "self_slot": s.get("self_slot"),
            "recent_damage_s": t - self.last_damage_t, "recent_fire_s": t - self.last_fire_t,
        }
        self.prev = s
        return ev

    def finish(self) -> list[Event]:
        """对局结束时调用（离线：最后一行之后；在线：video.done / HUD 消失）。
        剩余人数 ≤ 队伍存活人数（≤4）且对局结束 = 吃鸡。不能用 alive==1：OCR 把 11 读成 1（P1 25:11）。"""
        s = self.last_row
        if s is None or self.alive_smooth is None or "win" in self.alive_stages_done:
            return []
        team_alive = 1 + len([1 for h in ((s.get("team_panel") or {}).get("hp_pct") or []) if h is not None and h > 0.02]) - 1
        if self.alive_smooth <= max(4, team_alive):
            self.alive_stages_done.add("win")
            return [Event("MATCH_WON", s["game_t"], s["t_video"], "edge", {"alive": self.alive_smooth})]
        return []
