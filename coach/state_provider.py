# -*- coding: utf-8 -*-
"""StateProvider：Observer 的事实来源。Observer 不关心事实从哪来。

    现在（离线回放）：OracleGameState —— 读感知数据集的 gamestate_p1p8_v1.jsonl
    以后（在线）：      CV / OCR / VLM 在线读 HUD（现有 observer.py 的横幅链路就是第一块）

两者都要产出同一份 dict（字段名和感知模型的输出一致），下游 events.py 只认这份 schema。

时间轴（FIELDS.md）：
    game_t   对局内秒，事件、冷却、时序判据全用它
    t_video  录像秒（录像是 2 倍速，game_t ≈ 2*t_video - 11），只用来回视频定位帧
"""
from __future__ import annotations

import bisect
import json
import os
from collections import defaultdict
from pathlib import Path

def _ocr_root() -> Path:
    """感知数据集根目录（逐帧读数 + 录像抽帧）。录像与读数不随仓库发布，用环境变量指过去。"""
    return Path(os.environ.get("COACH_DATA_ROOT", Path(__file__).resolve().parent.parent / "private_data"))


OCR_ROOT = _ocr_root()
DEFAULT_JSONL = OCR_ROOT / "data" / "gamestate_p1p8_v1.jsonl"   # 已废弃，只用于回归对比
# video → match 目录（逐场 gamestate.jsonl 的位置）。
# **别手写**：这张表必须和 data/gamestate_p1p8_v1.jsonl 里每行的 video/match 对上。
# 2026-09-06 我凭印象手写过一版，八场错了五场（P5 写成了 P6 的 match），跑出来 P6/P7 凭空多出
# 12–14 次 SIGNAL_LOSS、时长对不上，查了两轮才发现是映射错位。所以改成从合并版自动读一次。
def _match_map() -> dict[str, str]:
    m: dict[str, str] = {}
    try:
        for line in DEFAULT_JSONL.open(encoding="utf-8"):
            d = json.loads(line)
            m.setdefault(d["video"], d["match"])
    except FileNotFoundError:
        pass
    return m


MATCH_OF = _match_map()


def parse_countdown(s: str | None) -> int | None:
    """'mm:ss' → 秒；None → None。'00:00' = 正在缩圈（FIELDS.md 七），返回 0。"""
    if not s:
        return None
    try:
        m, sec = s.split(":")
        return int(m) * 60 + int(sec)
    except Exception:
        return None


DERIVED_DIR = Path(__file__).resolve().parent.parent / "outputs" / "derived"


def _circ_median(angles: list[float]) -> float:
    import math
    x = sum(math.cos(math.radians(a)) for a in angles); y = sum(math.sin(math.radians(a)) for a in angles)
    return math.degrees(math.atan2(y, x)) % 360


def _merge_zone_bearing(rows: list[dict], f: Path, conf_min: float = 0.2, win: int = 2) -> int:
    """把 tools/derive_zone_bearing.py 的产物并进行：zone_bearing（±win 行内 conf≥conf_min 的圆周中位）/ zone_bearing_conf。
    读不到（圈内、置信度低）就是 None —— Director/Speaker 据此**不说方向**。"""
    if not f.is_file():
        return 0
    raw = {}
    for l in f.open(encoding="utf-8"):
        d = json.loads(l)
        raw[d["t_video"]] = (d.get("bearing"), d.get("conf") or 0.0)
    n = 0
    for i, r in enumerate(rows):
        r["zone_bearing"] = None; r["zone_bearing_conf"] = None
        if r.get("zone_dist_m") is None or r["t_video"] not in raw:
            continue
        cand = []
        for j in range(max(0, i - win), min(len(rows), i + win + 1)):
            b, c = raw.get(rows[j]["t_video"], (None, 0.0))
            if b is not None and c >= conf_min:
                cand.append((b, c))
        if len(cand) >= 2:
            r["zone_bearing"] = round(_circ_median([b for b, _ in cand]))
            r["zone_bearing_conf"] = round(sum(c for _, c in cand) / len(cand), 2); n += 1
    return n


# 录像是单个玩家的第一视角；队伍面板四行里包含自己，自己在第几行每场不同。
# 2026-09-02 审片时发现（P6 22:01 把自己的红血当成"1 号队友"）。行号来源：逐场人工核对面板
# + 自己血条与四行面板血条的误差最小行，五场四排局两法一致；P2/P3/P4 单人局用误差法。
SELF_SLOT = {"P1": 4, "P5": 4, "P6": 1, "P7": 1, "P8": 4, "P2": 1, "P3": 1, "P4": 1}


def infer_self_slot(rows: list[dict]) -> int | None:
    """兜底：自己血条 <0.95 时与四行面板血条的平均绝对误差，最小的那行。"""
    err = [[] for _ in range(4)]
    for r in rows:
        hp = (r.get("team_panel") or {}).get("hp_pct"); s = r.get("hp")
        if s is None or not hp or s >= 0.95:
            continue
        for i in range(4):
            if i < len(hp) and hp[i] is not None:
                err[i].append(abs(s - hp[i]))
    mae = [(sum(e) / len(e)) if len(e) >= 20 else None for e in err]
    if all(m is None for m in mae):
        return None
    return min(range(4), key=lambda i: mae[i] if mae[i] is not None else 9) + 1


def _merge_zone_dist(rows: list[dict], f: Path) -> dict:
    """并入 tools/reocr_zone_dist.py 的重读（本地 8B 读 HUD 圈距 ROI）—— **只当否决票**：
    8B 读这么小的裁块不如上游（"1,962m" 会读成 "2"/"0m"），所以不覆盖数值；
    只有 重读没有 m 且原值 <100 的行（标记点距离小圆标被当圈距，P1 10:11 "24"）置 None。
    千位被吃（"1,962m"→962）目前无可靠下游修法，记为感知解析层待修。原始值另存 zone_dist_orig。"""
    if not f.is_file():
        return {}
    rd = {}
    for l in f.open(encoding="utf-8"):
        d = json.loads(l); rd[d["t_video"]] = d
    st = {"override": 0, "nulled": 0}
    for r in rows:
        r["zone_dist_orig"] = r.get("zone_dist_m")
        d = rd.get(r["t_video"])
        if d is None or (d.get("text") or "").startswith("ERR"):
            continue
        if (not d.get("has_m")) and r.get("zone_dist_m") is not None and r["zone_dist_m"] < 100:
            r["zone_dist_m"] = None; st["nulled"] += 1
    return st


def _merge_zone_dist_v2(rows: list[dict], f: Path) -> dict:
    """并入重读 v2（读数栏大裁块 + 前后 3 帧叠加，本地 8B）——比 v1 可靠得多（P1 一致 101/125，P5 抓出 278 行千位被吃）。
    两条规则，都要求重读带 m：
      1) 千位被吃：重读值 − 原值 ∈ {1000, 2000, 3000} → 用重读值（P5 04:21 "1,962m" 原记 962）
      2) 原值为空但相邻两行重读都有 m 且相差 ≤40m → 补上（上游漏读）
    其它不一致（±10m 级抖动）保留原值。"""
    if not f.is_file():
        return {}
    rd = {}
    for l in f.open(encoding="utf-8"):
        d = json.loads(l); rd[d["t_video"]] = d
    st = {"thousands": 0, "filled": 0}
    for i, r in enumerate(rows):
        d = rd.get(r["t_video"])
        if d is None or not d.get("has_m") or d.get("value_m") is None:
            continue
        v = d["value_m"]; o = r.get("zone_dist_m")
        if o is not None and any(abs((v - o) - k * 1000) <= 15 for k in (1, 2, 3)):   # 千位被吃（容许 ±15m 的帧间抖动）
            r["zone_dist_orig"] = o; r["zone_dist_m"] = v; st["thousands"] += 1
        elif o is None:
            nb = [rd.get(rows[j]["t_video"]) for j in (i - 1, i + 1) if 0 <= j < len(rows)]
            nb = [x for x in nb if x and x.get("has_m") and x.get("value_m") is not None]
            if len(nb) == 2 and all(abs(x["value_m"] - v) <= 40 for x in nb):
                r["zone_dist_orig"] = None; r["zone_dist_m"] = v; st["filled"] += 1
    return st


def _drop_dist_islands(rows: list[dict], max_len: int = 4, max_val: int = 100) -> int:
    """时序核对：圈距非空的"孤岛"（长度 ≤max_len 行、前后都是空、值都 <max_val）是标记点距离小圆标，不是圈外。
    真出圈至少持续十几秒且距离连续变化；P1 10:09–10:19 的 "24" 就是这种孤岛。"""
    n = 0; i = 0; L = len(rows)
    while i < L:
        if rows[i].get("zone_dist_m") is None:
            i += 1; continue
        j = i
        while j < L and rows[j].get("zone_dist_m") is not None:
            j += 1
        seg = rows[i:j]
        if len(seg) <= max_len and all((r["zone_dist_m"] or 0) < max_val for r in seg):
            for r in seg:
                r["zone_dist_orig"] = r.get("zone_dist_orig", r["zone_dist_m"]); r["zone_dist_m"] = None; n += 1
        i = j
    return n


class OracleGameState:
    """数据源：**逐场 `data/frames/<match>/gamestate.jsonl`**（09-05 人工核查后的权威版）。

    2026-09-06 换源：原来读合并版 `data/gamestate_p1p8_v1.jsonl`，那份停在 09-02，之后逐场版被人工
    改过几百条，两者已分叉。光 P8 一场 ammo_reserve 就差 205 帧（弹匣末位粘进备弹：7120/720 → 120），
    另有 ammo_mag 134、team_panel 51、stance 49、supplies 33、compass 28、zone_dist_m 23、alive 20。
    方向是合并版错、逐场版对。合并版在重新生成之前视为已废弃（由标注侧负责重新生成）。

    圈距的两层修复（千位掉位 `_merge_zone_dist*`、孤岛置空 `_drop_dist_islands`）默认关闭：
    那是旧管线的失败模式，逐场版已修，HUD 模型也没有这个失败模式（P8/P4 共 1730 帧，掉位 0 条）。
    留着反而会去"修"本来正确的值。要复现旧行为设 COACH_LEGACY_ZONE_FIX=1。
    """

    def __init__(self, jsonl: Path | None = None):
        # 回归对比用：COACH_GAMESTATE_JSONL 指向旧合并版就按旧口径读（配合 COACH_LEGACY_ZONE_FIX=1）
        if jsonl is None and os.environ.get("COACH_GAMESTATE_JSONL"):
            jsonl = Path(os.environ["COACH_GAMESTATE_JSONL"])
        self.by_video: dict[str, list[dict]] = defaultdict(list)
        if jsonl is not None:                                   # 显式给路径：还按合并版读（回归对比用）
            for line in Path(jsonl).open(encoding="utf-8"):
                d = json.loads(line)
                self.by_video[d["video"]].append(d)
        else:
            for v, match in MATCH_OF.items():
                f = OCR_ROOT / "data" / "frames" / match / "gamestate.jsonl"
                if not f.exists():
                    continue
                for line in f.open(encoding="utf-8"):
                    d = json.loads(line)
                    d.setdefault("video", v)
                    d.setdefault("match", match)
                    self.by_video[v].append(d)
        legacy = os.environ.get("COACH_LEGACY_ZONE_FIX") == "1"
        self.self_slot: dict[str, int | None] = {}
        for v, rows in self.by_video.items():
            rows.sort(key=lambda r: r["t_video"])
            if legacy:
                _merge_zone_dist(rows, DERIVED_DIR / f"{v}_zone_dist.jsonl")    # 先修圈距，再算方位（方位只在圈外行有意义）
                _merge_zone_dist_v2(rows, DERIVED_DIR / f"{v}_zone_dist_v2.jsonl")
                _drop_dist_islands(rows)
            _merge_zone_bearing(rows, DERIVED_DIR / f"{v}_zone_bearing.jsonl")
            # 模型 gamestate 只有原始 energy、没有 energy_fit（后者是标注侧的锚点拟合产物）。
            # 不兜底的话 ENERGY_TOPUP 和 run_speed 在模型链路上永远不触发 —— 那不是模型的能力问题，
            # 是字段没接上。人工 gamestate 一直有 energy_fit，这段对它是空操作。
            for r in rows:
                if r.get("energy_fit") is None and r.get("energy") is not None:
                    r["energy_fit"] = r["energy"]
            slot = SELF_SLOT.get(v) or infer_self_slot(rows)
            self.self_slot[v] = slot
            for r in rows:
                r["self_slot"] = slot
        self._tv = {v: [r["t_video"] for r in rows] for v, rows in self.by_video.items()}

    @property
    def videos(self) -> list[str]:
        return sorted(self.by_video)

    def rows(self, video: str) -> list[dict]:
        return self.by_video[video]

    def at(self, video: str, t_video: float) -> dict | None:
        """录像时刻 t_video 对应的最近一行（不晚于 t_video；早于首行返回首行）。在线回放用。"""
        tv = self._tv[video]
        i = bisect.bisect_right(tv, t_video) - 1
        return self.by_video[video][max(i, 0)]

    @staticmethod
    def match_id(rows: list[dict]) -> str:
        return rows[0]["match"]
