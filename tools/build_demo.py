#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 data/coach/live_P1/ 打包成一个自包含的回放页 demo/index.html（数据和语音都内嵌，双击即可打开）。

    python tools/build_demo.py                         # → demo/index.html
    python tools/build_demo.py --fragment out.html     # 另出一份不带 <html>/<head> 外壳的片段（发布成网页用）
只用标准库。
"""
from __future__ import annotations

import argparse
import base64
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "coach" / "live_P1"
SELF_SLOT = 4                         # P1 这场，队伍面板第 4 行是玩家自己（每场不同，逐场人工核对）

# 事件类型 → [中文名, 所属模块]（模块 = 决策树 7 支规则的分组，见 docs/05_rules_reference.md）
EVENT_NAMES = {
    "ZONE_OUTSIDE_URGENT": ["圈外来不及", "安全区"], "ZONE_OUTSIDE_TIGHT": ["圈外偏紧", "安全区"],
    "ZONE_OUTSIDE_PLAN": ["该规划进圈了", "安全区"], "SIGNAL_DROPPING": ["正在掉信号", "安全区"],
    "ZONE_SHRINK_START": ["开始缩圈", "安全区"], "ZONE_ENTERED": ["进圈了", "安全区"],
    "SELF_HP_DROP": ["掉血", "生命值"], "SELF_HP_LOW": ["血低", "生命值"], "SURVIVED_BURST": ["顶住一波", "生命值"],
    "SELF_DOWNED": ["自己倒地", "生命值"], "SELF_REVIVED": ["被扶起", "生命值"], "HEAL_IN_PROGRESS": ["正在打药", "生命值"],
    "BOOST_IN_PROGRESS": ["正在喝能量", "生命值"], "ENERGY_TOPUP": ["趁空补能量", "生命值"],
    "AMMO_MAG_LOW": ["弹匣快空", "弹药"], "AMMO_RESERVE_LOW": ["备弹不足", "弹药"],
    "GEAR_DAMAGED": ["护具快烂", "护具"], "GEAR_MISSING": ["没有护具", "护具"],
    "TEAMMATE_DOWNED": ["队友倒地", "队友"], "TEAMMATE_DANGER": ["队友危险", "队友"], "TEAM_MSG": ["队友发消息", "队友"],
    "ENTER_VEHICLE": ["上车", "载具"], "EXIT_VEHICLE": ["下车", "载具"], "APPROACHING_DEST": ["快到圈了", "载具"],
    "SELF_KNOCK": ["击倒", "战果与阶段"], "SELF_KILL": ["淘汰", "战果与阶段"], "TEAM_KILL": ["队友拿人头", "战果与阶段"],
    "ALIVE_STAGE": ["人数跨档", "战果与阶段"], "MATCH_WON": ["吃鸡", "战果与阶段"],
}

# 逐句审读时发现的问题（第 i 句）。原样保留，不改数据。
NOTES = {
    15: "审读：这是全场唯一一句“必须说”（血从 100% 掉到 11%），说话人却把上一句原样复读了，没说掩体。"
        "句子里的“往左”没有画面里的实物撑着，离线回放的方向词校验会把它打回重生（tests/ 里有这条用例），"
        "实跑链路还没接上这层校验；而“和上一句一字不差”，现有三道校验都管不到。",
    34: "审读：依据是“只剩 1 人”，来自左上角剩余人数的读数；这一段读数一直是 1，但对局之后又打了 4 分半钟，"
        "真实人数不可能是 1，多半是 11 被读掉了一位。识别错会直接变成说错话。",
}

SUP_ORDER = ["frag", "smoke", "medkit", "firstaid", "stun", "molotov", "adrenaline", "painkiller", "emp", None, "drink", "bandage"]


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def detail(e: dict) -> str:
    t = e["type"]
    if t.startswith("ZONE_OUTSIDE"):
        return f"离圈 {e['dist']} m · 需 {e['t_need']} s / 可用 {e['available']} s"
    if t == "SELF_HP_DROP":
        return f"{round(e['frm'] * 100)}% → {round(e['to'] * 100)}%"
    if t == "SELF_HP_LOW":
        return f"血 {round(e['hp'] * 100)}%" + (" · 交战中" if e.get("in_combat") else "")
    if t in ("TEAM_KILL", "SELF_KILL", "SELF_KNOCK", "HEAL_IN_PROGRESS", "BOOST_IN_PROGRESS"):
        return e.get("raw") or ""
    if t == "TEAM_MSG":
        return e.get("msg") or ""
    if t in ("TEAMMATE_DANGER", "TEAMMATE_DOWNED"):
        return f"{e['member']} 号"
    if t == "ALIVE_STAGE":
        return f"剩 {e['alive']} 人"
    if t == "SIGNAL_DROPPING":
        return f"信号 {round(e['frm'] * 100)}% → {round(e['to'] * 100)}%"
    if t in ("ENTER_VEHICLE",):
        return e.get("seat") or ""
    if t == "APPROACHING_DEST":
        return f"离圈 {e['dist']} m"
    if t == "AMMO_MAG_LOW":
        return f"{e['weapon']} {e['mag']}/{e['cap']}"
    return ""


def build() -> str:
    rows = jsonl(LIVE / "readings.jsonl")
    keys = ["t", "alive", "hp", "sig", "en", "cd", "dist", "veh", "stance", "comp", "mag", "res", "hl", "hd", "al", "ad",
            "sup", "ban", "kf", "msg", "danger", "downed", "w1", "w2", "scope", "lp"]
    r2 = lambda x: None if x is None else round(x, 3)
    readings = []
    for r in rows:
        sup = r.get("supplies") or {}
        tp = r.get("team_panel") or {}
        readings.append([
            r["game_t"], r.get("alive"), r2(r.get("hp")), r2(r.get("signal")), r2(r.get("energy")), r.get("zone_countdown"),
            r.get("zone_dist_m"), bool(r.get("in_vehicle")), r.get("stance"), r.get("compass"), r.get("ammo_mag"),
            r.get("ammo_reserve"), r.get("helmet"), r2(r.get("helmet_dur_pct")), r.get("armor"), r2(r.get("armor_dur_pct")),
            [sup.get(k) or 0 if k else 0 for k in SUP_ORDER], (r.get("banner") or {}).get("raw"), r.get("killfeed") or [],
            r.get("team_msgs") or [], tp.get("danger") or [], tp.get("downed") or [], r.get("weapon_main"),
            r.get("weapon_other"), r.get("scope"), r2(r.get("lat_perc"))])
    events = [[e["t"], e["type"], detail(e)] for e in jsonl(LIVE / "events.jsonl")]
    speech = []
    for s in jsonl(LIVE / "speech.jsonl"):
        d = {k: s.get(k) for k in ("i", "t", "priority", "must", "interrupted", "action_cn", "branch", "item", "reason",
                                    "hint", "text", "lat_perc", "lat_dir", "first_token", "first_audio", "e2e_heard", "wav_dur")}
        if s["i"] in NOTES:
            d["note"] = NOTES[s["i"]]
        speech.append(d)
    audio = {}
    for s in speech:
        p = LIVE / "audio" / f"u{s['i']:03d}.mp3"
        if p.is_file():
            audio[str(s["i"])] = "data:audio/mpeg;base64," + base64.b64encode(p.read_bytes()).decode()
    rois = {f["key"]: [f["box"][k] for k in ("x", "y", "w", "h")]
            for f in json.loads((ROOT / "perception" / "rois_v2.json").read_text(encoding="utf-8"))["fields"]}
    rois["minimap"] = json.loads((ROOT / "perception" / "layout.json").read_text(encoding="utf-8"))["elements"]["小地图"]["bbox"]
    meta = json.loads((LIVE / "meta.json").read_text(encoding="utf-8"))
    perf = jsonl(LIVE / "perf.jsonl")
    stats = {"minutes": round(meta["wall_s"] / 60), "frames": meta["frames"], "events": len(events), "spoken": meta["spoken"],
             "perc_med": st.median(p["lat_perc"] for p in perf), "heard_med": st.median(s["e2e_heard"] for s in speech)}
    start = speech[12]["t"] + speech[12]["e2e_heard"] + 0.01          # 先找掩体 → 打药 → 转点 → P0 → 顶住了
    data = {"readingKeys": keys, "readings": readings, "events": events, "speech": speech, "audio": audio, "rois": rois,
            "stats": stats, "selfSlot": SELF_SLOT, "eventNames": EVENT_NAMES, "startAt": start}
    tpl = (ROOT / "demo" / "src" / "template.html").read_text(encoding="utf-8")
    return tpl.replace("/*__DEMO_DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "demo" / "index.html"))
    ap.add_argument("--fragment", default=None)
    a = ap.parse_args()
    page = build()
    head, body = page.split("<!--HEAD-END-->", 1)
    full = ('<!doctype html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + head + "</head>\n<body>" + body + "</body>\n</html>\n")
    Path(a.out).write_text(full, encoding="utf-8")
    print(f"→ {a.out}  {len(full.encode()) / 1e6:.2f} MB")
    if a.fragment:
        Path(a.fragment).write_text(head + body, encoding="utf-8")
        print(f"→ {a.fragment}")


if __name__ == "__main__":
    main()
