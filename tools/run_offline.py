#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线跑 Observer → Director（不调任何模型）：8 场 gamestate → 事件 + 决策时间线 + 统计。

用法：
    python tools/run_offline.py [--videos P1,P2] [--out outputs/offline_v1]
产物（每场一个目录）：events.jsonl / decisions.jsonl / timeline.md，外加总的 report.md
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from coach.director import Director          # noqa: E402
from coach.events import EventDetector       # noqa: E402
from coach.state_provider import OracleGameState  # noqa: E402


def mmss(t: int) -> str:
    return f"{t // 60:02d}:{t % 60:02d}"


def run_video(g: OracleGameState, v: str, out: Path) -> dict:
    rows = g.rows(v)
    det, dr = EventDetector(), Director()
    out.mkdir(parents=True, exist_ok=True)
    fe = (out / "events.jsonl").open("w", encoding="utf-8")
    fd = (out / "decisions.jsonl").open("w", encoding="utf-8")
    spoken, n_ev = [], collections.Counter()
    for r in rows:
        evs = det.update(r)
        for e in evs:
            n_ev[e.type] += 1
            fe.write(json.dumps(e.as_dict(), ensure_ascii=False) + "\n")
        d = dr.step(r["game_t"], r["t_video"], evs, det.ctx)
        if d.speak or d.dropped or d.interrupted:
            fd.write(json.dumps(d.as_dict(), ensure_ascii=False) + "\n")
        if d.speak:
            spoken.append(d)
    for e in det.finish():                     # 对局结束：吃鸡判定
        n_ev[e.type] += 1; fe.write(json.dumps(e.as_dict(), ensure_ascii=False) + "\n")
        d = dr.step(e.t, e.t_video, [e], det.ctx)
        if d.speak:
            fd.write(json.dumps(d.as_dict(), ensure_ascii=False) + "\n"); spoken.append(d)
    fe.close(); fd.close()

    dur_s = rows[-1]["game_t"] - rows[0]["game_t"]
    lines = [f"# {v}  （{g.match_id(rows)}）", "",
             f"对局 {mmss(rows[0]['game_t'])}–{mmss(rows[-1]['game_t'])}，{len(rows)} 行，开口 **{len(spoken)}** 次，"
             f"{len(spoken) / (dur_s / 60):.1f} 次/分钟；打断 {dr.stats['interrupts']}，"
             f"冷却拦下 {dr.stats['cooldown_blocked']}，过期丢弃 {sum(dr.stats['expired'].values())}，P3 忙时丢 {dr.stats['p3_dropped_busy']}", "",
             "| 对局时间 | 录像秒 | P | Must | action | 分支 | 决策项 | 依据 | 给说话人的提示 | 合并/次选 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for d in spoken:
        c = d.top3[0] if not d.must else d.must
        r0 = c["routes"][0]
        others = "；".join(f"{x['action']}({x['priority']})" for x in d.top3[1:3])
        merged = "、".join(m["action"] for m in c.get("merged", []))
        extra = (f"合并:{merged}" if merged else "") + ((" 次选:" + others) if others else "")
        lines.append(f"| {mmss(d.t)} | {d.t_video:.0f} | {c['priority']} | {'★' if d.must else ''}{'⚡' if d.interrupted else ''} | "
                     f"{c['action']} | {r0['branch']} | {r0['item'] or ''} | {c['reason']} | {c['hint']} | {extra} |")
    lines += ["", "## 事件计数", "", "| 事件 | 次数 |", "|---|---|"] + [f"| {k} | {n} |" for k, n in sorted(n_ev.items())]
    (out / "timeline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    by_p = collections.Counter(d.top3[0]["priority"] if not d.must else "P0" for d in spoken)
    by_a = collections.Counter((d.must or d.top3[0])["action"] for d in spoken)
    by_b = collections.Counter((d.must or d.top3[0])["routes"][0]["branch"] for d in spoken)
    return {"video": v, "rows": len(rows), "dur_min": dur_s / 60, "spoken": len(spoken), "by_priority": dict(by_p),
            "by_action": dict(by_a), "by_branch": dict(by_b), "events": dict(n_ev), "stats": dr.stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="")
    ap.add_argument("--out", default="outputs/offline_v1")
    a = ap.parse_args()
    g = OracleGameState()
    vids = a.videos.split(",") if a.videos else g.videos
    out = Path(a.out)
    res = [run_video(g, v, out / v) for v in vids]
    tot_a, tot_b, tot_p, tot_e = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    lines = ["# Offline Observer→Director 报告", "", "| 场 | 时长(分) | 开口 | 次/分 | P0 | P1 | P2 | P3 | 打断 | 过期 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        p = r["by_priority"]
        lines.append(f"| {r['video']} | {r['dur_min']:.1f} | {r['spoken']} | {r['spoken'] / r['dur_min']:.2f} | "
                     f"{p.get('P0', 0)} | {p.get('P1', 0)} | {p.get('P2', 0)} | {p.get('P3', 0)} | {r['stats']['interrupts']} | {sum(r['stats']['expired'].values())} |")
        tot_a.update(r["by_action"]); tot_b.update(r["by_branch"]); tot_p.update(r["by_priority"]); tot_e.update(r["events"])
    lines += ["", "## 按 action", "", "| action | 次数 |", "|---|---|"] + [f"| {k} | {n} |" for k, n in tot_a.most_common()]
    lines += ["", "## 按框架分支（首路由）", "", "| 分支 | 次数 |", "|---|---|"] + [f"| {k} | {n} |" for k, n in tot_b.most_common()]
    lines += ["", "## 事件总计", "", "| 事件 | 次数 |", "|---|---|"] + [f"| {k} | {n} |" for k, n in sorted(tot_e.items())]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
