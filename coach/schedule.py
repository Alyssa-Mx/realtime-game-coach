# -*- coding: utf-8 -*-
"""把 speech.jsonl 排到成品视频的时间轴上（render.py / speech_srt.py / serve_review.py 共用同一套规则）：
视频 0 秒 = 第一行 t_video − pad；一句放在它的决策时刻 (t_video − tv_from)×slow；上一句没念完顺延 0.3s。"""
from __future__ import annotations

import json
from pathlib import Path

from .state_provider import OracleGameState


def schedule(run_dir: Path, pad: float = 3.0, slow: float = 1.0, g: OracleGameState | None = None) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    rows = (g or OracleGameState()).rows(meta["video"])
    t_from = meta["t_from"] if meta["t_from"] is not None else rows[0]["game_t"]
    tv_from = max(0.0, next(r["t_video"] for r in rows if r["game_t"] >= t_from) - pad)
    recs = [json.loads(l) for l in (run_dir / "speech.jsonl").open(encoding="utf-8")]
    out, prev_end = [], 0.0
    for r in recs:
        if not r.get("text") or not r.get("wav_dur"):
            continue
        start = max((r["t_video"] - tv_from) * slow, prev_end + 0.3)
        end = start + r["wav_dur"]; prev_end = end
        out.append({**r, "start": round(start, 2), "end": round(end, 2)})
    return {"video": meta["video"], "match": meta["match"], "tv_from": tv_from, "slow": slow, "utterances": out}
