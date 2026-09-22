#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线给 Oracle gamestate 补一列 zone_bearing（圈外时从小地图虚线读）→ outputs/derived/<P>_zone_bearing.jsonl
    python tools/derive_zone_bearing.py P1 P5 P6 P7 P8
每行 {t_video, game_t, bearing, conf}；圈内行不写。state_provider 会自动加载并合并（3 行中位数平滑）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from coach.minimap import zone_bearing               # noqa: E402
from coach.state_provider import OCR_ROOT, OracleGameState  # noqa: E402

out_dir = ROOT / "outputs" / "derived"; out_dir.mkdir(parents=True, exist_ok=True)
g = OracleGameState()
for v in sys.argv[1:] or g.videos:
    rows = g.rows(v); match = g.match_id(rows); n = 0
    with (out_dir / f"{v}_zone_bearing.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            if r.get("zone_dist_m") is None:
                continue
            img = cv2.imread(str(OCR_ROOT / "data" / "frames" / match / f"t{r['t_video']:07.1f}.jpg"))
            if img is None:
                continue
            b, conf = zone_bearing(img)
            f.write(json.dumps({"t_video": r["t_video"], "game_t": r["game_t"], "bearing": (None if b is None else round(b, 1)), "conf": conf}) + "\n")
            n += 1
    print(f"{v}: {n} 行圈外，写到 outputs/derived/{v}_zone_bearing.jsonl", flush=True)
