# -*- coding: utf-8 -*-
"""在线会话里的决策层：把 runtime 每帧的 pts 喂进来 → 推进事实源 → Observer → Director → 待说的决策。

runtime 的 hook（pipeline.py）只做三件事：
    on_frame_buffered  → CoachSession.on_frame(pts_ms)
    should_trigger_turn → CoachSession.take(is_generating)   有决策就开一轮；P0 must 在生成中也开（= 打断）
    build_engine_prompt → speaker.build_messages(...)        画面从 runtime 的 buffer 里取

现在的事实源是 Oracle gamestate（按 pts_ms=录像毫秒 对齐到最近一行）。换成在线 CV/OCR 时只换 provider。
"""
from __future__ import annotations

import logging

from .director import Director
from .events import EventDetector
from .state_provider import OracleGameState

logger = logging.getLogger(__name__)


class CoachSession:
    def __init__(self, video: str, provider: OracleGameState | None = None, utter_s: float = 3.5):
        self.g = provider or OracleGameState()
        self.video = video
        self.rows = self.g.rows(video)
        self.match = self.g.match_id(self.rows)
        self.det = EventDetector()
        # 在线时"忙不忙"由 runtime 的 is_generating 再把一道关；Director 内部仍按 3.5s/句模拟，和离线一致
        # （8B 0.3s 就答完、帧又 4 倍速灌时 is_generating 几乎不为真，只靠它会比离线吵三倍）
        self.dr = Director(utter_s=utter_s)
        self.i = -1
        self.pending: tuple | None = None       # (Decision, chosen candidate dict, row)
        self.recent: list[dict] = []            # 已说出的 [{t, text}]
        self.last_turn: tuple | None = None
        self.ctx: dict = {}
        self.n_overwritten = 0

    def on_frame(self, pts_ms) -> None:
        if pts_ms is None:
            return
        tv = pts_ms / 1000.0
        while self.i + 1 < len(self.rows) and self.rows[self.i + 1]["t_video"] <= tv + 1e-6:
            self.i += 1
            r = self.rows[self.i]
            evs = self.det.update(r)
            d = self.dr.step(r["game_t"], r["t_video"], evs, self.det.ctx)
            if d.speak:
                c = d.must or d.top3[0]
                if self.pending is not None:
                    self.n_overwritten += 1
                    logger.info("[COACH] 未说出就被覆盖：%s ← %s", self.pending[1]["action"], c["action"])
                self.pending = (d, c, r)
        self.ctx = dict(self.det.ctx)

    def take(self, is_generating: bool):
        if self.pending is None:
            return None
        d, c, r = self.pending
        if is_generating and not (c["priority"] == "P0" and d.must):
            return None                         # 等当前这句说完；P0 must 直接打断
        self.pending = None
        self.last_turn = (d, c, r)
        return self.last_turn

    def on_spoken(self, text: str) -> None:
        if self.last_turn and text:
            self.recent.append({"t": self.last_turn[2]["game_t"], "text": text})
