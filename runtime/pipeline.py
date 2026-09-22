"""本项目的 pipeline：填 base handler 留的三个 hook。

**视觉部分走"buffer 存图 + 每帧带时间戳"** —— 这是 2026-09-01 定的方案，依据：

1. 读 vllm `qwen3_vl.py` 发现，图片通道的 prompt 里 **没有任何时间信息**
   （`[image_token] * n`），而视频通道是每帧前插一段文本 `<X.X seconds>`
   （`get_video_repl()`）。M-RoPE 的时间轴只按帧递增，编码次序不编码间隔。
2. 本地 8B 实测：同样 12 帧问"第几秒"，不带时间戳答"18:20 秒"（去读 HUD 倒计时了），
   带上答"7秒"（真值约 7 秒）。
3. v2 上 A/B 各 3 次：mp4 段 与 帧+时间戳 在输出质量上**分辨不出差别**
   （组间差 < 组内极差），但帧的方式**不必等整段录完才能发** —— 这才是选它的理由。
   记录见 `pubg-skill-router` 的 `experiment/vision-mode-ab` 分支。

**帧选取 = 时间等距**（不是按下标等距）：在 buffer 的 [最老, 最新] 之间取
`num_frames` 个等距时刻，各找最近的一帧。客户端帧率抖动时，按下标等距会在时间上分布不均。
拿不到 `pts_ms` 时自动退回按下标等距。

**EVS 相似度过滤默认关闭**（`enable_frame_filter=False`，我们改的默认值，上游是 True）。
不是否定这个机制 —— 它的价值场景是**画面长时间不变**（新手卡住不知道该干什么、
等待、跑图），那时候逐帧判断确实没必要。但当前素材全是激战，画面一直在动，
过滤只会丢掉信息（实测 5fps 游戏画面上默认阈值 0.95 丢 90%）。
**接口完整保留**，`session.config` 里传 `enable_frame_filter: true` 即可开启。

时间戳基准（`TS_MODE`）：
  window  最老的那帧记作 0.0s，其余相对它 —— **默认**。值域小、留在训练分布内，
          而每次 query 本来就是无状态请求，绝对会话时间对模型没有意义，有意义的是间隔。
  session 会话开始至今的秒数。长会话会出现 `<900.0 seconds>` 这种值，未验证是否仍在分布内。
"""
from __future__ import annotations

import logging
import os
from typing import Any

from .observer import SessionState, observe, take_pending
from .video_stream_base import (
    _BAD_FRAME,
    OmniStreamingVideoHandler,
    StreamingVideoSessionConfig,
    VideoStreamTurnTrigger,
)

logger = logging.getLogger(__name__)

TS_MODE = os.getenv("TS_MODE", "window")          # window | session | off
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "2"))
COACH_MODE = os.getenv("COACH_MODE", "")           # "" = 原来的横幅记录员；"oracle" = coach/ 决策层 + Oracle gamestate
COACH_VIDEO = os.getenv("COACH_VIDEO", "P1")       # oracle 模式下客户端灌的是哪一场（pts_ms = 录像毫秒）
COACH_FRAMES = int(os.getenv("COACH_FRAMES", "4"))
COACH_WINDOW_MS = int(os.getenv("COACH_WINDOW_MS", "3000"))   # 说话人看最近多少录像毫秒（= 2 倍对局秒）
if COACH_MODE:
    from coach.session import CoachSession
    from coach.speaker import build_messages as coach_build_messages


def select_indices(all_pts: list, k: int) -> list[int]:
    """从 buffer 里挑 k 帧的下标。**纯函数** —— build_engine_prompt 和
    _sample_frame_metadata 共用它，保证 `video.frames.consumed` 报的就是真正喂进去的帧。

    优先**时间等距**：在 [最老, 最新] 之间取 k 个等距时刻，各找最近的一帧。
    客户端帧率抖动时，按下标等距会在时间上分布不均（实测末尾出现过 1.4s 空洞）。
    拿不到 pts_ms 时退回按下标等距。
    """
    n = len(all_pts)
    if n == 0:
        return []
    if n <= k:
        return list(range(n))
    if all(p is not None for p in all_pts) and all_pts[-1] > all_pts[0]:
        t0, t1 = all_pts[0], all_pts[-1]
        targets = [t0 + (t1 - t0) * j / (k - 1) for j in range(k)]
        idx, used = [], set()
        for tt in targets:
            cand = min((i for i in range(n) if i not in used),
                       key=lambda i: abs(all_pts[i] - tt))
            idx.append(cand)
            used.add(cand)
        return sorted(idx)
    stride = max(1, n // k)
    return [i * stride for i in range(k - 1)] + [n - 1]


def _pts_of(meta: list, i: int):
    m = meta[i] if i < len(meta) and isinstance(meta[i], dict) else None
    return m.get("pts_ms") if m else None


class LocalStreamingVideoHandler(OmniStreamingVideoHandler):

    # ---- 每会话的状态口袋（上游把它当 message_history 用，SessionState 继承 list） ----
    def create_message_history(self, config: StreamingVideoSessionConfig) -> Any:
        st = SessionState()
        st.coach = CoachSession(COACH_VIDEO) if COACH_MODE == "oracle" else None
        return st

    @property
    def _banner_reader(self):
        """把 backend 的 read_image 交给记录员；backend 没有这个方法就退化成只做 cv2 闸门。"""
        fn = getattr(self._engine_client, "read_image", None)
        return fn if callable(fn) else None

    # ---- 记录员：每帧看一眼，只更新状态 ----
    def on_frame_buffered(self, raw_bytes: bytes, frame_b64: str,
                          message_history: Any, config: StreamingVideoSessionConfig,
                          frame_meta: dict | None = None) -> None:
        if isinstance(message_history, SessionState):
            # should_trigger_turn 的入参里没有 message_history，只能在这里留个引用。
            # 单会话安全；多会话并发时每帧都会覆盖成"当前这帧所属会话"，而
            # should_trigger_turn 紧接着同一帧被调用，所以仍然对得上。
            self._state_for_trigger = message_history
            coach = getattr(message_history, "coach", None)
            if coach is not None:
                coach.on_frame((frame_meta or {}).get("pts_ms"))
            else:
                observe(message_history, raw_bytes, reader=self._banner_reader)

    # ---- hook 1：拍板要不要开口 ----
    def should_trigger_turn(self, trigger: VideoStreamTurnTrigger) -> bool:
        """记录员判断在 on_frame_buffered 里做完了，这里只读结论。

        `trigger` 里只有 frame_count / is_generating / config，**看不到画面** ——
        所以"看到什么"必须在 on_frame_buffered 里判断，这里只负责拍板。
        正在说话时不打断（v1 不做优先级抢占）。
        """
        st = getattr(self, "_state_for_trigger", None)
        if st is None:
            return False
        coach = getattr(st, "coach", None)
        if coach is not None:
            turn = coach.take(trigger.is_generating)
            if turn is None:
                return False
            d, c, r = turn
            logger.info("[TRIGGER] %s%s %s @%02d:%02d  %s（buffer %d 帧%s）", c["priority"], "★" if d.must else "",
                        c["action"], r["game_t"] // 60, r["game_t"] % 60, c["reason"], trigger.frame_count,
                        "，打断当前生成" if trigger.is_generating else "")
            return True
        if trigger.is_generating:
            return False
        reason = take_pending(st)
        if reason:
            logger.info("[TRIGGER] 开一轮：%s（buffer %d 帧）", reason, trigger.frame_count)
            return True
        return False

    # ---- hook 2：buffer 里的帧 -> prompt（带时间戳） ----
    def build_engine_prompt(
        self,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
        prewarmed_frames: dict[str, tuple[Any, str]],
        frame_metadata: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        meta = frame_metadata or []
        all_pts = [_pts_of(meta, i) for i in range(len(frame_buffer))]
        coach = getattr(message_history, "coach", None)
        if coach is not None and coach.last_turn is not None:
            return self._build_coach_prompt(coach, frame_buffer, all_pts, prewarmed_frames)
        idx = select_indices(all_pts, config.num_frames)
        pts = [all_pts[i] for i in idx]
        base = None
        if TS_MODE == "window":
            known = [p for p in pts if p is not None]
            base = min(known) if known else None

        prewarmed = prewarmed_frames or {}
        content: list[dict] = []
        for i, p in zip(idx, pts):
            b64 = frame_buffer[i]
            if prewarmed.get(b64) is _BAD_FRAME:
                continue
            if TS_MODE != "off" and p is not None:
                t = (p - base) / 1000.0 if base is not None else p / 1000.0
                # 写法逐字复刻 vllm qwen3_vl.py::get_video_repl 的 f"<{t:.1f} seconds>"
                content.append({"type": "text", "text": f"<{t:.1f} seconds>"})
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        if query_text:
            content.append({"type": "text", "text": query_text})

        user_message = {"role": "user", "content": content}
        messages: list[dict[str, Any]] = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        recent = message_history[-HISTORY_TURNS:] if len(message_history) > HISTORY_TURNS else message_history
        for m in recent:
            messages.append(self._text_only_message(m))
        messages.append(user_message)
        return messages, user_message

    def _build_coach_prompt(self, coach, frame_buffer, all_pts, prewarmed_frames):
        """决策层模式：最近 COACH_WINDOW_MS 录像毫秒里时间等距取 COACH_FRAMES 帧，时间戳按对局秒（= 2×录像秒）。"""
        d, c, r = coach.last_turn
        prewarmed = prewarmed_frames or {}
        known = [(i, p) for i, p in enumerate(all_pts) if p is not None and prewarmed.get(frame_buffer[i]) is not _BAD_FRAME]
        if known:
            t_last = known[-1][1]
            win = [(i, p) for i, p in known if t_last - p <= COACH_WINDOW_MS] or known[-1:]
            sel = select_indices([p for _, p in win], COACH_FRAMES)
            chosen = [win[j] for j in sel]
            base = chosen[0][1]
            frames = [((p - base) / 1000.0 * 2.0, frame_buffer[i]) for i, p in chosen]
            self._coach_sel = [i for i, _ in chosen]      # 让 video.frames.consumed 报的就是这几张
        else:
            frames = []
            self._coach_sel = []
        messages = coach_build_messages(c, coach.ctx, frames, coach.recent, r["game_t"])
        return messages, messages[-1]

    # ---- hook 3：一轮结束后怎么记历史（下一步换 v2 的 history_lines + dedupe） ----
    def on_turn_complete(self, message_history: list[dict[str, Any]],
                         user_message: dict[str, Any], response_text: str) -> None:
        coach = getattr(message_history, "coach", None)
        if coach is not None:
            coach.on_spoken(response_text.strip())
            logger.info("[SPOKEN] %s", response_text.strip())
        message_history.append(user_message)
        message_history.append({"role": "assistant", "content": response_text})

    # ---- 绕开 vllm 的 chat 预处理：走 HTTP 时服务端会做 ----
    async def _preprocess_to_engine_prompt(self, request) -> Any:
        return request.messages

    # ---- 让 video.frames.consumed 报的就是真正喂进去的帧 ----
    def _sample_frame_metadata(self, frame_metadata: list, num_frames: int) -> list:
        """[本仓库分叉] 上游这里自己又做一遍按下标 stride 采样，会与
        build_engine_prompt 的选择不一致（覆盖 hook 之后尤其明显）。改成共用 select_indices。"""
        meta = frame_metadata or []
        sel = getattr(self, "_coach_sel", None)
        if sel is not None:                                  # 决策层模式：用 _build_coach_prompt 选的那几张
            self._coach_sel = None
            return [meta[i] for i in sel if i < len(meta)]
        all_pts = [_pts_of(meta, i) for i in range(len(meta))]
        return [meta[i] for i in select_indices(all_pts, num_frames)]
