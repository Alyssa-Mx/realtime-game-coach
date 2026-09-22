"""记录员（observer）—— 只管"什么时候该开口"，不产生要说的话。

分工（2026-09-01 定）：
    记录员  每帧看一眼，便宜（本文件），决定 **何时** 触发
    说话人  被触发时拿到 **整个 buffer**（pipeline.build_engine_prompt），决定 **说什么**

说话人不是瞎的：他看得到画面，而且是带时间戳的一整个时间窗。

## 两级判据

    每帧 ──> ① cv2 亮像素闸门（6ms）──> 亮着？
                                        │否 → 什么都不做
                                        │是
                                        ▼
                          ② 8B 读横幅文字（0.27s，限速 + 单飞行）
                                        │
                                        ▼
                          ③ 结构化去重 (动作, 玩家号) ──> 变了才触发

**① 的 ROI 直接引用 `perception/layout.json` 的 `elements.主视角击杀横幅.bbox`；
② 的 prompt 精简自 `perception/prompts/banner.txt`。二者均为只读引用。**

## 为什么是两级（实测依据）

| 判据 | 实测 |
|---|---|
| ① cv2 亮像素占比，阈值 [0.012, 0.30] | 90 帧：召回 1.00、精确 0.91、准确率 94%，**6.1ms/帧** |
| ② 8B 读 599×44 裁剪图（留 8px、放大 3 倍） | **460 token / 0.27s**，有横幅 4/4 逐字读对、无横幅 2/2 返回 null |

**曾经用过、已废弃的判据**：ROI 的 32×4 二值指纹变化。实测在 53 个亮着的帧里触发 52 次
——横幅半透明、背景一直在动，指纹每帧都变，**纯噪声**。当时是冷却时钟在假装事件驱动。

## 为什么非读文字不可

素材里 7.0–16.9 秒横幅**一直亮着没灭**（连续 4 次击杀）。只看"亮/灭"边沿，
4 个事件里只能抓到第一个。要抓全就必须知道横幅**写的是什么**。
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# perception/layout.json :: elements.主视角击杀横幅.bbox（1280x720 基准）
BANNER_BBOX = (337, 477, 599, 44)

BRIGHT_LO = float(os.getenv("BANNER_BRIGHT_LO", "0.012"))
BRIGHT_HI = float(os.getenv("BANNER_BRIGHT_HI", "0.30"))
READ_MIN_GAP_S = float(os.getenv("BANNER_READ_GAP_S", "0.5"))   # 亮着时最快多久读一次
OFF_DEBOUNCE = int(os.getenv("BANNER_OFF_DEBOUNCE", "3"))
CROP_PAD, CROP_SCALE = 8, 3        # perception/prompts/README.md 的约定

_SYS = ("你是标注员。只输出 JSON，不输出任何其他文字。只记录画面里直接可见的事实。"
        "看不清或不存在的用 null，禁止猜测。")
_USR = ('这是《和平精英》屏幕中央横幅区域的裁剪图。此处可能出现一条白色文字的横幅通知，'
        '也可能什么都没有。\n'
        '输出 JSON：{"raw": "横幅完整文字，一字不改地照抄；没有则 null"}')

# 结构化字段：<动作> ... 玩家<两位数>。读不出结构的（OCR 抖动、物品提示）自动丢弃
_PAT = re.compile(r"(击倒|淘汰).{0,8}?玩家[^\d]{0,2}(\d{1,3})")


class SessionState(list):
    """每会话一个。继承 list 是为了让上游把它当 message_history 用时行为不变。"""

    def __init__(self):
        super().__init__()
        self.banner_on = False
        self.off_streak = 0
        self.last_key: tuple | None = None      # (动作, 玩家号)
        self.last_read_wall = 0.0
        self.reading = False                    # 单飞行，避免堆积
        self.pending: str | None = None
        self.events: list[tuple[float, str]] = []


def _roi(img):
    h, w = img.shape[:2]
    sx, sy = w / 1280.0, h / 720.0
    x, y, bw, bh = BANNER_BBOX
    return img[int(y*sy):int((y+bh)*sy), int(x*sx):int((x+bw)*sx)]


def bright_ratio(raw_bytes: bytes):
    """① cv2 闸门。返回 (亮像素占比, 原图)；解码失败返回 (None, None)。"""
    img = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    roi = _roi(img)
    if roi.size == 0:
        return None, None
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float((g > 200).mean()), img


def _crop_b64(img) -> str | None:
    h, w = img.shape[:2]
    sx, sy = w / 1280.0, h / 720.0
    x, y, bw, bh = BANNER_BBOX
    x0, y0 = max(0, int((x-CROP_PAD)*sx)), max(0, int((y-CROP_PAD)*sy))
    x1, y1 = min(w, int((x+bw+CROP_PAD)*sx)), min(h, int((y+bh+CROP_PAD)*sy))
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    roi = cv2.resize(roi, (roi.shape[1]*CROP_SCALE, roi.shape[0]*CROP_SCALE),
                     interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


async def _read_banner(reader, state: SessionState, img) -> None:
    """② + ③：读文字 → 解结构 → 变了才置 pending。"""
    try:
        b64 = _crop_b64(img)
        if not b64:
            return
        raw = await reader(_SYS, _USR, b64)
        text = ""
        if raw:
            try:
                text = (json.loads(re.search(r"\{.*\}", raw, re.S).group(0)).get("raw") or "")
            except Exception:
                text = raw
        m = _PAT.search((text or "").replace(" ", ""))
        if not m:
            return
        key = (m.group(1), m.group(2))
        if key != state.last_key:
            state.last_key = key
            reason = f"{key[0]}玩家{key[1]}"
            state.pending = reason
            state.events.append((time.monotonic(), reason))
            logger.info("[OBSERVER] 触发：%s  [原文 %s]", reason, (text or "")[:40])
    except Exception as e:
        logger.warning("[OBSERVER] 读横幅失败: %s", e)
    finally:
        state.reading = False


def observe(state: SessionState, raw_bytes: bytes, reader=None) -> None:
    """每帧调一次。① 是同步的；② 命中时以后台 task 发出，不阻塞收帧。"""
    ratio, img = bright_ratio(raw_bytes)
    if ratio is None:
        return
    lit = BRIGHT_LO <= ratio <= BRIGHT_HI
    now = time.monotonic()

    if not lit:
        state.off_streak += 1
        if state.banner_on and state.off_streak >= OFF_DEBOUNCE:
            state.banner_on = False
            state.last_key = None          # 熄灭后重置，同一条横幅再出现算新事件
        return

    state.off_streak = 0
    state.banner_on = True
    if reader is None or state.reading or now - state.last_read_wall < READ_MIN_GAP_S:
        return
    state.reading = True
    state.last_read_wall = now
    try:
        asyncio.get_running_loop().create_task(_read_banner(reader, state, img))
    except RuntimeError:
        state.reading = False


def take_pending(state: SessionState) -> str | None:
    r, state.pending = state.pending, None
    return r
