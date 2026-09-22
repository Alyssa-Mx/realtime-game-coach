# -*- coding: utf-8 -*-
"""小地图读方位（知识库《小地图信息读取》路径 2）：小地图北朝上、以玩家为中心；圈外时会画一条从玩家指向
安全区最近边缘的白色虚线，虚线的方向就是安全区方位（0=北，顺时针）。

只在圈外（zone_dist_m 非 null）时读；圈内没有虚线，读出来的是路网/网格噪声。
ROI 来自 perception/layout.json 的 `小地图` [1101,0,179,187]（1280×720 基准），只读引用。

2026-09-02 在 P1 上验证：圈外 122–286 秒虚线方位 166–182°（南），小地图人工看也是朝下，一致。
"""
from __future__ import annotations

import cv2
import numpy as np

MINIMAP_BBOX = (1101, 0, 179, 187)
R_MIN, R_MAX = 14, 85          # 排除中心玩家图标，只看这一圈环带
MIN_PIX = 15


def zone_bearing(img_bgr: np.ndarray) -> tuple[float | None, float]:
    """返回 (方位角 0–360 或 None, 置信度 0–1)。置信度 = 主方向 ±15° 内白像素占环带白像素的比例，
    再乘以"沿半径分布是否连续"（虚线应从内环到外环都有点）。"""
    h, w = img_bgr.shape[:2]
    sx, sy = w / 1280.0, h / 720.0
    x, y, bw, bh = MINIMAP_BBOX
    mm = img_bgr[int(y * sy):int((y + bh) * sy), int(x * sx):int((x + bw) * sx)]
    if mm.size == 0:
        return None, 0.0
    hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 2] > 200) & (hsv[..., 1] < 40)
    ys, xs = np.nonzero(white)
    cy, cx = mm.shape[0] / 2.0, mm.shape[1] / 2.0
    dx, dy = xs - cx, ys - cy
    r = np.hypot(dx, dy)
    keep = (r > R_MIN * sy) & (r < R_MAX * sy)
    if keep.sum() < MIN_PIX:
        return None, 0.0
    ang = (np.degrees(np.arctan2(dx[keep], -dy[keep]))) % 360
    hist, edges = np.histogram(ang, bins=36, range=(0, 360))
    k = int(hist.argmax()); c = edges[k] + 5
    d = ((ang - c + 180) % 360) - 180
    sel = np.abs(d) < 15
    if sel.sum() < MIN_PIX:
        return None, 0.0
    bearing = (c + float(d[sel].mean())) % 360
    frac = sel.sum() / keep.sum()
    rr = r[keep][sel]
    bins = np.histogram(rr, bins=6, range=(R_MIN * sy, R_MAX * sy))[0]
    spread = float((bins > 0).mean())          # 6 段半径里有几段有点
    return bearing, round(float(frac * spread), 3)


def rel_dir(bearing: float, facing: float) -> str:
    """知识库 2.2 的报点方位约定（以视野正前方为 0°）。"""
    d = (bearing - facing + 360) % 360
    names = ["正前方", "右前方", "右手边", "右后方", "正后方", "左后方", "左手边", "左前方"]
    return names[int(((d + 22.5) % 360) // 45)]


def compass_name(bearing: float) -> str:
    names = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
    return names[int(((bearing + 22.5) % 360) // 45)]
