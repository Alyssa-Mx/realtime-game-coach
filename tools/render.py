#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把一次回放（speech.jsonl + wav/）渲染成教练版视频：原片段放慢到真实对局速度 + 决策字幕 + 语音。

    python tools/render.py outputs/replay_P1_a [--slow 2] [--realistic-latency] [--out coached.mp4]

- 原录像是 2 倍速（FIELDS.md），默认 --slow 2 把它拉回 1 倍对局速度，语音时长才和战况对得上。
- 字幕四行（沿用 pubg-skill-router-v2/render_video.py 的做法，字体也只读引用它的 assets/fonts）：
      ★P0·生存中断  先找掩体          ← 决策层标签（Must 带 ★，打断带 ⚡）
      走位/找掩体 · 最近的掩体
      依据：血 98%→74%
      快趴下！左边那堵断墙就是掩体。   ← 模型说的话（也就是语音内容）
- 语音放在决策时刻（--realistic-latency 则加上当次实测延迟）；上一句没念完顺延 0.3s；原声压 0.75、语音 1.6。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from coach.playbook import PRIORITY_CN                  # noqa: E402
from coach.state_provider import OracleGameState        # noqa: E402

# 录像不随仓库发布（内部对局画面），渲染只能在有录像的环境里跑
VIDEO_ROOT = Path(os.environ.get("COACH_VIDEO_ROOT", "private_data/videos"))
FONT_DIR = Path(os.environ.get("COACH_FONT_DIR", "assets/fonts"))
FFMPEG, FFPROBE = os.environ.get("FFMPEG", "ffmpeg"), os.environ.get("FFPROBE", "ffprobe")


def srt_t(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def probe_dur(p: Path) -> float:
    return float(subprocess.check_output([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]).decode().strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--slow", type=float, default=1.0,
                    help="放慢倍数。录像是 2 倍速，2 = 拉回真实对局速度；默认保持原速（1），放慢后帧重复看着卡")
    ap.add_argument("--pad", type=float, default=3.0, help="片段前后各留几录像秒")
    ap.add_argument("--realistic-latency", action="store_true")
    ap.add_argument("--no-subs", action="store_true")
    ap.add_argument("--max-size-mb", type=float, default=0,
                    help="成品体积上限（v2 render_video.py 的规矩是 20MB）：按时长换算码率封顶；0=不限")
    ap.add_argument("--scale", default="", help="输出分辨率，如 640:360（缩小后再烧字幕，字是清晰的）")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--font-size", type=int, default=15)
    ap.add_argument("--part", default="", help="K/N：把整段按时间等分成 N 份只渲第 K 份（发送上限 30MiB 时全场切两半）")
    ap.add_argument("--sub-box", action="store_true",
                    help="字幕加半透明底框（BorderStyle=4）：低码率下文字仍清楚，因为框内背景是静态的")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    meta = json.loads((a.run_dir / "meta.json").read_text(encoding="utf-8"))
    recs = [json.loads(l) for l in (a.run_dir / "speech.jsonl").open(encoding="utf-8")]
    recs = [r for r in recs if r.get("text") and r.get("wav")]
    if not recs:
        raise SystemExit("没有可渲染的话")
    g = OracleGameState(); rows = g.rows(meta["video"])
    t_from = meta["t_from"] if meta["t_from"] is not None else rows[0]["game_t"]
    t_to = meta["t_to"] if meta["t_to"] is not None else rows[-1]["game_t"]
    tv_from = next(r["t_video"] for r in rows if r["game_t"] >= t_from) - a.pad
    tv_to = max(r["t_video"] for r in rows if r["game_t"] <= t_to) + a.pad
    mp4 = next(iter((VIDEO_ROOT / meta["match"]).glob("*.mp4")))
    tv_from = max(0.0, tv_from); tv_to = min(probe_dur(mp4), tv_to)
    if a.part:
        k, n = (int(x) for x in a.part.split("/"))
        L = (tv_to - tv_from) / n
        tv_from, tv_to = tv_from + (k - 1) * L, tv_from + k * L
    seg_len = tv_to - tv_from
    out_len = seg_len * a.slow

    # 1. 排期（输出时间轴 = (t_video - tv_from) * slow）
    prev_end, kept, dropped = 0.0, [], 0
    for r in recs:
        start = (r["t_video"] - tv_from) * a.slow + ((r.get("latency") or 0) if a.realistic_latency else 0)
        if start < 0:
            continue                                    # 不在这一份里
        start = max(start, prev_end + 0.3)
        if start > out_len - 1.0:
            dropped += 1; continue
        dur = r["wav_dur"] or probe_dur(a.run_dir / r["wav"])
        r["start"], r["end"] = start, min(start + dur, out_len)
        prev_end = r["end"]; kept.append(r)
        print(f"  [{start:6.1f}s +{dur:.1f}s] {r['priority']}{'★' if r['must'] else ''} {r['action_cn']:<8} {r['text']}")
    if dropped:
        print(f"  （{dropped} 条超出片尾，丢弃）")

    # 2. 字幕
    srt = a.run_dir / ("coached.srt" if not a.part else f"coached_part{a.part.replace('/', 'of')}.srt")
    with srt.open("w", encoding="utf-8") as f:
        for i, r in enumerate(kept, 1):
            tag = ("★" if r["must"] else "") + ("⚡" if r.get("interrupted") else "")
            l1 = f'<font size="12">{tag}{r["priority"]}·{PRIORITY_CN[r["priority"]]}  {r["action_cn"]}</font>'
            l2 = f'<font size="12">{r["branch"]}' + (f' · {r["item"]}' if r.get("item") else "") + "</font>"
            l3 = f'<font size="11">依据：{r["reason"]}</font>'
            f.write(f"{i}\n{srt_t(r['start'])} --> {srt_t(r['end'] + 1.0)}\n{l1}\n{l2}\n{l3}\n{r['text']}\n\n")

    # 3. ffmpeg
    out_path = a.run_dir / (a.out or f"{meta['video']}_{t_from}-{t_to}_coached.mp4")
    inputs = ["-ss", f"{tv_from:.3f}", "-t", f"{seg_len:.3f}", "-i", str(mp4)]
    for r in kept:
        inputs += ["-i", str((a.run_dir / r["wav"]).resolve())]
    vf = f"[0:v]setpts={a.slow}*PTS"
    if a.scale:
        vf += f",scale={a.scale}"
    if not a.no_subs and FONT_DIR.is_dir():
        style = f"FontName=Noto Sans CJK SC,FontSize={a.font_size},Outline=1,MarginV=30,PrimaryColour=&H00FFFFFF,OutlineColour=&HA0000000"
        if a.sub_box:
            style += ",BorderStyle=4,BackColour=&H90000000,Outline=0,Shadow=0"
        vf += f",subtitles=filename={srt.resolve()}:fontsdir={FONT_DIR.resolve()}:force_style='{style}'"
    fparts = [vf + "[vout]", f"[0:a]atempo={1 / a.slow:.4f},volume=0.75[game]"]
    amix = "[game]"
    for i, r in enumerate(kept):
        ms = int(r["start"] * 1000)
        fparts.append(f"[{i + 1}:a]adelay={ms}|{ms},volume=1.6[t{i}]"); amix += f"[t{i}]"
    fparts.append(f"{amix}amix=inputs={len(kept) + 1}:duration=first:normalize=0[aout]")
    if a.max_size_mb > 0:                       # 体积封顶：留 7% 给容器开销（同 v2）
        a_kbps = 40
        v_kbps = int(max((a.max_size_mb * 8192 / out_len - a_kbps) * 0.93, 30))
        venc = ["-c:v", "libx264", "-preset", "slow", "-b:v", f"{v_kbps}k", "-maxrate", f"{v_kbps}k",
                "-bufsize", f"{2 * v_kbps}k"]
        aenc = ["-c:a", "aac", "-b:a", f"{a_kbps}k", "-ac", "1"]
        print(f"  体积上限 {a.max_size_mb}MB → 视频 {v_kbps}kbps + 音频 {a_kbps}kbps")
    else:
        venc = ["-c:v", "libx264", "-preset", "medium", "-crf", "24", "-maxrate", "2500k", "-bufsize", "5000k"]
        aenc = ["-c:a", "aac", "-b:a", "96k"]
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "warning"] + inputs + \
          ["-filter_complex", ";".join(fparts), "-map", "[vout]", "-map", "[aout]", "-r", str(a.fps)] + venc + aenc + \
          ["-movflags", "+faststart", str(out_path.resolve())]
    subprocess.run(cmd, check=True, cwd=a.run_dir)
    print(f"→ {out_path}  ({len(kept)} 句，{out_path.stat().st_size / 1e6:.1f}MB，{out_len:.0f}s，录像 {tv_from:.0f}-{tv_to:.0f}s)")


if __name__ == "__main__":
    main()
