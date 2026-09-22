#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线回放一场（或一段）：Oracle gamestate → Observer → Director → Speaker，产出可渲染的 speech.jsonl。

    python tools/replay_match.py --video P1 --from 600 --to 820 --backend omni_vllm --out outputs/replay_P1_a
    python tools/replay_match.py --video P1 --backend none        # 只跑决策，不调模型（干跑）

时间参数 --from/--to 是**对局秒 game_t**。画面用标注阶段抽好的帧（1 录像秒一张 = 2 对局秒），
每次开口取决策时刻往前 3 张 + 当前 1 张，时间戳按对局秒写 <0.0/2.0/4.0/6.0 seconds>。
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from coach.director import Director                      # noqa: E402
from coach.events import EventDetector                   # noqa: E402
from coach.playbook import ACTION_CN                     # noqa: E402
from coach.speaker import LocalVLSpeaker, OmniLocalSpeaker, OmniSpeaker, build_messages, check_direction, check_persona, echoes_hint, strip_direction, write_wav   # noqa: E402
from coach.state_provider import OCR_ROOT, OracleGameState   # noqa: E402

FRAMES_ROOT = OCR_ROOT / "data" / "frames"


def frame_b64(match: str, t_video: float, max_w: int) -> str | None:
    p = FRAMES_ROOT / match / f"t{t_video:07.1f}.jpg"
    if not p.is_file():
        return None
    img = cv2.imread(str(p))
    if img is None:
        return None
    if img.shape[1] > max_w:
        h = int(img.shape[0] * max_w / img.shape[1])
        img = cv2.resize(img, (max_w, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="P1")
    ap.add_argument("--from", dest="t_from", type=int, default=None, help="对局秒")
    ap.add_argument("--to", dest="t_to", type=int, default=None)
    ap.add_argument("--backend", choices=["local", "omni_local", "omni_vllm", "none"], default="omni_vllm",
                    help="omni_vllm=本地 vllm-omni（文字+语音，OpenAI 协议）；local=本地 OpenAI 兼容只出文字；omni_local=transformers 全模型服务；none=只跑规则")
    ap.add_argument("--model", default=None)
    ap.add_argument("--backend-url", default="http://127.0.0.1:8321", help="local 后端的 OpenAI 兼容地址（8321=Qwen3-VL-8B，8322=Qwen3-Omni Thinker）")
    ap.add_argument("--voice", default="Chelsie")
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--max-w", type=int, default=640)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="最多开口多少次（试跑用）")
    a = ap.parse_args()

    g = OracleGameState()
    rows = g.rows(a.video)
    match = g.match_id(rows)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.backend == "local":
        spk = LocalVLSpeaker(base_url=a.backend_url, model=a.model or "qwen3vl8b")
    elif a.backend == "omni_vllm":
        spk = OmniSpeaker(model=a.model or "qwen3omni-talker", voice=a.voice,
                          endpoint=(a.backend_url if a.backend_url != "http://127.0.0.1:8321" else "http://127.0.0.1:8324") + "/v1/chat/completions")
    elif a.backend == "omni_local":
        spk = OmniLocalSpeaker(base_url=a.backend_url if a.backend_url != "http://127.0.0.1:8321" else "http://127.0.0.1:8323", speaker=a.voice)
    else:
        spk = None

    det, dr = EventDetector(), Director()
    fs = (out / "speech.jsonl").open("w", encoding="utf-8")
    fd = (out / "decisions.jsonl").open("w", encoding="utf-8")
    fe = (out / "events.jsonl").open("w", encoding="utf-8")
    recent: list[dict] = []
    n_spoken, n_err, t_start = 0, 0, time.time()
    for r in rows:
        t = r["game_t"]
        evs = det.update(r)
        for e in evs:
            fe.write(json.dumps(e.as_dict(), ensure_ascii=False) + "\n")
        d = dr.step(t, r["t_video"], evs, det.ctx)
        if a.t_from is not None and t < a.t_from:
            continue                        # 决策层照常运转（历史/冷却要热身），只是不开口
        if a.t_to is not None and t > a.t_to:
            break
        if not d.speak:
            continue
        c = d.must or d.top3[0]
        fd.write(json.dumps(d.as_dict(), ensure_ascii=False) + "\n")
        rec = {"i": n_spoken, "t": t, "t_video": r["t_video"], "video": a.video, "match": match,
               "action": c["action"], "action_cn": ACTION_CN.get(c["action"], c["action"]), "priority": c["priority"],
               "must": bool(d.must), "interrupted": d.interrupted, "branch": c["routes"][0]["branch"],
               "item": c["routes"][0]["item"], "reason": c["reason"], "hint": c["hint"], "emotional": c["emotional"],
               "text": "", "wav": None, "wav_dur": None, "latency": None, "usage": None, "error": None}
        if spk is not None:
            frames = []
            for k in range(a.n_frames - 1, -1, -1):
                tv = r["t_video"] - k            # 1 录像秒一张 = 2 对局秒
                b = frame_b64(match, tv, a.max_w)
                if b:
                    frames.append(((a.n_frames - 1 - k) * 2.0, b))
            msgs = build_messages(c, det.ctx, frames, recent, t)
            res = spk.say(msgs)
            # ---- 方向词校验：prompt 禁不住，代码把关（一次重生 + 兜底删词）----
            # 方位已知：安全区方位读到了，或这条候选自带方位（队友标记消息里的罗盘读数，依据里有"罗盘"）
            zone_known = det.ctx.get("zone_bearing") is not None or ("罗盘" in (c.get("reason") or ""))
            goal_for_check = "zone" if ("罗盘" in (c.get("reason") or "")) else c["goal"]
            marker_known = "罗盘" in (c.get("reason") or "") and c["goal"] in ("info", "threat")
            ok, why = check_direction(res["text"], goal_for_check, zone_known, marker_known) if res["text"] else (True, "")
            if ok and res["text"]:
                ok, why = check_persona(res["text"])
            if ok and res["text"] and echoes_hint(res["text"], c.get("hint") or ""):
                ok, why = False, "把提示词原样念出来了"
            rec["dir_check"] = None
            if not ok:
                first = res["text"]
                msgs2 = build_messages(c, det.ctx, frames, recent, t)
                msgs2[-1]["content"].append({"type": "text", "text": f"上一句「{first}」被驳回：{why}。重写一句：" + ("你不在游戏里，只用'你'对玩家说，队友的动作说'让 N 号…'。" if "队友了" in why else
              "提示是给你的指导，不是台词，用自己的话说。" if "提示词" in why else
              "除了画面里点名的物体，不要用任何方向词。")})
                res2 = spk.say(msgs2)
                ok2, why2 = check_direction(res2["text"], goal_for_check, zone_known, marker_known) if res2["text"] else (False, "空")
                if ok2:
                    ok2, why2 = check_persona(res2["text"])
                if ok2 and echoes_hint(res2["text"], c.get("hint") or ""):
                    ok2, why2 = False, "把提示词原样念出来了"
                if ok2:
                    res = res2; rec["dir_check"] = {"first": first, "why": why, "fix": "regen"}
                else:
                    fixed = strip_direction(res2["text"] or first)
                    rec["dir_check"] = {"first": first, "why": why, "second": res2["text"], "fix": "strip"}
                    # 兜底删词后文字变了，语音得重配：再要一次纯语音（把改好的句子当任务让它照念）
                    res3 = spk.say([{"role": "system", "content": "你是配音员。只把用户给的这句话原样念出来，不增不减。"},
                                    {"role": "user", "content": [{"type": "text", "text": fixed}]}])
                    res = {**res3, "text": fixed if not res3.get("error") else fixed, "latency": (res["latency"] or 0) + (res2["latency"] or 0) + (res3["latency"] or 0)}
                print(f"      ↳ 方向校验未过（{why}）：「{first}」→「{res['text']}」", flush=True)
            # 直播延迟 = 这一帧的感知耗时 + 说话人耗时（thinker 出字 + talker 出音）。
            # lat_perc 由 gen_gamestate_model.sh 逐帧实测写入；人工 gamestate 没有这个字段，退化成只算说话人。
            lp = r.get("lat_perc")
            rec.update({"text": res["text"], "lat_perc": lp, "lat_speak": res["latency"],
                        "latency": round((lp or 0) + (res["latency"] or 0), 3),
                        "first_token": res.get("first_token"),
                        "usage": res["usage"], "error": res["error"], "n_frames": len(frames)})
            if res["pcm"]:
                wav = out / "wav" / f"u{n_spoken:03d}.wav"
                rec["wav"] = str(wav.relative_to(out)); rec["wav_dur"] = round(write_wav(res["pcm"], wav), 2)
            if res["error"]:
                n_err += 1
            if res["text"]:
                recent.append({"t": t, "text": res["text"]})
            print(f"[{t // 60:02d}:{t % 60:02d}] {c['priority']}{'★' if d.must else ''} {rec['action_cn']:<8} "
                  f"{res['text']!r}  ({(res['latency'] or 0):.1f}s, wav {rec['wav_dur']}s){'  ERR ' + res['error'] if res['error'] else ''}", flush=True)
        else:
            print(f"[{t // 60:02d}:{t % 60:02d}] {c['priority']}{'★' if d.must else ''} {rec['action_cn']:<8} {c['reason']}", flush=True)
        fs.write(json.dumps(rec, ensure_ascii=False) + "\n"); fs.flush()
        n_spoken += 1
        if a.limit and n_spoken >= a.limit:
            break
        if n_err >= 5:
            print("连续失败过多，中止", flush=True); break
    # 对局结束：吃鸡判定（不再靠 alive==1）
    for e in det.finish():
        fe.write(json.dumps(e.as_dict(), ensure_ascii=False) + "\n")
        d = dr.step(e.t, e.t_video, [e], det.ctx)
        if d.speak and spk is not None and not (a.t_to is not None and e.t > a.t_to):
            c = d.must or d.top3[0]; r = rows[-1]
            frames = [((a.n_frames - 1 - k) * 2.0, b) for k in range(a.n_frames - 1, -1, -1) if (b := frame_b64(match, r["t_video"] - k, a.max_w))]
            res = spk.say(build_messages(c, det.ctx, frames, recent, e.t))
            rec = {"i": n_spoken, "t": e.t, "t_video": r["t_video"], "video": a.video, "match": match, "action": c["action"],
                   "action_cn": ACTION_CN.get(c["action"], c["action"]), "priority": c["priority"], "must": bool(d.must), "interrupted": False,
                   "branch": c["routes"][0]["branch"], "item": c["routes"][0]["item"], "reason": c["reason"], "hint": c["hint"],
                   "emotional": c["emotional"], "text": res["text"], "wav": None, "wav_dur": None,
                   "lat_perc": r.get("lat_perc"), "lat_speak": res["latency"],
                   "latency": round((r.get("lat_perc") or 0) + (res["latency"] or 0), 3), "usage": res["usage"], "error": res["error"]}
            if res["pcm"]:
                wav = out / "wav" / f"u{n_spoken:03d}.wav"; rec["wav"] = str(wav.relative_to(out)); rec["wav_dur"] = round(write_wav(res["pcm"], wav), 2)
            fs.write(json.dumps(rec, ensure_ascii=False) + "\n"); n_spoken += 1
            print(f"[{e.t // 60:02d}:{e.t % 60:02d}] {c['priority']} {rec['action_cn']:<8} {res['text']!r}", flush=True)
    fs.close(); fd.close(); fe.close()
    (out / "meta.json").write_text(json.dumps({"video": a.video, "match": match, "t_from": a.t_from, "t_to": a.t_to,
                                               "backend": a.backend, "model": a.model, "voice": a.voice, "n_frames": a.n_frames,
                                               "max_w": a.max_w, "spoken": n_spoken, "errors": n_err,
                                               "wall_s": round(time.time() - t_start, 1)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"== 开口 {n_spoken} 次，失败 {n_err}，用时 {time.time() - t_start:.0f}s → {out}")


if __name__ == "__main__":
    main()
