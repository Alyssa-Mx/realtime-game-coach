#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""真流式回放：按真实墙钟节奏喂帧，感知与说话**并发**，端到端延迟全部实测。

与 replay_match.py 的区别（后者是批处理：先把整场感知跑完，再从头跑决策+说话）：
  · 帧按真实到达节奏进来（1 帧 = 2 对局秒 = 2 秒墙钟），不是想跑多快跑多快
  · 感知（GPU0）和说话（GPU3+4）真并发，互相抢 CPU / 磁盘 I/O —— 这才是上线时的样子
  · 说话人忙的时候新决策**丢弃**（真实教练不会把 5 秒前的建议堆着说），计数上报

    python tools/replay_live.py --video P1 --out outputs/live_P1 [--speed 1.0] [--limit N]

每句记录的时间戳（都是相对触发帧到达时刻的秒数）：
    lat_perc     感知读完这一帧（31 个框）
    lat_dir      Observer+Director 决策
    first_token  thinker 吐出第一个字
    first_audio  talker 吐出第一个音频块   ← 玩家真正听到声音的时刻
    lat_speak    整段语音收完
    play_start   实际开口时刻（上一句没说完要等）
"""
from __future__ import annotations
import argparse, base64, json, os, queue, sys, threading, time
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from coach.director import Director
from coach.events import EventDetector
from coach.playbook import ACTION_CN
from coach.speaker import OmniSpeaker, build_messages, write_wav
from coach.state_provider import DERIVED_DIR, OCR_ROOT, SELF_SLOT, _merge_zone_bearing

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'perception' / 'train'))
FRAMES = OCR_ROOT / "data" / "frames"


def frame_b64(match, t_video, max_w=640):
    p = FRAMES / match / f"t{t_video:07.1f}.jpg"
    img = cv2.imread(str(p))
    if img is None:
        return None
    if img.shape[1] > max_w:
        img = cv2.resize(img, (max_w, int(img.shape[0] * max_w / img.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="P1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed", type=float, default=1.0, help="1.0=真实时间；>1 加速（会放大争抢，只用于冒烟）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8324/v1/chat/completions")
    a = ap.parse_args()
    out = Path(a.out); (out / "wav").mkdir(parents=True, exist_ok=True)

    from infer_vllm import read_frame_parsed, to_gamestate, shutdown as _shutdown
    match = None
    for line in (OCR_ROOT / "data" / "gamestate_p1p8_v1.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        if d["video"] == a.video:
            match = d["match"]; break
    assert match
    gt = [json.loads(l) for l in (FRAMES.parent / "frames" / match / "gamestate.jsonl").open(encoding="utf-8")]
    gt.sort(key=lambda r: r["t_video"])
    if a.limit:
        gt = gt[:a.limit]
    t0_game = gt[0]["game_t"]

    # 圈方位来自离线派生（tools/derive_zone_bearing.py），不是每帧推理，所以不计入延迟。
    # 用一份"整场并好"的副本做查表，逐帧取，行为和批处理版一致。
    _bear = {r["t_video"]: r for r in [dict(x) for x in gt]}
    _rows = sorted(_bear.values(), key=lambda r: r["t_video"])
    _merge_zone_bearing(_rows, DERIVED_DIR / f"{a.video}_zone_bearing.jsonl")
    BEAR = {r["t_video"]: (r.get("zone_bearing"), r.get("zone_bearing_conf")) for r in _rows}

    spk = OmniSpeaker(model="qwen3omni-talker", voice="Chelsie", endpoint=a.endpoint)
    det, dr = EventDetector(), Director()
    fs = (out / "speech.jsonl").open("w", encoding="utf-8")
    fe = (out / "events.jsonl").open("w", encoding="utf-8")
    fp = (out / "perf.jsonl").open("w", encoding="utf-8")   # 每帧都记：感知耗时、是否跟得上、说话人当时忙不忙

    jobs: queue.Queue = queue.Queue(maxsize=1)
    state = {"busy": False, "play_end": 0.0, "n": 0, "dropped": 0, "dropped_must": 0, "err": 0}
    lock = threading.Lock()
    recent: list[dict] = []

    def speaker_loop():
        while True:
            job = jobs.get()
            if job is None:
                break
            c, ctx, frames, t_game, rec = job
            t_req = time.time()
            res = spk.say(build_messages(c, ctx, frames, recent, t_game))
            now = time.time()
            fa = res.get("first_audio")
            heard = t_req + fa if fa else now                  # 第一个音能播出来的时刻
            with lock:
                play = max(heard, state["play_end"])            # 上一句没说完就得等
                dur = 0.0
                if res.get("pcm"):
                    w = out / "wav" / f"u{rec['i']:03d}.wav"
                    dur = write_wav(res["pcm"], w); rec["wav"] = w.name
                state["play_end"] = play + dur
                if res.get("text"):
                    recent.append({"t": t_game, "text": res["text"]})
                if res.get("error"):
                    state["err"] += 1
            rec.update({"text": res["text"], "wav_dur": round(dur, 2), "error": res["error"],
                        "lat_queue": round(t_req - rec["_t_dec"], 3),
                        "first_token": res.get("first_token"), "first_audio": fa,
                        "lat_speak": round(now - t_req, 3),
                        "e2e_heard": round(heard - rec["_t_arrive"], 3),
                        "e2e_play": round(play - rec["_t_arrive"], 3),
                        "wait_prev": round(play - heard, 3)})
            for k in ("_t_arrive", "_t_dec"):
                rec.pop(k, None)
            fs.write(json.dumps(rec, ensure_ascii=False) + "\n"); fs.flush()
            print(f"  ↳ [{t_game//60:02d}:{t_game%60:02d}] 听到+{rec['e2e_heard']:.1f}s 开口+{rec['e2e_play']:.1f}s "
                  f"({rec['wav_dur']}s) {res['text']!r}", flush=True)
            with lock:
                state["busy"] = False
            jobs.task_done()

    th = threading.Thread(target=speaker_loop, daemon=True); th.start()
    wall0 = time.time()
    print(f"== 真流式 {a.video}（{len(gt)} 帧，1 帧=2 对局秒，speed={a.speed}）预计 {len(gt)*2/a.speed/60:.0f} 分钟", flush=True)

    for r in gt:
        due = wall0 + (r["game_t"] - t0_game) / a.speed
        d = due - time.time()
        if d > 0:
            time.sleep(d)                                    # 帧按真实节奏到达
        t_arrive = time.time()
        fr = cv2.imread(str(FRAMES / match / f"t{r['t_video']:07.1f}.jpg"))
        if fr is None:
            continue
        row = to_gamestate(read_frame_parsed(fr, match), game_t=r["game_t"], t_video=r["t_video"])
        row["video"], row["match"] = a.video, match
        if row.get("energy_fit") is None and row.get("energy") is not None:
            row["energy_fit"] = row["energy"]
        row["self_slot"] = SELF_SLOT.get(a.video)
        row["zone_bearing"], row["zone_bearing_conf"] = BEAR.get(r["t_video"], (None, None))
        t_perc = time.time()
        with lock:
            _busy_now = state["busy"]
        fp.write(json.dumps({"game_t": r["game_t"], "lat_perc": round(t_perc - t_arrive, 3),
                             "lag": round(t_arrive - due, 3),          # >0 = 没按时开始处理，掉队了
                             "speaking": _busy_now}, ensure_ascii=False) + "\n")
        evs = det.update(row)
        for e in evs:
            fe.write(json.dumps(e.as_dict(), ensure_ascii=False) + "\n")
        dec = dr.step(r["game_t"], r["t_video"], evs, det.ctx)
        t_dec = time.time()
        if not dec.speak:
            continue
        c = dec.must or dec.top3[0]
        with lock:
            busy = state["busy"]
            if busy:
                state["dropped"] += 1
                state["dropped_must"] += bool(dec.must)
            else:
                state["busy"] = True
                i = state["n"]; state["n"] += 1
        if busy:
            print(f"[{r['game_t']//60:02d}:{r['game_t']%60:02d}] ✗丢弃(说话人忙) {ACTION_CN.get(c['action'],c['action'])}", flush=True)
            continue
        frames = [((a.n_frames - 1 - k) * 2.0, b) for k in range(a.n_frames - 1, -1, -1)
                  if (b := frame_b64(match, r["t_video"] - k))]
        rec = {"i": i, "t": r["game_t"], "t_video": r["t_video"], "video": a.video, "match": match,
               "action": c["action"], "action_cn": ACTION_CN.get(c["action"], c["action"]), "priority": c["priority"],
               "must": bool(dec.must), "interrupted": dec.interrupted, "branch": c["routes"][0]["branch"],
               "item": c["routes"][0]["item"], "reason": c["reason"], "hint": c["hint"], "emotional": c["emotional"],
               "lat_perc": round(t_perc - t_arrive, 3), "lat_dir": round(t_dec - t_perc, 3),
               "wav": None, "_t_arrive": t_arrive, "_t_dec": t_dec}
        print(f"[{r['game_t']//60:02d}:{r['game_t']%60:02d}] {c['priority']}{'★' if dec.must else ''} "
              f"{rec['action_cn']:<8} 感知{rec['lat_perc']:.2f}s → 送说话人", flush=True)
        jobs.put((c, det.ctx, frames, r["game_t"], rec))

    jobs.join(); jobs.put(None); th.join(timeout=10)
    fs.close(); fe.close(); fp.close()
    (out / "meta.json").write_text(json.dumps(
        {"video": a.video, "match": match, "mode": "live-stream", "speed": a.speed, "frames": len(gt),
         "spoken": state["n"], "dropped_busy": state["dropped"], "dropped_must": state["dropped_must"],
         "errors": state["err"], "wall_s": round(time.time() - wall0, 1)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"== 开口 {state['n']} 次，因说话人忙丢弃 {state['dropped']} 次（其中 P0 {state['dropped_must']}），"
          f"失败 {state['err']}，墙钟 {(time.time()-wall0)/60:.1f} 分钟", flush=True)
    _shutdown(exit_process=True)


if __name__ == "__main__":
    main()
