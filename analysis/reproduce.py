#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 data/ 重算 README 与 docs 里引用的全部数字。只用标准库，不需要 GPU、不需要原始录像。

    python analysis/reproduce.py              # 全部
    python analysis/reproduce.py perception   # 只看一节：perception / rules / speaker / live
"""
from __future__ import annotations

import ast
import collections
import csv
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SQUAD = ["P1", "P5", "P6", "P7", "P8"]


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def table(p: Path) -> dict[str, dict]:
    with p.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    key = list(rows[0].keys())[0]
    return {r[key]: r for r in rows}


def h(title: str):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def med(xs):
    return st.median(xs) if xs else float("nan")


def event_types() -> set[str]:
    """静态枚举 coach/events.py 能产出的全部事件类型：E(...)/Event(...) 的首参 + 赋给 name 的常量 + 跑毒三档。"""
    src = (ROOT / "coach/events.py").read_text(encoding="utf-8")

    def consts(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            return {n.value}
        if isinstance(n, ast.IfExp):
            return consts(n.body) | consts(n.orelse)
        if isinstance(n, ast.JoinedStr) and ast.unparse(n).startswith("f'ZONE_OUTSIDE_"):
            return {"ZONE_OUTSIDE_URGENT", "ZONE_OUTSIDE_TIGHT", "ZONE_OUTSIDE_PLAN"}
        return set()

    out = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("E", "Event") and node.args:
            out |= consts(node.args[0])
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "name" for t in node.targets):
            out |= consts(node.value)
    return out


# ------------------------------------------------------------------ ⑦ 感知
def perception():
    h("⑦ 读仪表盘的模型（Qwen3-VL-8B + LoRA）")
    pool = json.loads((DATA / "perception/train_pool_v7.json").read_text(encoding="utf-8"))
    print(f"v7 训练池：人审 {len(pool['fine_matches'])} 场 {pool['fine_frames']:,} 帧 + 粗标 {pool['coarse_matches']} 场 "
          f"{pool['coarse_frames']:,} 帧；{pool['steps']} 步；留出场 {'/'.join(pool['held_out'])} 不进训练")

    acc = table(DATA / "perception/field_accuracy_P8.csv")
    fields = [k for k in acc if k not in ("MEAN21", "FRAME_EXACT")]
    assert len(fields) == 21
    for v in ("v4", "v5", "v6", "v7"):
        mean = sum(float(acc[f][v]) for f in fields) / 21
        print(f"  P8 留出场 {v}: 21 字段均值 {mean:.4f}（报告 {acc['MEAN21'][v]}）  帧全对率 {acc['FRAME_EXACT'][v]}")
    print(f"  同一份 v7 权重、推理时按老口径全部 ×3 裁：{acc['MEAN21']['v7_scale_x3']}"
          f"（主武器 {acc['weapon_slot1']['v7']} → {acc['weapon_slot1']['v7_scale_x3']}）")
    weak = [f for f in fields if acc[f]["zero_shot"] and float(acc[f]["zero_shot"]) < 0.35]
    print(f"  零样本 <0.35 的字段（P8 分场）：{', '.join(weak)}")
    for f in ("hp", "signal", "armor", "weapon_slot1", "weapon_slot2"):
        print(f"    {f:<13} 零样本 {acc[f]['zero_shot']:>5} → v7 {acc[f]['v7']}")

    t = table(DATA / "perception/tiers_P7.csv")
    fr = table(DATA / "perception/tiers_P7_frame.csv")
    s1, s4 = float(t["MEAN21"]["S1_crop"]), float(t["MEAN21"]["S4_whole"])
    sp1, sp4 = float(fr["S1"]["sec_per_frame_vllm"]), float(fr["S4"]["sec_per_frame_vllm"])
    print(f"\n切图 vs 整图（v3 四档都训过，P7 215 帧配对，同一个 vLLM 进程）")
    print(f"  切图 S1：{sp1:.3f} s/帧，21 字段 {s1:.3f}，帧全对 {fr['S1']['frame_exact']}，{fr['S1']['requests']} 个请求并行")
    print(f"  整图 S4：{sp4:.3f} s/帧，21 字段 {s4:.3f}，帧全对 {fr['S4']['frame_exact']}，{fr['S4']['requests']} 个请求")
    print(f"  → 整图慢 {sp4 / sp1:.2f}×，准确率低 {100 * (s1 - s4):.1f}pp")
    gap = sorted(((float(t[f]['S1_crop']) - float(t[f]['S4_whole']), f) for f in t if f != "MEAN21"), reverse=True)[:4]
    print("  差距最大的字段：" + "、".join(f"{f} {t[f]['S1_crop']}→{t[f]['S4_whole']}" for _, f in gap))

    lat = list(csv.DictReader((DATA / "perception/latency_steps.csv").open(encoding="utf-8")))
    base = float(lat[0]["sec_per_frame"])
    print("\n读一帧（31 个字段，单张 H20）")
    for r in lat:
        s = float(r["sec_per_frame"])
        print(f"  {r['step']:<28} {s:>6.3f} s   累计 {base / s:.2f}×")


# ------------------------------------------------------------------ ⑥ 规则层（记录员 + 编导）
def rules():
    h("⑥ 记录员（Observer）+ 编导（Director）：纯规则")
    types = event_types()
    print(f"events.py 静态枚举的事件类型：{len(types)} 类（跑毒按 来不及 / 偏紧 / 该规划 三档计 3 类）")
    print("  " + " ".join(sorted(types)))
    ev_types = set()
    for v in SQUAD:
        ev_types |= {e["type"] for e in jsonl(DATA / f"coach/offline_final/{v}_events.jsonl")}
    ev_types |= {e["type"] for e in jsonl(DATA / "coach/live_P1/events.jsonl")}
    print(f"五场离线 + 实跑里实际触发过 {len(ev_types)} 类")

    summ = json.loads((DATA / "coach/offline_final/summary.json").read_text(encoding="utf-8"))
    tot = collections.Counter()
    print("\n最终规则 × 人审读数，五场四排整场离线回放（不调任何模型）")
    print(f"  {'场':<4}{'分钟':>6}{'事件':>6}{'开口':>6}{'次/分':>7}{'P0':>5}{'P1':>5}{'P2':>5}{'P3':>5}{'打断':>6}{'过期':>6}")
    for s in summ:
        p = s["by_priority"]
        ne = sum(s["events"].values())
        exp = sum(s["stats"]["expired"].values())
        tot.update({"min": s["dur_min"], "ev": ne, "spk": s["spoken"], "p0": p.get("P0", 0), "p1": p.get("P1", 0),
                    "p2": p.get("P2", 0), "p3": p.get("P3", 0), "int": s["stats"]["interrupts"], "exp": exp,
                    "cool": s["stats"]["cooldown_blocked"], "intent": s["stats"].get("intent_blocked", 0),
                    "p3busy": s["stats"]["p3_dropped_busy"]})
        print(f"  {s['video']:<4}{s['dur_min']:>6.1f}{ne:>6}{s['spoken']:>6}{s['spoken'] / s['dur_min']:>7.2f}"
              f"{p.get('P0', 0):>5}{p.get('P1', 0):>5}{p.get('P2', 0):>5}{p.get('P3', 0):>5}{s['stats']['interrupts']:>6}{exp:>6}")
    print(f"  合计 {tot['min']:.0f} 分钟：事件 {tot['ev']}，开口 {tot['spk']}（{tot['spk'] / tot['min']:.2f} 次/分），"
          f"P0/P1/P2/P3 = {tot['p0']}/{tot['p1']}/{tot['p2']}/{tot['p3']}，P0 打断 {tot['int']} 次，"
          f"过期丢弃 {tot['exp']}，冷却拦下 {tot['cool']}，意图间隔拦下 {tot['intent']}，P3 忙时丢 {tot['p3busy']}")
    print(f"  事件 → 开口的压缩比：{tot['ev'] / tot['spk']:.1f} 件事说 1 句")


# ------------------------------------------------------------------ ⑥ 说话人
def speaker():
    h("⑥ 说话人（Qwen3-Omni thinker + talker，vLLM-Omni 本地部署，全部本地推理）")
    allv = []
    for v in SQUAD:
        r = jsonl(DATA / f"coach/speaker_local/{v}_speech.jsonl")
        lat = [x["latency"] for x in r if x.get("latency") is not None]
        allv += lat
        print(f"  {v}: {len(r)} 句，每句（含语音）中位 {med(lat):.2f} s，最大 {max(lat):.1f} s，"
              f"失败 {sum(bool(x.get('error')) for x in r)}，缺语音 {sum(not x.get('wav_dur') for x in r)}，"
              f"方向词校验介入 {sum(bool(x.get('dir_check')) for x in r)}")
    print(f"  合计 {len(allv)} 句：中位 {med(allv):.2f} s，p90 {sorted(allv)[int(0.9 * len(allv))]:.2f} s")


# ------------------------------------------------------------------ 真流式整场
def live():
    h("真流式整场：v7 实时读数 → 规则 → 本地 Omni（P1，墙钟 30 分钟）")
    meta = json.loads((DATA / "coach/live_P1/meta.json").read_text(encoding="utf-8"))
    perf = jsonl(DATA / "coach/live_P1/perf.jsonl")
    sp = jsonl(DATA / "coach/live_P1/speech.jsonl")
    ev = jsonl(DATA / "coach/live_P1/events.jsonl")
    lp = [p["lat_perc"] for p in perf]
    print(f"  {meta['frames']} 帧（1 帧 = 2 游戏秒），{len(ev)} 个事件，开口 {meta['spoken']} 句；"
          f"说话人忙而丢弃 {meta['dropped_busy']}，失败 {meta['errors']}")
    print(f"  感知每帧：中位 {med(lp):.2f} s，p90 {sorted(lp)[int(0.9 * len(lp))]:.2f} s（与说话人同机并发；单卡独占时 0.93 s）")
    print(f"  掉队（帧到了没按时开始处理 >0.5 s）：{sum(p['lag'] > 0.5 for p in perf)} / {len(perf)} 帧")
    for k, name in (("lat_perc", "感知"), ("lat_dir", "规则决策"), ("first_token", "说话人首字"),
                    ("first_audio", "说话人首个语音块"), ("e2e_heard", "画面到达 → 玩家听到")):
        v = [x[k] for x in sp if x.get(k) is not None]
        print(f"  {name:<18} 中位 {med(v):.3f} s   最大 {max(v):.3f} s")
    print("  优先级分布：" + "，".join(f"{k} {n}" for k, n in sorted(collections.Counter(x['priority'] for x in sp).items())))


SECTIONS = {"perception": perception, "rules": rules, "speaker": speaker, "live": live}

if __name__ == "__main__":
    for name in (sys.argv[1:] or SECTIONS):
        SECTIONS[name]()
