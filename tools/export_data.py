#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把内部实验产物导出成仓库里 data/ 下的脱敏数据（本仓库所有数字都从 data/ 现算）。

原始产物（录像、逐帧读数全集、权重、场次 ID）不随仓库发布，所以这个脚本只能在有原始产物的机器上跑；
路径全部由参数给，不写死。导出时做四件事：
  1. 去掉场次 ID、录像文件名、wav 路径这类内部标识，只留 P1–P8 代号；
  2. 评测报告（Markdown 表）解析成 CSV，数字原样搬运、不重算；
  3. 教练的逐句 / 逐事件 / 逐帧日志只保留分析和 demo 要用的字段；
  4. 文本字段按 --scrub-file 的替换表统一替换（决策框架原件是同事整理的、不公开，条目名换成转述；
     另有少量内部标识），替换表不随仓库发布。

    python tools/export_data.py \
        --ocr-eval  <感知仓库>/eval  --ocr-pool <感知仓库>/data/sft/v7 \
        --offline   <run_offline.py 的输出目录>  --o3-root <replay_<P>O3_full 所在目录> \
        --live      <replay_live.py 的输出目录>  --readings <live 那场的模型逐帧读数 jsonl> \
        --scrub-file <替换表 json> --out data
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

SQUAD = ["P1", "P5", "P6", "P7", "P8"]          # 四排局（P2–P4 是单排，不测教练）
DROP_KEYS = {"match", "wav", "video_file", "usage"}
SCRUB: dict[str, str] = {}             # 由 --scrub-file 读入：{原文: 替换成}


def scrub(x):
    if isinstance(x, str):
        for a in sorted(SCRUB, key=len, reverse=True):          # 长的先换，避免短词截断长词
            x = x.replace(a, SCRUB[a])
        return x
    if isinstance(x, list):
        return [scrub(v) for v in x]
    if isinstance(x, dict):
        return {scrub(k): scrub(v) for k, v in x.items()}
    return x


def md_table(text: str, header_first: str) -> list[list[str]]:
    """取第一张首列表头为 header_first 的 Markdown 表，返回数据行（去掉 ** 和空白）。"""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if ln.lstrip().startswith("|") and cells and cells[0] == header_first:
            rows = []
            for ln2 in lines[i + 2:]:
                if not ln2.lstrip().startswith("|"):
                    break
                rows.append([c.strip().replace("**", "") for c in ln2.strip().strip("|").split("|")])
            return [cells] + rows
    raise ValueError(f"找不到表头 {header_first!r}")


def write_csv(path: Path, rows: list[list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def clean(rec: dict) -> dict:
    return scrub({k: v for k, v in rec.items() if k not in DROP_KEYS and not k.startswith("_")})


def jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def dump_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 感知
def export_perception(ev: Path, pool_dir: Path, out: Path):
    # 1) P8 留出场逐字段：v4–v7 同一批标签（09-05 修标后），零样本取报告里的 P8 分场列
    t = md_table((ev / "v7_final_P8.md").read_text(encoding="utf-8"), "字段")
    zs = md_table((ev / "zero_shot_20260903.md").read_text(encoding="utf-8"), "任务")
    zs_p8 = {}
    for r in zs[1:]:
        m = re.search(r"P8:([0-9.]+)", r[-1])
        zs_p8[r[0]] = m.group(1) if m else ""
    rows = [["field", "zero_shot", "v4", "v5", "v6", "v7", "v7_scale_x3", "n"]]
    for r in t[1:]:
        f = r[0]
        if f.startswith("21 字段均值") or f.startswith("帧全对率"):
            rows.append([{"21 字段均值": "MEAN21", "帧全对率": "FRAME_EXACT"}[f], ""] + r[1:6] + [""])
        else:
            rows.append([f, zs_p8.get(f, "")] + r[1:7])
    write_csv(out / "field_accuracy_P8.csv", rows)

    # 2) 切图 vs 整图（v3，唯一四档都训过的版本；P7 215 帧配对）
    t = md_table((ev / "v3_val_final.md").read_text(encoding="utf-8"), "字段")
    rows = [["field", "S1_crop", "S2_patch", "S3_region", "S4_whole", "n"]]
    for r in t[1:]:
        rows.append(["MEAN21" if r[0].startswith("21 字段均值") else r[0]] + r[1:6] + ([] if len(r) > 5 else [""]))
    write_csv(out / "tiers_P7.csv", rows)
    fr = md_table((ev / "v3_val_final.md").read_text(encoding="utf-8"), "档")
    wc = md_table((ev / "whole_vs_crop.md").read_text(encoding="utf-8"), "输入方式")
    speed = {("S1" if "S1" in r[0] else "S4"): r for r in wc[1:]}
    rows = [["tier", "frames", "frame_exact", "sec_per_frame_vllm", "requests", "gen_tokens"]]
    for r in fr[1:]:
        s = speed.get(r[0])
        rows.append([r[0], r[1], r[2]] + ([s[1], s[2], s[3]] if s else ["", "", ""]))
    write_csv(out / "tiers_P7_frame.csv", rows)

    # 3) 读一帧（31 个字段）的耗时：每步都在 P8 上核对过精度
    lat = (ev / "latency_v7.md").read_text(encoding="utf-8")
    t = md_table(lat, "档")
    rows = [["step", "sec_per_frame", "engine", "note"]]
    for r in t[1:]:
        rows.append([r[0], r[1], "HF transformers", r[2]])
    v = md_table(lat, "")                                  # vLLM 实测表（首列表头为空）
    for r in v[1:]:
        if r[0].startswith("单帧"):
            rows.append(["F vLLM（merge 后权重）", re.sub(r"[^0-9.]", "", r[2]), "vLLM 0.19.1", "vs C " + r[3]])
    write_csv(out / "latency_steps.csv", rows)

    # 4) v7 训练池：人审帧 / 粗标帧 / 粗标场数（只数不列 ID）
    pool = json.loads((pool_dir / "pool.json").read_text(encoding="utf-8"))
    coarse = set()
    for ln in (pool_dir / "plan.jsonl").open(encoding="utf-8"):
        d = json.loads(ln)
        if d.get("q") == "coarse":
            coarse.update(m for m, _ in d["ids"])
    (out / "train_pool_v7.json").write_text(json.dumps({
        "fine_frames": pool["fine_frames"], "coarse_frames": pool["coarse_frames"],
        "coarse_matches": len(coarse), "fine_matches": ["P1", "P2", "P3", "P5", "P6", "P7"],
        "held_out": pool["val"], "steps": sum(1 for _ in (pool_dir / "plan.jsonl").open()),
        "field_weights": pool["weights"]}, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 教练
def export_coach(offline: Path, o3_root: Path, live: Path, readings: Path, out: Path):
    # 1) 最终规则（纯规则、人审读数）离线整场：每场事件 + 每次开口 / 丢弃 / 打断
    summ = json.loads((offline / "summary.json").read_text(encoding="utf-8"))
    for s in summ:
        v = s["video"]
        dump_jsonl(out / "offline_final" / f"{v}_events.jsonl", [clean(e) for e in jsonl(offline / v / "events.jsonl")])
        dump_jsonl(out / "offline_final" / f"{v}_decisions.jsonl", [clean(d) for d in jsonl(offline / v / "decisions.jsonl")])
    (out / "offline_final" / "summary.json").write_text(json.dumps(scrub(summ), ensure_ascii=False, indent=1), encoding="utf-8")

    # 2) 全本地说话人（vLLM-Omni，Qwen3-Omni thinker + talker）：五场逐句耗时
    for v in SQUAD:
        rows = [clean(r) for r in jsonl(o3_root / f"replay_{v}O3_full" / "speech.jsonl")]
        dump_jsonl(out / "speaker_local" / f"{v}_speech.jsonl", rows)

    # 3) 真流式整场（v7 感知实时读数 → 规则 → 本地 Omni，墙钟 30 分钟）
    d = out / "live_P1"
    dump_jsonl(d / "speech.jsonl", [clean(r) for r in jsonl(live / "speech.jsonl")])
    dump_jsonl(d / "events.jsonl", [clean(r) for r in jsonl(live / "events.jsonl")])
    dump_jsonl(d / "perf.jsonl", jsonl(live / "perf.jsonl"))
    keep = ("game_t", "t_video", "alive", "hp", "signal", "energy", "zone_countdown", "zone_dist_m", "in_vehicle",
            "stance", "compass", "ammo_mag", "ammo_reserve", "helmet", "helmet_dur_pct", "armor", "armor_dur_pct",
            "supplies", "killfeed", "team_msgs", "banner", "team_panel", "weapon_main", "weapon_other", "scope", "lat_perc")
    dump_jsonl(d / "readings.jsonl", [scrub({k: r.get(k) for k in keep}) for r in jsonl(readings)])
    meta = json.loads((live / "meta.json").read_text(encoding="utf-8"))
    (d / "meta.json").write_text(json.dumps(clean(meta), ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    for k in ("ocr-eval", "ocr-pool", "offline", "o3-root", "live", "readings"):
        ap.add_argument("--" + k, required=True, type=Path)
    ap.add_argument("--out", default=Path("data"), type=Path)
    ap.add_argument("--scrub-file", type=Path, default=None, help="JSON {原文: 替换成}，不随仓库发布")
    a = ap.parse_args()
    if a.scrub_file:
        SCRUB.update(json.loads(a.scrub_file.read_text(encoding="utf-8")))
    export_perception(a.ocr_eval, a.ocr_pool, a.out / "perception")
    export_coach(a.offline, a.o3_root, a.live, a.readings, a.out / "coach")
    print("导出完成 →", a.out)


if __name__ == "__main__":
    main()
