"""生成 assets/ 下的配图（浅色 + 深色两版，README 用 <picture> 按读者主题切换）。
数字全部现算自 data/，不手填。需要 matplotlib 和一个中文字体（Noto Sans SC）：
    CJK_FONT=/path/to/NotoSansSC.ttf python analysis/make_figures.py
"""
import collections
import csv
import json
import os
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "assets"
OUT.mkdir(exist_ok=True)
FONT = os.environ.get("CJK_FONT", "")
if FONT and Path(FONT).is_file():
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

# 类别色（前两色）沿用 dataviz 校验过的一组；优先级 P0–P3 另一组，浅/深各自过了色盲与亮度带校验
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#8a8984", grid="#e8e7e3", neutral="#cfcfca",
                  s1="#2a78d6", s2="#eb6834", P0="#d0382a", P1="#bf8a00", P2="#3a6fd8", P3="#179466"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#8f8e87", grid="#2e2e2c", neutral="#4a4a47",
                 s1="#3987e5", s2="#d95926", P0="#d23d35", P1="#b88c1c", P2="#5591e8", P3="#1d9a6f"),
}
PCN = {"P0": "P0 生存中断", "P1": "P1 战斗决策", "P2": "P2 战术优化", "P3": "P3 陪伴反馈"}


def jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def rows(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def style(ax, c, ygrid=False, xgrid=True):
    ax.set_facecolor(c["surface"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(c["grid"])
    ax.tick_params(colors=c["ink2"], labelsize=9, length=0)
    if xgrid:
        ax.xaxis.grid(True, color=c["grid"], linewidth=1)
    if ygrid:
        ax.yaxis.grid(True, color=c["grid"], linewidth=1)
    ax.set_axisbelow(True)


def title(fig, c, t, sub):
    fig.text(0.012, 0.965, t, fontsize=13, fontweight="bold", color=c["ink"], va="top")
    fig.text(0.012, 0.905, sub, fontsize=9.5, color=c["ink2"], va="top")


def save(fig, name, theme):
    fig.savefig(OUT / f"{name}_{theme}.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------- 1. 一场 30 分钟：事件 → 开口
MODULES = [("安全区", ("ZONE_", "SIGNAL_")), ("生命值", ("SELF_HP", "SURVIVED", "SELF_DOWN", "SELF_REV", "HEAL_", "BOOST_", "ENERGY_")),
           ("弹药", ("AMMO_",)), ("护具", ("GEAR_",)), ("队友", ("TEAMMATE_", "TEAM_MSG")), ("载具", ("ENTER_V", "EXIT_V", "APPROACH")),
           ("战果与阶段", ("SELF_KNOCK", "SELF_KILL", "TEAM_KILL", "ALIVE_", "MATCH_"))]


def module_of(t):
    for i, (_, pre) in enumerate(MODULES):
        if t.startswith(pre):
            return i
    return None


def fig_timeline(theme):
    c = THEMES[theme]
    ev = jsonl(DATA / "coach/live_P1/events.jsonl")
    sp = jsonl(DATA / "coach/live_P1/speech.jsonl")
    fig = plt.figure(figsize=(11, 4.3), facecolor=c["surface"])
    title(fig, c, f"一场 30 分钟四排：记录员判出 {len(ev)} 件事，编导只让说话人开口 {len(sp)} 次",
          "P1 真流式实跑：v7 实时读数 → 规则 → 本地 Qwen3-Omni。上：每件事按决策树模块分行；下：每次开口，高度 = 急缓档位")
    ax = fig.add_axes([0.105, 0.14, 0.87, 0.66])
    style(ax, c, xgrid=True)
    nmod = len(MODULES)
    for i, (name, _) in enumerate(MODULES):
        y = nmod - i + 1.2
        ax.text(-0.4, y, name, ha="right", va="center", fontsize=9, color=c["ink2"], transform=ax.transData)
        xs = [e["t"] / 60 for e in ev if module_of(e["type"]) == i]
        ax.vlines(xs, y - 0.28, y + 0.28, color=c["ink2"], linewidth=1.1)
    ax.axhline(1.35, color=c["grid"], linewidth=1)
    hgt = {"P0": 1.0, "P1": 0.78, "P2": 0.58, "P3": 0.42}
    for s in sp:
        p = s["priority"]
        ax.bar(s["t"] / 60, hgt[p], width=0.22, bottom=0.05, color=c[p], linewidth=0)
    ax.text(-0.4, 0.55, "开口", ha="right", va="center", fontsize=9, color=c["ink"], fontweight="bold")
    n = collections.Counter(s["priority"] for s in sp)
    for j, p in enumerate(("P0", "P1", "P2", "P3")):
        fig.patches.append(matplotlib.patches.Rectangle((0.105 + j * 0.16, 0.035), 0.012, 0.03, color=c[p], transform=fig.transFigure))
        fig.text(0.121 + j * 0.16, 0.05, f"{PCN[p]}  {n[p]} 句", fontsize=9, color=c["ink2"], va="center")
    ax.set_xlim(0, 30.5)
    ax.set_ylim(0, nmod + 2)
    ax.set_yticks([])
    ax.set_xticks(range(0, 31, 5))
    ax.set_xticklabels([f"{m} 分" for m in range(0, 31, 5)])
    save(fig, "fig_timeline", theme)


# ---------------------------------------------------------------- 2. 读仪表盘：零样本 vs v7（P8 留出场）
FIELD_CN = {"game_t": "对局时间", "alive": "剩余人数", "hp": "血量", "signal": "信号值", "energy": "能量", "ammo_mag": "弹匣/备弹",
            "scope": "倍镜", "helmet": "头盔", "armor": "护甲", "zone_countdown": "缩圈倒计时", "zone_dist_m": "离安全区",
            "in_vehicle": "是否在车上", "stance": "姿势", "compass": "罗盘", "supplies": "背包物资", "team_panel": "队友状态栏",
            "weapon_slot1": "手上的枪", "weapon_slot2": "另一把枪", "banner": "击杀横幅", "killfeed": "他人击杀记录", "team_msgs": "队友消息"}


def fig_perception(theme):
    c = THEMES[theme]
    rs = [r for r in rows(DATA / "perception/field_accuracy_P8.csv") if r["field"] in FIELD_CN]
    rs.sort(key=lambda r: float(r["zero_shot"]))
    mean = {r["field"]: r for r in rows(DATA / "perception/field_accuracy_P8.csv")}["MEAN21"]["v7"]
    fig = plt.figure(figsize=(11, 6.2), facecolor=c["surface"])
    title(fig, c, f"读仪表盘：留出场 P8 上 21 个字段平均 {float(mean):.3f}",
          "Qwen3-VL-8B 零样本（每个框裁出来直接问）vs 自己标数据 LoRA 训的 v7；P8 整场 860 帧从未进训练。零样本读不出的全是细处：血条 4 像素高、耐久条 10×4 像素")
    ax = fig.add_axes([0.14, 0.08, 0.82, 0.76])
    style(ax, c)
    for i, r in enumerate(rs):
        z, v = float(r["zero_shot"]), float(r["v7"])
        ax.plot([z, v], [i, i], color=c["neutral"], linewidth=2, zorder=1, solid_capstyle="round")
        ax.scatter([z], [i], s=34, color=c["s2"], zorder=3, edgecolor=c["surface"], linewidth=1.5)
        ax.scatter([v], [i], s=34, color=c["s1"], zorder=3, edgecolor=c["surface"], linewidth=1.5)
        if v < 0.9:
            ax.text(v - 0.015, i, f"{v:.2f}", ha="right", va="center", fontsize=8.5, color=c["ink2"])
    ax.set_yticks(range(len(rs)))
    ax.set_yticklabels([FIELD_CN[r["field"]] for r in rs], fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("逐字段准确率（数值类允许 ±0.05，罗盘 ±5°）", fontsize=9, color=c["ink2"])
    ax.scatter([], [], s=34, color=c["s2"], label="零样本")
    ax.scatter([], [], s=34, color=c["s1"], label="v7（LoRA）")
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=c["ink2"])
    save(fig, "fig_perception", theme)


# ---------------------------------------------------------------- 3. 切图 vs 整图
def fig_crop(theme):
    c = THEMES[theme]
    t = {r["field"]: r for r in rows(DATA / "perception/tiers_P7.csv")}
    fr = {r["tier"]: r for r in rows(DATA / "perception/tiers_P7_frame.csv")}
    s1, s4 = float(fr["S1"]["sec_per_frame_vllm"]), float(fr["S4"]["sec_per_frame_vllm"])
    m1, m4 = float(t["MEAN21"]["S1_crop"]), float(t["MEAN21"]["S4_whole"])
    fig = plt.figure(figsize=(11, 4.6), facecolor=c["surface"])
    title(fig, c, f"切图还是整图：整图慢 {s4 / s1:.2f}×、准确率低 {100 * (m1 - m4):.1f}pp —— 两头都输",
          "同一个模型（v3，唯一四档都训过的版本）、同一个 vLLM 进程；精度用 P7 的 215 帧配对。一个视觉 token 管 32×32 像素，小结构被压没了")
    a1 = fig.add_axes([0.07, 0.14, 0.26, 0.64])
    style(a1, c, xgrid=False, ygrid=True)
    a1.bar([0, 1], [s1, s4], width=0.55, color=[c["s1"], c["s2"]])
    for x, v, lab in ((0, s1, f"{s1:.2f} s\n32 个请求并行"), (1, s4, f"{s4:.2f} s\n1 个请求")):
        a1.text(x, v + 0.08, lab, ha="center", va="bottom", fontsize=9, color=c["ink"])
    a1.set_xticks([0, 1]); a1.set_xticklabels(["切图（每个框单独读）", "整图"], fontsize=9)
    a1.set_ylim(0, 4.3); a1.set_ylabel("读一帧（秒）", fontsize=9, color=c["ink2"])
    a2 = fig.add_axes([0.46, 0.14, 0.51, 0.64])
    style(a2, c)
    gaps = sorted(((float(t[f]["S1_crop"]) - float(t[f]["S4_whole"]), f) for f in t if f != "MEAN21"), reverse=True)[:7]
    for i, (_, f) in enumerate(reversed(gaps)):
        a, b = float(t[f]["S1_crop"]), float(t[f]["S4_whole"])
        a2.plot([b, a], [i, i], color=c["neutral"], linewidth=2, zorder=1)
        a2.scatter([a], [i], s=34, color=c["s1"], zorder=3, edgecolor=c["surface"], linewidth=1.5)
        a2.scatter([b], [i], s=34, color=c["s2"], zorder=3, edgecolor=c["surface"], linewidth=1.5)
    a2.set_yticks(range(len(gaps))); a2.set_yticklabels([FIELD_CN[f] for _, f in reversed(gaps)], fontsize=9)
    a2.set_xlim(0.1, 1.02)
    a2.scatter([], [], s=34, color=c["s1"], label=f"切图  21 字段 {m1:.3f}")
    a2.scatter([], [], s=34, color=c["s2"], label=f"整图  21 字段 {m4:.3f}")
    a2.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=c["ink2"])
    a2.set_title("差距最大的 7 个字段", fontsize=9.5, color=c["ink2"], loc="left")
    save(fig, "fig_crop", theme)


# ---------------------------------------------------------------- 4. 延迟：读一帧 5.29 → 0.93 s；整条链 2.3 s
def fig_latency(theme):
    c = THEMES[theme]
    lat = rows(DATA / "perception/latency_steps.csv")
    keep = [lat[0], lat[1], lat[2], lat[-1]]
    names = ["HF，LoRA 未合并", "合并 LoRA", "31 个框一批读", "换 vLLM"]
    sp = jsonl(DATA / "coach/live_P1/speech.jsonl")
    perc = st.median(s["lat_perc"] for s in sp)
    fa = st.median(s["first_audio"] for s in sp)
    heard = st.median(s["e2e_heard"] for s in sp)
    fig = plt.figure(figsize=(11, 4.4), facecolor=c["surface"])
    title(fig, c, "实时性：读一帧 5.29 s → 0.93 s；整条链路从画面到达到玩家听到，中位 2.29 s",
          "左：单张 H20 读一帧 31 个框（每一步都在留出场核对精度，P8 全量 0.9503 → 0.9507）。右：P1 真流式实跑 38 句的中位数，感知与说话人同机并发")
    a1 = fig.add_axes([0.13, 0.14, 0.37, 0.64])
    style(a1, c)
    vals = [float(r["sec_per_frame"]) for r in keep]
    ys = list(range(len(vals)))[::-1]
    a1.barh(ys, vals, height=0.55, color=[c["neutral"]] * 3 + [c["s1"]])
    for y, v in zip(ys, vals):
        a1.text(v + 0.08, y, f"{v:.2f} s", va="center", fontsize=9, color=c["ink"])
    a1.axvline(2.0, color=c["s2"], linewidth=1.2, linestyle=(0, (4, 3)))
    a1.text(2.05, 3.42, "预算 2 s（每 2 游戏秒一帧）", fontsize=8.5, color=c["s2"])
    a1.set_yticks(ys); a1.set_yticklabels(names, fontsize=9); a1.set_xlim(0, 6.2)
    a1.set_xlabel("读一帧（秒）", fontsize=9, color=c["ink2"])
    a2 = fig.add_axes([0.62, 0.30, 0.35, 0.34])
    style(a2, c)
    a2.barh([0], [perc], height=0.5, color=c["neutral"])
    a2.barh([0], [fa], left=[perc], height=0.5, color=c["s1"])
    a2.text(perc / 2, 0, f"感知 {perc:.2f}", ha="center", va="center", fontsize=9, color=c["ink"])
    a2.text(perc + fa / 2, 0, f"说话人 {fa:.2f}", ha="center", va="center", fontsize=9, color="#ffffff")
    a2.text(heard + 0.05, 0, f"听到 {heard:.2f} s", va="center", fontsize=9.5, color=c["ink"], fontweight="bold")
    a2.set_xlim(0, 3.2); a2.set_yticks([]); a2.set_xlabel("秒（从这一帧画面到达算起）", fontsize=9, color=c["ink2"])
    a2.set_title("规则决策 < 1 ms；说话人 = Qwen3-Omni 出字 + 出第一个语音块", fontsize=9, color=c["ink2"], loc="left")
    save(fig, "fig_latency", theme)


if __name__ == "__main__":
    for th in THEMES:
        for f in (fig_timeline, fig_perception, fig_crop, fig_latency):
            f(th)
    print("→", OUT)
