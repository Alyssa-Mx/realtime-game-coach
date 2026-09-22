# -*- coding: utf-8 -*-
"""Director：决定"值不值得说、现在是不是该说、最该说什么"。**纯规则。**

每步（每行 gamestate）：
    events → playbook.candidates_for → 进候选池（带 ttl）
    过期的丢掉并记账；冷却中的同 action 排除（冷却按重复次数退避）
    同 goal 合并（"进圈/找车/沿掩体"是一句）
    打分：优先级底分 + urgency + 新鲜度 − 重复惩罚
    Must = 分最高的 P0 且 must=True
    说不说：
        P0  立刻说；在说别的（且那句不是 P0）就打断
        P1  空闲就说；忙就排队等
        P2  空闲且离上一句 ≥ GAP[P2] 才说；过期就丢
        P3  空闲且离上一句 ≥ GAP[P3] 且池里没有 P0/P1/P2 才说；忙就丢
三份历史分开：state（EventDetector 的窗）/ events / advice（这里）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .playbook import PRIORITY_BASE, Candidate, candidates_for

UTTER_S = 3.5                          # 一句话占用的对局秒（离线模拟；在线时用真实 TTS 时长）
GAP = {"P0": 0, "P1": 2, "P2": 6, "P3": 12}   # 距上一句结束至少多久
NOVEL_S = 120                          # 多久没说过这个 action 算"新鲜"
GOAL_GAP = {"zone": 30, "heal": 15, "team": 20, "vehicle": 60, "emotion": 20, "info": 30, "ammo": 20,
            "gear": 60, "stage": 60, "survive": 8, "threat": 20}   # 同一目标两句之间至少隔多久（P0 must 例外）

# goal 分的是"教练关心哪个话题"，但撞车发生在"玩家要做哪个动作"这一层：血低要进掩体、开完枪要
# 转移进掩体、弹匣空了要退到掩体后换弹、打药前也要先进掩体 —— 四个 goal，同一个动作。goal 维度
# 拦不住，于是 6 秒内连说两三遍同一件事（O7/A1 各 14 对，占 6.7%，换模型不变 = 结构问题）。
# 这里补一个动作维度：没登记的动作，意图就是它自己。
INTENT = {"TAKE_COVER": "cover", "COVER_BEFORE_HEAL": "cover",
          "REPOSITION_AFTER_SHOT": "cover", "AMMO_RELOAD": "cover",
          "HEAL_NOW": "heal", "HEAL_AFTER_COVER": "heal"}       # 打药也是同一件事的两个入口（交火中/安静后）
INTENT_GAP = {"cover": 20, "heal": 60}
BALANCE_SHARE = 0.25                  # 某子模块占本局已说句数超过这个比例才开始降权
BALANCE_PENALTY = 8.0                 # 占比到 2×BALANCE_SHARE 时扣满 8 分（≈一个优先级档的一半）    # 单位是对局秒（视频 2 倍速，60 对局秒 = 30 录像秒）。
                                          # 扫过 20–90：30–60 零代价，90 开始误伤（10→7 句）。45 放过了 P1 10:47/11:33 那对（隔 46 秒）
DOWNGRADE_TO = "HEAL_AFTER_COVER"      # 掩体那半句刚说过时，这条候选换成的标题（见 PLAYBOOK §20）


def intent_of(action: str) -> str:
    return INTENT.get(action, action)


@dataclass
class Decision:
    t: int
    t_video: float
    speak: bool
    must: dict | None
    top3: list
    interrupted: bool = False
    dropped: list = field(default_factory=list)
    queue: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"t": self.t, "t_video": self.t_video, "speak": self.speak, "must": self.must,
                "top3": self.top3, "interrupted": self.interrupted, "dropped": self.dropped, "queue": self.queue}


class Director:
    def __init__(self, utter_s: float = UTTER_S):
        self.utter_s = utter_s
        self.pool: list[Candidate] = []
        self.advice: list[dict] = []           # 已说过的：{t, action, goal, priority, reason}
        self.spoken_count: dict[str, int] = {}
        self.busy_until = -1.0
        self.busy_priority: str | None = None
        self.suppress_goal_until: dict[str, float] = {}
        self.stats = {"expired": {}, "cooldown_blocked": 0, "interrupts": 0, "p3_dropped_busy": 0}
        self.finished = False                  # 吃鸡/对局结束后闭嘴（后面是结算和回放画面）

    # ---------- 工具 ----------
    def _last_spoken(self, action: str) -> dict | None:
        for a in reversed(self.advice):
            if a["action"] == action:
                return a
        return None

    def _cooldown(self, c: Candidate) -> float:
        n = self.spoken_count.get(c.action, 0)
        return c.cooldown_s * min(3, 1 + n * 0.5)        # 第 1 次 1×，第 2 次 1.5×，第 3 次 2×，封顶 3×

    def _score(self, c: Candidate, t: int) -> float:
        s = PRIORITY_BASE[c.priority] + c.urgency
        last = self._last_spoken(c.action)
        if last is None or t - last["t"] >= NOVEL_S:
            s += 10
        elif t - last["t"] < 2 * self._cooldown(c):
            s -= 15
        # 类别均衡（借鉴自一份教练 Skill 设计）：本局某个子模块说得越多，它的新候选越往后排。
        # 我们 F 版五场 23% 在说安全区、13% 说载具，HUD 数字最多的话题永远赢。
        # 只对 P2/P3 生效；P0/P1 是生存和战斗，不因为"说多了"就让路。
        if c.priority in ("P2", "P3") and len(self.advice) >= 8 and c.routes:
            branch = c.routes[0][0]
            share = sum(1 for a in self.advice if a.get("branch") == branch) / len(self.advice)
            if share > BALANCE_SHARE:
                s -= BALANCE_PENALTY * (share - BALANCE_SHARE) / BALANCE_SHARE
        return s

    # ---------- 主入口 ----------
    def step(self, t: int, t_video: float, events: list, ctx: dict) -> Decision:
        dropped = []
        if self.finished and not any(e.type == "MATCH_WON" for e in events):
            self.pool = []
            return Decision(t=t, t_video=t_video, speak=False, must=None, top3=[])
        # 1. 新候选入池
        for ev in events:
            if ev.type == "MATCH_WON":
                self.finished = True           # 这一句照常说，之后闭嘴
            if ev.type in ("HEAL_IN_PROGRESS", "BOOST_IN_PROGRESS"):
                self.suppress_goal_until["heal"] = t + 20
                continue
            for c in candidates_for(ev, ctx):
                # 同 action 已在池里：更新为最新的（persist 事件反复来）
                self.pool = [x for x in self.pool if x.action != c.action]
                self.pool.append(c)
        # 2. 过期
        keep = []
        for c in self.pool:
            if t - c.t > c.ttl_s:
                self.stats["expired"][c.action] = self.stats["expired"].get(c.action, 0) + 1
                dropped.append({"action": c.action, "priority": c.priority, "why": "expired", "t": c.t})
            else:
                keep.append(c)
        self.pool = keep
        # 3. 冷却 / 目标压制
        avail = []
        last_goal_t = {}
        last_intent_t = {}
        for a in self.advice:
            last_goal_t[a["goal"]] = a["t"]
            last_intent_t[intent_of(a["action"])] = a["t"]
        for c in self.pool:
            if self.suppress_goal_until.get(c.goal, -1) > t:
                continue
            if not (c.priority == "P0" and c.must) and t - last_goal_t.get(c.goal, -9999) < GOAL_GAP.get(c.goal, 20):
                continue
            ci = intent_of(c.action)
            if ci in INTENT_GAP and not (c.priority == "P0" and c.must) and t - last_intent_t.get(ci, -9999) < INTENT_GAP[ci]:
                # "先进掩体再打药"整条放行，不改标题也不改提示（2026-09-03 定）。
                # 它是唯一带"用哪个药"的候选，拦掉会让 P1 13:11 / P7 17:28 / P8 15:56 三处
                # 血 10~48% 时 90 秒内一句用药建议都没有。
                # 试过把标题换成"现在打药"，结果标题的"现在"和提示的"进掩体之后"互相打架，
                # 模型输出了"血还行，先别打药"这种反过来的话（P1O9 13:11）—— 已回退。
                if c.action == "COVER_BEFORE_HEAL":
                    c.action = DOWNGRADE_TO
                    c.routes = [("资源与状态/时机安不安全", "现在能不能打药"),
                                ("资源与状态/自身状态", "血量")]
                    ci = intent_of(c.action)
                    # 降级换了意图，对新意图再查一次间隔（否则"现在打药"刚说完，降级出来的能绕过去）
                    if ci in INTENT_GAP and t - last_intent_t.get(ci, -9999) < INTENT_GAP[ci]:
                        self.stats["intent_blocked"] = self.stats.get("intent_blocked", 0) + 1
                        continue
                else:
                    self.stats["intent_blocked"] = self.stats.get("intent_blocked", 0) + 1
                    continue
            last = self._last_spoken(c.action)
            if last is not None and t - last["t"] < self._cooldown(c):
                self.stats["cooldown_blocked"] += 1
                continue
            c.score = self._score(c, t)
            avail.append(c)
        # 4. 同 goal 合并
        by_goal: dict[str, Candidate] = {}
        for c in sorted(avail, key=lambda x: -x.score):
            if c.goal in by_goal:
                by_goal[c.goal].merged.append(c.as_dict()) if c.action not in {m["action"] for m in by_goal[c.goal].merged} else None
            else:
                c.merged = []
                by_goal[c.goal] = c
        # 4b. 同意图合并：goal 不同但落到玩家身上是同一个动作，合成一句说（被合掉的依据当"其它角度"带走，
        #     增量信息不丢 —— 例如"进掩体"合并"进去之后用医疗箱回满"）
        by_intent: dict[str, Candidate] = {}
        for c in sorted(by_goal.values(), key=lambda x: -x.score):
            ci = intent_of(c.action)
            # ⚠ 已知 bug（2026-09-22 整理仓库、写单测时发现）：本意是"没登记的动作各自独立"，应写成
            #   `c.action not in INTENT`；现在拿意图名去查动作表，条件恒真，所以同一步内的同意图合并从没触发过。
            #   真正起作用的是上面第 3 步的"同意图间隔"（INTENT_GAP）。仓库里所有数据都是这版代码跑出来的，保留原样。
            if ci not in INTENT:                       # 没登记的动作各自独立，不参与合并
                by_intent[f"_{c.action}_{id(c)}"] = c
                continue
            if ci in by_intent:
                keep = by_intent[ci]
                if c.action not in {m["action"] for m in keep.merged} and c.action != keep.action:
                    keep.merged.append(c.as_dict())
            else:
                by_intent[ci] = c
        ranked = sorted(by_intent.values(), key=lambda x: -x.score)
        must = next((c for c in ranked if c.priority == "P0" and c.must), None)
        top3 = ranked[:3]
        # 5. 说不说
        busy = t < self.busy_until
        since_free = t - self.busy_until
        speak, interrupted, chosen = False, False, None
        if must is not None:
            if busy and self.busy_priority == "P0":
                pass                                  # P0 正在说 P0，等它说完
            else:
                if busy:
                    interrupted = True; self.stats["interrupts"] += 1
                speak, chosen = True, must
        elif ranked:
            best = ranked[0]
            has_urgent = any(c.priority in ("P0", "P1") for c in ranked)
            has_p2 = any(c.priority == "P2" for c in ranked)
            if best.priority == "P0":
                if not busy:
                    speak, chosen = True, best
            elif best.priority == "P1":
                if not busy and since_free >= GAP["P1"]:
                    speak, chosen = True, best
            elif best.priority == "P2":
                if not busy and since_free >= GAP["P2"] and not has_urgent:
                    speak, chosen = True, best
            else:  # P3
                if busy:
                    self.stats["p3_dropped_busy"] += 1
                    self.pool = [c for c in self.pool if c is not best]
                    dropped.append({"action": best.action, "priority": "P3", "why": "busy", "t": best.t})
                elif since_free >= GAP["P3"] and not has_urgent and not has_p2:
                    speak, chosen = True, best
        if speak and chosen is not None:
            self.busy_until = t + self.utter_s
            self.busy_priority = chosen.priority
            self.spoken_count[chosen.action] = self.spoken_count.get(chosen.action, 0) + 1
            self.advice.append({"t": t, "t_video": t_video, "action": chosen.action, "goal": chosen.goal,
                                "priority": chosen.priority, "reason": chosen.reason,
                                "branch": chosen.routes[0][0] if chosen.routes else ""})
            # 同 goal 的候选都算说过了
            self.pool = [c for c in self.pool if c.goal != chosen.goal]
            # 说了情绪句之后短时间不再说情绪句
            if chosen.emotional:
                self.suppress_goal_until["emotion"] = t + 20
        queue = [{"action": c.action, "priority": c.priority, "t": c.t} for c in self.pool if c is not chosen]
        return Decision(t=t, t_video=t_video, speak=speak, must=(must.as_dict() if must else None),
                        top3=[c.as_dict() for c in top3], interrupted=interrupted, dropped=dropped, queue=queue)
