# -*- coding: utf-8 -*-
"""Playbook：事件 → 候选行动（action）+ 优先级 + 决策框架路由 + 冷却。**这张表就是教练的决策政策。**

    Event ──(ctx 条件)──> Candidate{action, goal, priority, routes, reason, hint}

- `routes` 里的分支路径和决策项必须在 knowledge/decision_tree.md 上（决策框架的转述），加载时用 framework.check_route 校验。
- `goal` 相同的候选会被 Director 合并成一句（"进圈 / 找车进圈 / 沿掩体进圈"是一件事）。
- `priority`：P0 生存中断（可打断当前话）/ P1 战斗决策（排队）/ P2 战术优化（过期就丢）/ P3 陪伴反馈（忙就丢）。
- `hint` 给 Speaker：gamestate 没有的东西（有没有车、掩体在哪、地形）让它看画面补，不让它重做决策。
- 情绪价值单列 `emotional=True`，P0/P1 在场时让路。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .framework import check_route

PRIORITY_BASE = {"P0": 100, "P1": 70, "P2": 45, "P3": 20}


@dataclass
class Candidate:
    action: str
    goal: str
    priority: str
    routes: list            # [(分支路径, 决策项)]
    reason: str             # 中文依据，进字幕/日志
    hint: str = ""          # 给 Speaker 的落地提示
    cooldown_s: int = 30
    ttl_s: int = 15
    urgency: float = 0.0    # 0–20，同优先级内排序
    emotional: bool = False
    must: bool = False
    event_type: str = ""
    t: int = 0
    t_video: float = 0.0
    score: float = 0.0
    merged: list = field(default_factory=list)

    def __post_init__(self):
        for path, item in self.routes:
            check_route(path, item)

    def as_dict(self) -> dict:
        return {"action": self.action, "goal": self.goal, "priority": self.priority,
                "routes": [{"branch": p, "item": i} for p, i in self.routes],
                "reason": self.reason, "hint": self.hint, "urgency": self.urgency,
                "emotional": self.emotional, "must": self.must, "event": self.event_type,
                "t": self.t, "score": round(self.score, 1), "merged": self.merged}


def _throwables(ctx: dict) -> str:
    """身上的投掷物清单（知识库《四排战术配合》§4.1：烟 6 秒引信、燃烧瓶命中即燃、雷 55 米）。
    supplies 里 frag/smoke/molotov/stun 都是 OCR 读出来的真实数量。"""
    sup = ctx.get("supplies") or {}
    names = {"smoke": "烟", "frag": "雷", "molotov": "燃烧瓶", "stun": "闪光"}
    have = [f"{names[k]}×{sup[k]}" for k in ("smoke", "frag", "molotov", "stun") if sup.get(k)]
    return "、".join(have)


def _pct(x) -> str:
    return f"{int(round((x or 0) * 100))}%"


def _hp_word(hp) -> str:
    """血量的定性标签，写进依据给模型看。分档按知识库血线：绷带急救包封顶 75，85 才扛得住一枪狙。

    **为什么要有它**：原来依据只给裸数字，模型自己把 13% 和 48% 都翻译成了"血还行"（P1 13:11 /
    P8 15:56 各三个版本都错）。给了定性词之后这个错误消失。

    **措辞取舍（2026-09-03 实测，同一份代码跑两遍逐字相同，差异都是改动造成的）**：
      - 试过写成内部标签风格（残血 / 血量不到一半 / 血量过半…）：复述血量 7→2 句、字面雷同 4→2 对，
        但**打药句点名具体药从 4 句掉到 1 句**（只剩"打药""吃药""补血"这种笼统说法）。
        标签越不像台词，模型越不抄它，但句子重心整个转到掩体和位置上，把药名也一起略过了。
      - 现在这版（血量只剩一点 / 掉了一半多…）保住了药名，代价是偶尔复述一句"血量太低"。
        权衡后选它 —— 说清用哪个药比少说一句状态更重要。"""
    hp = hp or 0
    if hp < 0.30: return "血量只剩一点"
    if hp < 0.50: return "血量掉了一半多"
    if hp < 0.75: return "血量不满"
    if hp < 0.85: return "血量差一点满"
    return "血量基本满"


def _dir_sentence(d: dict) -> str:
    """方位只来自小地图虚线（知识库路径 2）；读不到就明确告诉说话人别说方向。"""
    if d.get("bearing") is not None and d.get("rel"):
        return f"安全区在你的{d['rel']}（罗盘 {int(d['bearing']):03d}，{d.get('compass_name') or ''}）"
    return "安全区方位未知：不要说任何方向词"


def _signal_sentence(d: dict) -> str:
    sig = d.get("signal")
    parts = []
    if sig is not None and sig < 0.999:
        parts.append(f"信号值 {_pct(sig)}（圈外掉的是信号值不是血量，急救包绷带在圈外没用）")
        if sig < 0.75:
            parts.append("信号低于 75% 挨打增伤，路上别接战")
        meds = []
        if d.get("painkiller"): meds.append(f"止痛药×{d['painkiller']}（+75 信号，6 秒）")
        if d.get("adrenaline"): meds.append(f"肾上腺素×{d['adrenaline']}（回满，8 秒）")
        if d.get("drink"): meds.append(f"饮料×{d['drink']}（+30，4 秒）")
        parts.append(("有 " + "、".join(meds) + "，先补信号再跑") if meds else "身上没有补信号的药，只能靠跑")
    return "；".join(parts)


_CAR_CN = {"run": "不到 200 米，直接跑，别找车", "car_if_handy": "200–600 米：车就在手边才开，否则跑",
           "prefer_car": "600 米以上：优先找车（取车要 15–30 秒，算进去）", "must_car": "1200 米以上：必须开车",
           "leave_early": "2000 米以上：必须开车且提前动身"}


def _tally_sentence(t: dict) -> str:
    """吃鸡时可以夸的、全场有记录的事实（说话人只看最近几秒，不知道这局发生过什么）。"""
    parts = []
    if t.get("self_kill") or t.get("self_knock"):
        parts.append(f"自己淘汰 {t.get('self_kill', 0)} 人、击倒 {t.get('self_knock', 0)} 人")
    if t.get("team_kill"):
        parts.append(f"队友拿下 {t['team_kill']} 个")
    parts.append("全场没掉过信号，进圈节奏好" if not t.get("signal_drops") else f"跑毒掉过 {t['signal_drops']} 次信号但都进来了")
    parts.append("自己没倒过" if not t.get("self_downed") else f"倒过 {t['self_downed']} 次都被扶起来了")
    if t.get("hp_low_episodes"):
        parts.append(f"血量见底 {t['hp_low_episodes']} 次都稳住了")
    return "；".join(parts)


def _heal_hint(ctx: dict, items: dict) -> str:
    """《药品速查表》：缺口≤25 绷带；中低血急救包只能到 75；过 75 只能靠能量；血<30 且安全用医疗箱（8 秒）；
    圈外先补信号（止痛药/饮料），急救包在圈外无效；85 血线（98K 打三级头 84.4）。"""
    hp = ctx.get("hp") or 0
    sup = ctx.get("supplies") or {}
    names = {"medkit": "医疗箱", "firstaid": "急救包", "bandage": "绷带", "drink": "饮料", "painkiller": "止痛药", "adrenaline": "肾上腺素"}
    have = {k: v for k, v in sup.items() if k in names and v}
    if ctx.get("outside"):
        boost = [names[k] for k in ("painkiller", "adrenaline", "drink") if have.get(k)]
        return ("圈外：先补信号（" + "、".join(boost) + "），急救包绷带在圈外没用") if boost else "圈外且没有补信号的药：别停下打急救包，先进圈"
    gap = 1 - hp
    if hp >= 0.75:
        boost = [names[k] for k in ("drink", "painkiller", "adrenaline") if have.get(k)]
        return ("血量过了 75，急救包绷带推不上去，用" + "、".join(boost) + "补能量慢慢回满，顺带拿移速加成") if boost else "血量过了 75，只有能量类药能再往上回，身上没有就不用管"
    if gap <= 0.25 and have.get("bandage"):
        return f"缺口不到 25，用绷带（4 秒，有 {have['bandage']} 个）最划算"
    if hp < 0.3 and have.get("medkit") and not ctx.get("in_combat"):
        return f"用医疗箱一次回满（8 秒，有 {have['medkit']} 个）"
    if have.get("firstaid"):
        return f"用急救包拉到 75（6 秒，有 {have['firstaid']} 个）；过 85 血线才能扛一枪狙" + ("，之后再喝饮料补满" if have.get("drink") else "")
    if have.get("bandage"):
        return f"只有绷带（{have['bandage']} 个）：连打几个，每个 10 血"
    if have.get("medkit"):
        return "只有医疗箱：找个安全角落 8 秒回满"
    return "没有回血的药：找队友要或顺路搜，别正面拼"


def candidates_for(ev, ctx: dict) -> list[Candidate]:
    """一个事件可以给出 0 到多个候选。ctx 是 EventDetector.ctx。"""
    d = ev.data
    T = ev.type
    C = []

    def add(**kw):
        kw.setdefault("event_type", T); kw.setdefault("t", ev.t); kw.setdefault("t_video", ev.t_video)
        C.append(Candidate(**kw))

    # ---------------- 毒圈 / 安全区 ----------------
    if T.startswith("ZONE_OUTSIDE_"):
        lvl = T.rsplit("_", 1)[1]
        dist, cd, shrink, veh = d["dist"], d.get("countdown"), d["shrinking"], d["in_vehicle"]
        where = (f"就在圈边（{dist} 米）" if dist < 60 else f"圈外约 {dist} 米") + ("，正在缩圈" if shrink else (f"，{cd} 秒后缩圈" if cd else ""))
        if d.get("approaching"):
            where += "，正驾车接近"
        where += f"；按知识库口径{'开车' if (veh or d.get('car') in ('prefer_car', 'must_car', 'leave_early')) else '跑步'}约 {d.get('t_need')} 秒，可用约 {d.get('available')} 秒（占 {int((d.get('ratio') or 0) * 100)}%）"
        routes = [("走位/毒圈", "安全区在哪"),
                  ("走位/怎么走", "转移路上会不会暴露")]
        if not veh:
            routes.append(("走位/用车", "要不要开车转移"))
        hint_parts = [_dir_sentence(d)]
        hint_parts.append("已在车上：路线走隐蔽地形，快到了提前想停哪" if veh else _CAR_CN.get(d.get("car"), ""))
        sig_s = _signal_sentence(d)
        if sig_s:
            hint_parts.append(sig_s)
        hint_parts.append("转移走隐蔽地形；进圈前收枪跑更快（**别为了移速临时打能量**：读条 10 秒只换回 4 秒，净亏）")
        hint = "；".join(x for x in hint_parts if x)
        # （曾有"跑毒前先扔烟"候选，只按距离触发。规则：封烟的前提是**空旷且周围有敌人**，
        #   没有敌人字段之前不做，否则是让玩家在没人的地方浪费烟。等视觉线出 open_ground / enemy 再加回。）
        if lvl == "URGENT":
            add(action="ZONE_ENTER_NOW", goal="zone", priority="P0", must=True, routes=routes,
                reason=where + "，按现在的方式进不去，必须换方式或补信号",
                hint=hint, cooldown_s=45, ttl_s=10,
                urgency=min(20.0, dist / 60))
        elif lvl == "TIGHT":
            add(action="ZONE_MOVE_NOW", goal="zone", priority="P1", routes=[("走位/毒圈", "什么时候缩圈")] + routes[1:],
                reason=where + ("，赶紧进去" if dist < 60 else "，余量不到四成，任何一次交火耽搁都会死在圈边"), hint=hint, cooldown_s=45, ttl_s=15, urgency=min(15.0, dist / 80))
        else:
            add(action="ZONE_PLAN_MOVE", goal="zone", priority="P2",
                routes=[("走位/毒圈", "什么时候缩圈"), ("走位/去哪", "下一个落脚点选哪")],
                reason=(("就在圈边，走两步就进去了" if dist < 60 else where + "，时间充裕，提前规划")),
                hint=(("圈边：直接进去，不用说方向" if dist < 60 else hint + "；现在不用催，把手头的事收尾再走")), cooldown_s=90, ttl_s=25,
                urgency=min(10.0, dist / 120))
    elif T == "SIGNAL_DROPPING":
        add(action="SIGNAL_LOSS", goal="zone", priority="P0", must=True,
            routes=[("走位/毒圈", "安全区在哪"), ("走位/怎么走", "借地形转移")],
            reason=f"信号值在掉（{_pct(d['frm'])}→{_pct(d['to'])}）" + (f"，圈外 {d['dist']} 米" if d.get("dist") else "") + f"，按当前信号还能扛约 {d.get('budget_left')} 秒",
            hint="；".join(x for x in [_dir_sentence(d), _signal_sentence({**d, "signal": d.get("to")}), "掉的是信号不是血，急救包没用；朝圈跑"] if x),
            cooldown_s=20, ttl_s=8, urgency=18)
    elif T == "ZONE_SHRINK_START":
        if d["outside"]:
            pass   # ZONE_OUTSIDE_* 会紧接着发，这里不重复
        else:
            add(action="ZONE_HOLD_CHECK", goal="zone", priority="P3",
                routes=[("走位/毒圈", "安全区在哪"), ("走位/去哪", "圈里哪里好守")],
                reason="圈开始缩，人在圈内", hint="提醒看一眼下个圈的位置、占个好点位；不用催", cooldown_s=240, ttl_s=10, urgency=2)
    elif T == "ZONE_ENTERED":
        # 进圈只是状态变化，不是成就。emotional 留给真正值得庆祝的事（击杀/吃鸡/倒地安慰），
        # 否则配音会用夸赞语气念一句流程话，听着别扭（P1O9 05:42 审片标注）。
        add(action="ZONE_ENTERED_OK", goal="zone", priority="P3",
            routes=[("走位/毒圈", "安全区在哪")],
            reason="进圈了", hint="一句肯定；别站圈心", cooldown_s=90, ttl_s=8, urgency=3)

    # ---------------- 血量 ----------------
    elif T == "SELF_HP_DROP":
        hp = d["to"]
        big = d["drop"] >= 0.3
        add(action="TAKE_COVER", goal="survive", priority="P0" if hp < 0.5 else "P1", must=hp < 0.5,
            routes=[("走位/找掩体", "最近的掩体"),
                    ("交战/撤出与保命", "继续打还是撤"),
                    ("走位/转点", "被发现了要不要换位置")],
            reason=f"血量 {_pct(d['frm'])}→{_pct(hp)}" + ("，一下掉了很多" if big else ""),
            hint=("先半句安慰，再指出画面里最近的掩体方向" if big else "指出画面里最近的掩体方向；被多方向打就转点")
                 + ("；有烟：烟扔在敌人看你的视线上，不是自己脚下" if ctx.get("smoke") else "")
                 + "；受击方向标识在屏幕哪边敌人就在哪边（上=正面，下=背后）",
            cooldown_s=12, ttl_s=8, urgency=min(20.0, (1 - hp) * 20))
    elif T == "SELF_HP_LOW":
        hp = d["hp"]
        if d["in_combat"]:
            add(action="COVER_BEFORE_HEAL", goal="heal", priority="P1",
                routes=[("走位/找掩体", "最近的掩体"), ("资源与状态/时机安不安全", "现在能不能打药")],
                reason=f"{_hp_word(hp)}（{_pct(hp)}）且刚交火", cooldown_s=25, ttl_s=10,
                # P1O4 13:11 审片标注：进掩体之后该用哪个药也要一起说（知识库v2 药品篇 §6.1：先问安全窗口，再问血线）
                # 提示不要重复"要说的事"标题（P1O5 10:47 因此被原样念出来）
                # 条件在前、动作在后的一句话。原来写成"进掩体之后：血很低且安全：用医疗箱…"两层冒号，
                # 模型把"且安全"当成待判定的前提，配上 6 秒前刚喊过找掩体，推出"还没安全所以先别打药"
                # （P1O9 13:11 说反了）。现在掩体信息仍在，但它是顺承不是前提。
                # 三层各说一件事：标题说时序（掩体后打药）、提示只说药、地物交给画面。
                # 提示里出现"掩体"两个字，模型就抄这个抽象词、不再去画面里找那堵墙（A 方案实测：
                # 点名地物 2/8、点名药 4/8，两头都不占）。
                hint=_heal_hint({**ctx, "in_combat": False}, d.get("items") or {}),
                urgency=min(15.0, (1 - hp) * 15))
        elif d["has_heal"]:
            add(action="HEAL_NOW", goal="heal", priority="P1",
                routes=[("资源与状态/时机安不安全", "现在能不能打药"), ("资源与状态/自身状态", "血量")],
                reason=f"{_hp_word(hp)}（{_pct(hp)}），已经一段时间没挨枪了", hint=_heal_hint(ctx, d.get("items") or {}), cooldown_s=40, ttl_s=15,
                urgency=min(12.0, (1 - hp) * 12))
        else:
            add(action="HEAL_NEED_SUPPLY", goal="heal", priority="P2",
                routes=[("资源与状态/物资够不够", "还要不要接着搜"),
                        ("交战/信息共享", "队友报的信息要不要跟")],
                reason=f"{_hp_word(hp)}（{_pct(hp)}）但没有药", hint="没药：找队友要或顺路搜；别正面拼", cooldown_s=90, ttl_s=20, urgency=6)

    # ---------------- 弹药 ----------------
    elif T == "AMMO_MAG_LOW":
        mag, cap = d["mag"], d["cap"]
        if d["in_combat"]:
            add(action="AMMO_RELOAD", goal="ammo", priority="P1",
                routes=[("资源与状态/时机安不安全", "现在能不能换弹"), ("交战/打法", "手上的枪合不合适这个距离")],
                reason=f"{d['weapon']} 弹匣只剩 {mag}/{cap}，正在交火", hint="进掩体换弹，或先切副武器", cooldown_s=20, ttl_s=8,
                urgency=min(15.0, (1 - mag / max(cap, 1)) * 15))
        else:
            add(action="AMMO_RELOAD", goal="ammo", priority="P2",
                routes=[("资源与状态/时机安不安全", "现在能不能换弹"), ("资源与状态/自身状态", "子弹够不够")],
                reason=f"{d['weapon']} 弹匣只剩 {mag}/{cap}", hint="现在安全，趁空换弹", cooldown_s=30, ttl_s=12, urgency=4)
    elif T == "AMMO_RESERVE_LOW":
        add(action="AMMO_RESUPPLY", goal="ammo", priority="P2",
            routes=[("资源与状态/自身状态", "子弹够不够"), ("资源与状态/物资够不够", "还要不要接着搜")],
            reason=f"{d['weapon']} 备弹只剩 {d['reserve']}", hint="顺路捡子弹或问队友要", cooldown_s=150, ttl_s=25, urgency=3)

    # ---------------- 护具 ----------------
    elif T == "GEAR_DAMAGED":
        add(action="GEAR_REPLACE", goal="gear", priority="P2",
            routes=[("资源与状态/自身状态", "头盔护甲还行吗"), ("资源与状态/物资够不够", "还要不要接着搜")],
            reason="、".join(d["parts"]) + "快打烂了", hint="安全时换掉，看到同级或更高的直接换", cooldown_s=150, ttl_s=25, urgency=3)
    elif T == "GEAR_MISSING":
        add(action="GEAR_FIND", goal="gear", priority="P2",
            routes=[("资源与状态/自身状态", "头盔护甲还行吗"), ("资源与状态/物资够不够", "还要不要接着搜")],
            reason="还没有" + "、".join(d["parts"]), hint="提醒补齐护具再接战", cooldown_s=240, ttl_s=30, urgency=2)

    # ---------------- 队友 ----------------
    elif T == "TEAMMATE_DOWNED":
        # 救人规则（2026-09-04 第二版）：**先躲到掩体再救**是首选；封烟只在敌人在远处打的时候才来得及，
        # 敌人贴脸时封烟根本来不及，得先处理眼前的人。我们没有敌人距离字段，所以不替玩家定"封不封"，
        # 把判断顺序交给说话人结合画面说。倒地原因（被毒倒/自伤）来自横幅，那两种周围不一定有人。
        cause = d.get("cause", "enemy")
        add(action="TEAM_REVIVE", goal="team", priority="P1",
            routes=[("交战/支援队友", "先清附近的威胁还是先扶人"), ("交战/支援队友", "怎么护住正在打药或扶人的队友")],
            reason=f"{d['member']} 号队友倒了" + {"zone": "（是被毒倒的，不是被打倒的）", "self": "（自己误伤倒的）"}.get(cause, "（被打倒的，周围有人）")
                   + (f"；你有 {d['smoke']} 颗烟" if d.get("smoke") else "；你没有烟"),
            hint=("让倒地的先报点（敌人在哪、几个）；救人顺序：靠掩体或斜坡卡住视野再扶是首选"
                  + ("；敌人在远处打才来得及封烟再扶，贴脸了封烟没用、先处理眼前的人" if d.get("smoke") and cause == "enemy" else "")
                  + ("；被毒倒的周围不一定有人，直接扶，别浪费烟" if cause == "zone" and d.get("smoke") else "")
                  + "；你离得近且安全就你去扶，否则架枪掩护让最近的队友去"
                  + ("；现在人已经很少，通常不扶，让他往掩体后爬、当眼睛报点" if (ctx.get("alive") or 100) <= 8 else "")),
            cooldown_s=25, ttl_s=15, urgency=14)
    elif T == "TEAMMATE_DANGER":
        if (ctx.get("hp") or 1) < 0.5 or ctx.get("downed"):
            return C                      # 自己都血低/倒地了，不谈支援
        add(action="TEAM_SUPPORT", goal="team", priority="P2" if d["in_combat"] else "P1",
            routes=[("局势/队友状况", "谁在打"), ("交战/支援队友", "要不要过去帮打")],
            # 面板标红只说明"有危险"，推不出血量高低（P1O4 17:52 审片标注：红≠血低）
            reason=f"{d['member']} 号队友的面板标红了（只知道他遇到了麻烦，具体是被打还是血量低都不知道）",
            hint="自己也在打就先顾自己；空闲就往他那边靠、架枪支援（画面里看不见他就不说方向）",
            cooldown_s=35, ttl_s=15, urgency=8)
    elif T == "TEAM_MSG":
        k, msg = d["msg_kind"], d["msg"]
        if k in ("mark", "mark_defend", "mark_vehicle", "mark_door"):
            kind_cn = {"mark": "一处地点", "mark_defend": "防守点", "mark_vehicle": "一辆载具", "mark_door": "一扇被打开过的门"}[k]
            who = d.get("who") or "队友"
            where = (f"，标记在你的{d['mark_rel']}（罗盘 {d['mark_bearing']:03d}，{d.get('mark_compass') or ''}）" if d.get("mark_rel") else "")
            add(action="CHECK_TEAM_MARKER", goal="info", priority="P2",
                routes=[("交战/信息共享", "队友报的信息要不要跟"), ("局势/队友状况", "队友都在哪")],
                reason=f"{who}标记了{kind_cn}{where}",
                hint=("标记方位可以说（来自消息里的罗盘读数）；" +
                      ("开门标记 = 有人来过，提高警惕" if k == "mark_door" else "按标记调整走位，别只复述消息")),
                cooldown_s=45, ttl_s=15, urgency=4)
        elif k == "enemy":
            add(action="ENEMY_INFO", goal="threat", priority="P1",
                routes=[("局势/周边风险", "附近可能有人吗"), ("交战/信息共享", "队友报没报敌情")],
                reason=f"{d.get('who') or '队友'}报敌情：{msg}" + (f"，方位在你的{d['mark_rel']}（罗盘 {d['mark_bearing']:03d}）" if d.get("mark_rel") else ""),
                hint="按消息里给的方位架枪；没给方位就不说方向；队友编号只用这里给的", cooldown_s=30, ttl_s=12, urgency=10)
        elif k == "gather":
            add(action="TEAM_GATHER", goal="team", priority="P2",
                routes=[("交战/配合队友", "要不要等队友到齐再打"), ("交战/信息共享", "队友报的信息要不要跟")],
                reason=f"{d.get('who') or '队友'}喊集合", hint="往队友那边靠拢，别单独接战；画面里看不见他就不说方向", cooldown_s=60, ttl_s=15, urgency=4)
        # supply / airdrop 类消息不出候选：不知道自己缺不缺，说了就是噪声

    # ---------------- 载具 ----------------
    elif T == "APPROACHING_DEST":
        add(action="PARK_PLAN", goal="vehicle", priority="P2",
            routes=[("走位/用车", "车停哪"), ("走位/用车", "车停这儿会不会暴露")],
            reason=f"开车快到了，离圈约 {int(d.get('dist') or 0)} 米",
            hint="提前想好停哪：停在房子背面或掩体后，别停开阔地正中；下车就有掩体可用",
            cooldown_s=180, ttl_s=15, urgency=4)
    elif T == "SURVIVED_BURST":
        add(action="HOLD_PRAISE", goal="emotion", priority="P3", emotional=True,
            routes=[("交战/撤出与保命", "继续打还是撤")],   # 我们的框架文件没有"情绪价值"这一节，挂在脱战下
            reason=f"刚才一下掉了很多血，{int(d.get('since_s') or 0)} 秒没再挨枪，顶住了",
            hint="肯定一句，点名做对的事（进了掩体 / 没正面拼 / 打了药）；顺带说下一步",
            cooldown_s=9999, ttl_s=12, urgency=2)
    elif T == "ENTER_VEHICLE":
        add(action="DRIVING_POLICY", goal="vehicle", priority="P2",
            routes=[("走位/用车", "开车走哪条路"), ("走位/用车", "这辆车还要不要开")],
            reason="上车了" + ("（驾驶）" if d.get("seat") == "驾驶" else ""),
            hint=("圈已经很小、人很少：车声大目标明显，容易被多方向集火，能不开就别开，或很快弃车" if ((ctx.get("alive") or 100) <= 15 or ev.t >= 1440) else
                  # 删掉了"不要四个人挤一辆车"：系统只知道自己在不在车上、是驾驶还是乘客，
                  # 同车人数读不出来（gamestate 里只有 in_vehicle / seat 两个字段），
                  # 等于在观测不到的前提下给建议。而且分车会让队伍分散，说错方向比不说更糟。
                  # （VL-30B 那轮实测：玩家自己在驾驶位时说出"别挤一辆车"，对他毫无意义。）
                  "路线走隐蔽地形；快到目的地提前想停哪"),
            cooldown_s=150, ttl_s=15, urgency=3)
    elif T == "EXIT_VEHICLE":
        add(action="PARK_CHECK", goal="vehicle", priority="P2",
            routes=[("走位/用车", "车停这儿会不会暴露"), ("走位/找掩体", "会不会被看见")],
            reason="下车了", hint="看画面：车停在开阔地正中就提醒下车离开；没别的遮挡时车身横过来能挡子弹", cooldown_s=150, ttl_s=10, urgency=3)

    # ---------------- 交战结果 / 情绪 ----------------
    elif T == "SELF_KILL":
        add(action="SELF_KILL_PRAISE", goal="emotion", priority="P3", emotional=True,
            routes=[("交战/打还是不打", "打还是避")],
            reason="自己淘汰一人", hint="夸一句，点名做对的事（时机/位置），不只夸枪法", cooldown_s=30, ttl_s=8, urgency=6)
    elif T == "SELF_KNOCK":
        add(action="SELF_KILL_PRAISE", goal="emotion", priority="P3", emotional=True,
            routes=[("交战/打还是不打", "打还是避")],
            reason="自己击倒一人", hint="夸一句", cooldown_s=30, ttl_s=8, urgency=5)
        add(action="REPOSITION_AFTER_SHOT", goal="survive", priority="P2",
            routes=[("走位/转点", "开枪后要不要挪位置"), ("交战/打法", "打的时候怎么用掩体")],
            reason="刚击倒一人，位置已暴露", hint="补人要看有没有队友补枪风险；否则先转点", cooldown_s=30, ttl_s=10, urgency=6)
    elif T == "TEAM_KILL":
        add(action="TEAM_KILL_PRAISE", goal="emotion", priority="P3", emotional=True,
            routes=[("交战/配合队友", "要不要和队友拉交叉火力")],
            # 不写枪名：庆祝队友不需要报枪，而依据里写什么模型就可能说什么（P1O9 13:43 审片标注）
            reason=(f"{d['who']} 号队友" if d.get("who") else "队友") + "拿下一人",
            hint="简短庆祝，顺带提醒配合队友压上/补位",
            cooldown_s=120, ttl_s=6, urgency=3)
    elif T == "ALIVE_STAGE":
        th, alive = d["threshold"], d["alive"]
        if th <= 10:
            add(action="ENDGAME_FOCUS", goal="stage", priority="P2",
                routes=[("局势/对局进程", "还剩多少人"), ("走位/找掩体", "这里会不会被多面夹击")],
                reason=f"只剩 {alive} 人（屏幕左上可见）",
                hint="卡圈边别占圈心、别贴树正后方（后期有人雷洗树）、少动、留掩体；语气带鼓励",
                cooldown_s=90, ttl_s=15, urgency=6)
        # 投掷物做成独立候选：挂在提示尾巴上 37 句只说出 1 句
        # 只在决赛阶段（≤15 人）出；第一版没加这个门，五场全在"只剩 50 人"时就说了
        if _throwables(ctx) and (alive or 100) <= 15:
            add(action="THROWABLE_ENDGAME", goal="gear", priority="P3",
                routes=[("交战/打法", "投掷物怎么用")],
                reason=f"只剩 {alive} 人，身上有 {_throwables(ctx)}",
                hint="说清怎么用：转移前先扔烟挡视线；卡点时雷和燃烧瓶封路（雷 55 米、燃烧瓶命中即燃）；别等被打了才想起来",
                cooldown_s=9999, ttl_s=20, urgency=3)
        # 50/20 人这种阶段提示不出候选：玩家看得见，模型还会复述数字（P1 14:33 "刚落地就剩50人"）
    elif T == "SELF_DOWNED":
        add(action="SELF_DOWNED_CALM", goal="survive", priority="P1", emotional=True,
            routes=[("交战/支援队友", "先清附近的威胁还是先扶人"), ("走位/找掩体", "会不会被看见")],
            reason="自己倒地了", hint="先安抚，再说：爬到掩体后面、别在开阔地等救，让队友先清掉威胁", cooldown_s=60, ttl_s=10, urgency=12)
    elif T == "SELF_REVIVED":
        add(action="SELF_REVIVED_HEAL", goal="heal", priority="P1",
            routes=[("资源与状态/时机安不安全", "现在能不能打药"), ("走位/找掩体", "最近的掩体")],
            reason=f"被扶起来了，血 {_pct(d['hp'])}", hint="起来先别露头，掩体后打药再动", cooldown_s=60, ttl_s=10, urgency=10)
    elif T == "MATCH_WON":
        add(action="WIN_CELEBRATE", goal="win", priority="P1", emotional=True,      # 独立 goal：别被 stage 的 60s 间隔挡掉（P5 曾因此没喊吃鸡）
            routes=[("局势/对局进程", "还剩多少人")],
            reason=f"对局结束，剩余 {d.get('alive')} 人就是自己队伍，吃鸡了",
            hint="庆祝；这局有记录的亮点：" + _tally_sentence(d.get("tally") or {}), cooldown_s=9999, ttl_s=15, urgency=15)
    elif T == "ENERGY_TOPUP":
        names = {"drink": "饮料", "painkiller": "止痛药", "adrenaline": "肾上腺素"}
        have = "、".join(f"{names[k]}×{v}" for k, v in (d.get("items") or {}).items() if k in names)
        add(action="ENERGY_TOPUP", goal="heal", priority="P3",
            routes=[("资源与状态/自身状态", "血量"), ("资源与状态/时机安不安全", "现在能不能打药")],
            reason=f"血量没满，绷带急救包已经推不动了（它们封顶 75），只有能量能补上去；已安静 {int(d.get('quiet_s') or 0)} 秒",
            hint=("现在安全，趁空把最后这点血用能量补满（" + (have or "身上的能量类药") + "）；"
                  "说'补满血'不要说'加速'——单为移速打能量是亏的（读条 10 秒只换 4 秒）"),
            cooldown_s=9999, ttl_s=15, urgency=1)
    # HEAL_IN_PROGRESS / BOOST_IN_PROGRESS：不出候选，Director 用它压掉 heal 目标
    return C


# action → 给说话人/字幕看的中文名（不是模型输出，是决策层的标签）
ACTION_CN = {
    "ZONE_ENTER_NOW": "立即进圈", "ZONE_MOVE_NOW": "马上动身进圈", "ZONE_PLAN_MOVE": "提前规划进圈",
    "SIGNAL_LOSS": "正在掉信号，朝圈跑", "ZONE_HOLD_CHECK": "看下个圈占位", "ZONE_ENTERED_OK": "进圈了",
    "TAKE_COVER": "先找掩体", "COVER_BEFORE_HEAL": "先进掩体再打药", "HEAL_NOW": "现在打药",
    "HEAL_AFTER_COVER": "掩体后打药",
    "HEAL_NEED_SUPPLY": "没药了要补", "AMMO_RELOAD": "换弹", "AMMO_RESUPPLY": "备弹不足", "GEAR_REPLACE": "换护具",
    "GEAR_FIND": "补齐护具", "TEAM_REVIVE": "救队友", "TEAM_SUPPORT": "支援队友", "CHECK_TEAM_MARKER": "看队友标记",
    "ENEMY_INFO": "队友报敌情", "TEAM_GATHER": "向队友集合", "DRIVING_POLICY": "开车路线", "PARK_CHECK": "停车位置",
    "SELF_KILL_PRAISE": "夸一句", "REPOSITION_AFTER_SHOT": "开枪后转点", "TEAM_KILL_PRAISE": "庆祝队友击杀",
    "ENDGAME_FOCUS": "圈小人少，稳住", "STAGE_NOTE": "阶段提示",
    "PARK_PLAN": "快到了，想好停哪", "HOLD_PRAISE": "这波顶住了",
    "THROWABLE_ENDGAME": "投掷物怎么用",
    "SELF_DOWNED_CALM": "倒地了别慌", "SELF_REVIVED_HEAL": "起来先打药", "WIN_CELEBRATE": "吃鸡了",
    "ENERGY_TOPUP": "趁空补满血",
}
PRIORITY_CN = {"P0": "生存中断", "P1": "战斗决策", "P2": "战术优化", "P3": "陪伴反馈"}
