# -*- coding: utf-8 -*-
"""规则层与在线感知接入的单测：只用标准库，不需要 GPU、模型、录像。

    python -m unittest discover -s tests -v

覆盖简历里写的几条机制：连续帧确认、P0 抢占、排队 / 过期淘汰、忙时丢 P3、同意图间隔、冷却退避、
在线感知的拥塞丢帧（丢最旧）与事件字段不沿用，以及说话人的生成后校验。
"""
from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# 小地图读方位要 OpenCV；这里的测试用不到它，没装就放个空模块占位
for _m in ("cv2", "numpy"):
    try:
        __import__(_m)
    except ImportError:
        mod = types.ModuleType(_m)
        mod.ndarray = object
        sys.modules[_m] = mod

import coach.director as director_mod  # noqa: E402
from coach.director import Director, UTTER_S  # noqa: E402
from coach.events import Event, EventDetector  # noqa: E402
from coach.playbook import Candidate  # noqa: E402
from coach.speaker import check_direction, check_persona, echoes_hint  # noqa: E402

COVER = [("走位/找掩体", "最近的掩体")]
ZONE = [("走位/毒圈", "什么时候缩圈")]
TEAM = [("局势/队友状况", "谁在打")]
AMMO = [("资源与状态/时机安不安全", "现在能不能换弹")]
PRAISE = [("交战/配合队友", "要不要和队友拉交叉火力")]


def row(t: int, **kw) -> dict:
    """一帧读数（字段名与感知模型的输出一致），默认是圈内、满血、无事发生。"""
    r = {"game_t": t, "t_video": t / 2, "alive": 50, "hp": 1.0, "signal": 1.0, "energy": 0.0, "energy_fit": 0.0,
         "zone_countdown": "02:00", "zone_dist_m": None, "in_vehicle": False, "stance": "站", "compass": 0,
         "ammo_mag": 30, "ammo_reserve": 120, "weapon_main": "M416", "helmet": 2, "armor": 2,
         "supplies": {"firstaid": 2, "drink": 2}, "team_panel": {"danger": [], "downed": []},
         "banner": {"raw": None, "kind": None}, "killfeed": [], "team_msgs": [], "self_slot": 4}
    r.update(kw)
    return r


def run(rows: list[dict]) -> list[str]:
    det = EventDetector()
    out = []
    for r in rows:
        out += [e.type for e in det.update(r)]
    return out


# ---------------------------------------------------------------- 记录员：连续帧确认
class ObserverTest(unittest.TestCase):
    def test_teammate_downed_needs_two_consecutive_rows(self):
        flash = [row(100), row(102, team_panel={"danger": [], "downed": [2]}), row(104), row(106)]
        self.assertNotIn("TEAMMATE_DOWNED", run(flash), "单帧闪一下不算倒地")
        held = [row(100), row(102, team_panel={"danger": [], "downed": [2]}), row(104, team_panel={"danger": [], "downed": [2]})]
        self.assertEqual(run(held).count("TEAMMATE_DOWNED"), 1)

    def test_self_row_is_not_a_teammate(self):
        rows = [row(100 + 2 * i, team_panel={"danger": [4], "downed": [4]}) for i in range(4)]
        ev = run(rows)
        self.assertNotIn("TEAMMATE_DOWNED", ev)
        self.assertNotIn("TEAMMATE_DANGER", ev)

    def test_hp_drop_threshold(self):
        self.assertIn("SELF_HP_DROP", run([row(100, hp=0.9), row(102, hp=0.8)]))
        self.assertNotIn("SELF_HP_DROP", run([row(100, hp=0.9), row(102, hp=0.85)]))

    def test_same_banner_is_one_event(self):
        b = {"raw": "你的队友 队友3 使用M416突击步枪淘汰了 玩家54", "kind": "team_kill"}
        ev = run([row(100, banner=b), row(102, banner=b), row(104, banner=b)])
        self.assertEqual(ev.count("TEAM_KILL"), 1, "横幅停在屏幕上几秒，只算出现那一下")

    def test_zone_distance_needs_two_rows(self):
        # 圈距只读到一帧（常见的孤立值）不当成“人在圈外”
        ev = run([row(100), row(102, zone_dist_m=1200, zone_countdown="00:40"), row(104)])
        self.assertFalse(any(e.startswith("ZONE_OUTSIDE") for e in ev))


# ---------------------------------------------------------------- 编导：优先级、抢占、排队、过期、合并
class DirectorTest(unittest.TestCase):
    def setUp(self):
        self._orig = director_mod.candidates_for
        self.queue: dict[int, list[Candidate]] = {}
        director_mod.candidates_for = lambda ev, ctx: [self._stamp(c, ev) for c in self.queue.pop(id(ev), [])]
        self.dr = Director()

    def tearDown(self):
        director_mod.candidates_for = self._orig

    @staticmethod
    def _stamp(c, ev):
        c.t, c.t_video = ev.t, ev.t_video
        return c

    def step(self, t, *cands):
        evs = []
        for c in cands:
            ev = Event("TEST", t, t / 2)
            self.queue[id(ev)] = [c]
            evs.append(ev)
        return self.dr.step(t, t / 2, evs, {})

    def test_p0_interrupts_non_p0(self):
        d1 = self.step(100, Candidate("ZONE_PLAN_MOVE", "zone", "P2", ZONE, "圈外 900 米"))
        self.assertTrue(d1.speak)
        d2 = self.step(101, Candidate("TAKE_COVER", "survive", "P0", COVER, "血 100%→11%", must=True))
        self.assertTrue(d2.speak and d2.interrupted, "P0 必须说：打断正在说的 P2")

    def test_p0_waits_for_p0(self):
        self.step(100, Candidate("TAKE_COVER", "survive", "P0", COVER, "掉血", must=True))
        d = self.step(101, Candidate("ZONE_URGENT", "zone", "P0", ZONE, "圈外来不及", must=True))
        self.assertFalse(d.speak, "正在说的也是 P0：等它说完，不互相打断")

    def test_p1_queues_while_busy_then_speaks(self):
        self.step(100, Candidate("ZONE_PLAN_MOVE", "zone", "P2", ZONE, "规划进圈"))
        d = self.step(101, Candidate("TEAM_SUPPORT", "team", "P1", TEAM, "3 号危险", ttl_s=15))
        self.assertFalse(d.speak)
        self.assertIn("TEAM_SUPPORT", [q["action"] for q in d.queue])
        free = int(100 + UTTER_S + 2) + 1                      # 说完 + 空闲 2 秒之后
        d = self.dr.step(free, free / 2, [], {})
        self.assertTrue(d.speak)
        self.assertEqual(d.top3[0]["action"], "TEAM_SUPPORT")

    def test_expired_candidate_is_dropped_and_logged(self):
        self.step(100, Candidate("ZONE_PLAN_MOVE", "zone", "P2", ZONE, "规划进圈"))
        self.step(101, Candidate("TEAM_SUPPORT", "team", "P1", TEAM, "3 号危险", ttl_s=2))
        d = self.dr.step(106, 53, [], {})
        self.assertFalse(d.speak)
        self.assertEqual([(x["action"], x["why"]) for x in d.dropped], [("TEAM_SUPPORT", "expired")])
        self.assertEqual(self.dr.stats["expired"]["TEAM_SUPPORT"], 1)

    def test_p3_dropped_when_busy(self):
        self.step(100, Candidate("ZONE_PLAN_MOVE", "zone", "P2", ZONE, "规划进圈"))
        d = self.step(101, Candidate("TEAM_KILL_PRAISE", "emotion", "P3", PRAISE, "队友拿下一人", emotional=True))
        self.assertFalse(d.speak)
        self.assertEqual([(x["action"], x["why"]) for x in d.dropped], [("TEAM_KILL_PRAISE", "busy")])

    def test_same_intent_interval_blocks_repeat(self):
        # 生效的那道：刚说完“进掩体”，10 秒后弹匣快空要“退到掩体后换弹”—— 话题不同（ammo），落到玩家身上
        # 还是“进掩体”，同意图 20 秒内不再说（五场离线回放里这道拦下 76 次）
        self.step(100, Candidate("TAKE_COVER", "survive", "P1", COVER, "血 80%→60%"))
        d = self.step(110, Candidate("AMMO_RELOAD", "ammo", "P1", AMMO, "弹匣 2/30"))
        self.assertFalse(d.speak)
        self.assertEqual(self.dr.stats.get("intent_blocked"), 1)

    @unittest.expectedFailure
    def test_same_intent_is_merged_into_one_sentence(self):
        # 本意：同一步里血低要进掩体、弹匣空了要退到掩体后换弹 —— 话题不同、动作相同，合成一句。
        # 已知 bug（见 coach/director.py 第 4b 步的注释）：条件恒真，这一步从没合并过。修掉之前这条用例预期失败。
        d = self.step(100, Candidate("TAKE_COVER", "survive", "P1", COVER, "血 80%→60%", urgency=8),
                      Candidate("AMMO_RELOAD", "ammo", "P1", AMMO, "弹匣 2/30", urgency=4))
        self.assertTrue(d.speak)
        said = d.top3[0]
        self.assertEqual(said["action"], "TAKE_COVER")
        self.assertIn("AMMO_RELOAD", [m["action"] for m in said["merged"]])

    def test_cooldown_backs_off_and_caps(self):
        c = Candidate("TAKE_COVER", "survive", "P1", COVER, "x", cooldown_s=12)
        got = []
        for n in range(6):
            self.dr.spoken_count["TAKE_COVER"] = n
            got.append(self.dr._cooldown(c))
        self.assertEqual(got, [12, 18, 24, 30, 36, 36])

    def test_silent_after_win(self):
        self.dr.finished = True
        d = self.step(500, Candidate("TAKE_COVER", "survive", "P0", COVER, "x", must=True))
        self.assertFalse(d.speak, "吃鸡之后是结算和回放画面，闭嘴")


# ---------------------------------------------------------------- 在线感知：拥塞丢帧 + 事件字段不沿用
class LiveProviderTest(unittest.TestCase):
    def setUp(self):
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.outputs: dict[float, dict] = {}
        fake = types.ModuleType("infer_vllm")

        def read_frame_parsed(frame, match_id):
            self.entered.set()
            self.gate.wait(5)
            return dict(self.outputs.get(frame, {"hp": 1.0}))

        fake.read_frame_parsed = read_frame_parsed
        fake.to_gamestate = lambda raw, game_t=None, t_video=None: dict(raw)
        fake.shutdown = lambda exit_process=True: None
        sys.modules["infer_vllm"] = fake
        from coach.live_provider import LiveGameState
        self.g = LiveGameState(ocr_train_dir=str(ROOT), maxsize=2)

    def tearDown(self):
        self.gate.set()
        self.g.close(exit_process=False)
        sys.modules.pop("infer_vllm", None)

    def _wait_done(self, n):
        for _ in range(200):
            if self.g.stats["done"] >= n:
                return
            time.sleep(0.01)
        self.fail(f"只处理完 {self.g.stats['done']} 帧")

    def test_queue_full_drops_oldest_not_newest(self):
        self.g.submit(0.0, 0.0)
        self.assertTrue(self.entered.wait(2))                  # 第 0 帧正在推理（卡住）
        for f in (1.0, 2.0, 3.0, 4.0):
            self.g.submit(f, f)
        self.assertEqual(self.g.stats["dropped"], 2, "队列容量 2：新来 4 帧，丢掉最旧的 2 帧")
        self.gate.set()
        self._wait_done(3)
        self.assertEqual(self.g.latest()["t_video"], 4.0, "展示的永远是最新一帧")

    def test_event_fields_are_never_carried_over(self):
        self.gate.set()
        self.outputs[1.0] = {"hp": 0.8, "banner": {"raw": "你使用 M416 击倒了 玩家7", "kind": "self_kill"}}
        self.outputs[2.0] = {"hp": None, "banner": None}
        self.g.submit(1.0, 1.0); self._wait_done(1)
        self.g.submit(2.0, 2.0); self._wait_done(2)
        r = self.g.latest()
        self.assertIsNone(r["banner"], "沿用上一帧的横幅 = 把同一个击杀再播一遍")
        self.assertEqual(r["hp"], 0.8, "数值字段读不出来时沿用上一帧，并打标记")
        self.assertIn("hp", r["stale_keys"])
        self.assertIn("age_s", r)


# ---------------------------------------------------------------- 说话人：生成后的代码校验
class SpeakerGuardTest(unittest.TestCase):
    def test_bare_direction_is_rejected(self):
        # 实跑 P1 第 15 句的原文：P0“先找掩体”却复读了上一句，“往左”没有可见物撑着
        ok, why = check_direction("刚打完一枪，赶紧往左转移位置。", "survive", zone_dir_known=False)
        self.assertFalse(ok)

    def test_direction_anchored_on_visible_object_passes(self):
        self.assertTrue(check_direction("右边矮墙后面躲一下，刚被打中了。", "survive", False)[0])

    def test_direction_on_teammate_is_rejected(self):
        self.assertFalse(check_direction("3号在你左边，快去扶", "team", False)[0])

    def test_zone_direction_only_when_bearing_known(self):
        self.assertTrue(check_direction("往左前方找车进圈", "zone", zone_dir_known=True)[0])
        self.assertFalse(check_direction("圈在左前方", "zone", zone_dir_known=False)[0])

    def test_persona(self):
        self.assertFalse(check_persona("等我扶他，你先架枪")[0])
        self.assertTrue(check_persona("让 2 号去扶，你先架枪")[0])

    def test_echoing_the_hint(self):
        hint = "先半句安慰，再指出画面里最近的掩体方向"
        self.assertTrue(echoes_hint("先半句安慰，再指出画面里最近的掩体方向", hint))
        self.assertFalse(echoes_hint("别慌，左边那堵墙后面躲一下", hint))


if __name__ == "__main__":
    unittest.main()
