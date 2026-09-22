"""在线感知：用自己训的 v7 读仪表盘模型逐帧出 gamestate，接口和 OracleGameState 的 at() 一致。

**和离线回放最根本的不同：感知有延迟（vLLM 单卡 0.93s/帧），状态是"截至某个时刻"的，不是"现在"。**
这里把三件事显式建模，不让下游自己猜：

1. `game_t` 是**那一帧画面上显示的时间**，是这条状态的时间戳，不校正、不外推。
   面板要显示"现在几点"时自己加上 `age_s`，不要直接把 game_t 当现在（会稳定慢约 1 秒）。
2. 每行带 `age_s`（这条状态产出时已经过去多久）和 `stale`（是不是沿用上一帧）。
   **事件型字段（banner/killfeed/team_msgs）不允许 stale** —— 沿用等于凭空重放一次击杀播报。
3. **队列满时丢最旧的帧**，不排队。排队会让延迟单向增长，最后面板显示的是几十秒前的战况；
   丢帧只损失中间过程，展示的永远是最新状态。丢了多少帧计在 `stats['dropped']` 里，别悄悄丢。

用法（vLLM 常驻）：
    from coach.live_provider import LiveGameState
    with LiveGameState(match_id="...") as g:      # with 保证 vLLM 引擎被显式关闭
        g.submit(frame_bgr, t_video)              # 非阻塞，队列满则丢最旧
        row = g.latest()                          # 最近一条已完成的状态
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time

EVENT_FIELDS = ("banner", "killfeed", "team_msgs")   # 沿用旧值会造成假事件，只允许"这一帧确实没有"


class LiveGameState:
    def __init__(self, match_id: str | None = None, ocr_train_dir: str | None = None,
                 maxsize: int = 2, backend: str = "vllm"):
        d = ocr_train_dir or os.environ.get(
            "OCR_TRAIN_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "perception", "train"))
        if d not in sys.path:
            sys.path.insert(0, d)
        if backend == "vllm":
            import infer_vllm as M                      # 0.93s/帧，常驻服务用
        else:
            import infer_gamestate as M                 # 3.35s/帧，跑一帧就退的脚本用
        self._M = M
        self._read = getattr(M, "read_frame_parsed", None) or M.read_frame
        self.match_id = match_id
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.stats = {"submitted": 0, "dropped": 0, "done": 0, "last_infer_s": 0.0}
        self._latest: dict | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    # ---------- 采集侧 ----------
    def submit(self, frame_bgr, t_video: float) -> bool:
        """非阻塞。队列满时丢**最旧**的一帧再放入新帧，返回是否发生丢帧。"""
        self.stats["submitted"] += 1
        dropped = False
        while True:
            try:
                self.q.put_nowait((frame_bgr, t_video, time.time()))
                return dropped
            except queue.Full:
                try:
                    self.q.get_nowait()
                    self.stats["dropped"] += 1
                    dropped = True
                except queue.Empty:
                    pass

    # ---------- 推理侧 ----------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame, tv, t_sub = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            t0 = time.time()
            try:
                raw = self._read(frame, self.match_id)
                row = self._M.to_gamestate(raw, game_t=None, t_video=tv)
            except Exception as e:                       # 单帧失败不能拖垮整条链路
                self.stats["errors"] = self.stats.get("errors", 0) + 1
                self.stats["last_error"] = repr(e)[:200]
                continue
            row["t_video"] = tv
            row["produced_at"] = time.time()
            row["infer_s"] = row["produced_at"] - t0
            row["queue_s"] = t0 - t_sub
            self.stats["last_infer_s"] = row["infer_s"]
            self.stats["done"] += 1
            with self._lock:
                prev = self._latest
                # 事件型字段读不出来就是"这一帧没有"，绝不沿用上一帧
                if prev is not None:
                    for k, v in prev.items():
                        if k in EVENT_FIELDS or k in ("t_video", "produced_at", "infer_s", "queue_s", "stale_keys"):
                            continue
                        if row.get(k) is None and v is not None:
                            row[k] = v
                            row.setdefault("stale_keys", []).append(k)
                self._latest = row

    # ---------- 消费侧 ----------
    def latest(self) -> dict | None:
        """最近一条已完成的状态。附 `age_s`：这条状态距现在多久（面板要显示"现在"就自己加上它）。"""
        with self._lock:
            row = dict(self._latest) if self._latest else None
        if row:
            row["age_s"] = time.time() - row["produced_at"]
        return row

    def at(self, video: str, t_video: float) -> dict | None:      # 与 OracleGameState 同名，便于替换
        return self.latest()

    # ---------- 生命周期 ----------
    def close(self, exit_process: bool = True) -> None:
        """**必须调用**：vLLM 的 EngineCore 是独立子进程，不显式关会被 init 收养并一直占 85% 显存。"""
        self._stop.set()
        self._worker.join(timeout=5)
        # 感知侧 2026-09-07 加了 shutdown(exit_process=False)：只关引擎、进程继续活着，
        # 实测显存 5 秒内归还（82.6G → 1151 MiB），30 秒无回涨。所以直接用它，不再自己绕。
        # ⚠️ 关完之后模块里的引擎就不可用了，要再推理必须重起进程 ——
        #    所以这只能用于"进程退出前的清理"，不能用于"换场景时释放、之后再用"。
        fn = getattr(self._M, "shutdown", None)
        if fn:
            try:
                fn(exit_process=exit_process)
            except TypeError:                       # 老版本只接受位置参数且必然 os._exit
                if exit_process:
                    fn(0)
            except SystemExit:
                raise
            except Exception as e:
                print(f"engine shutdown 失败: {e!r}", file=sys.stderr)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # 退出 with 块只关引擎、不杀进程（教练是常驻的）。EngineCore 是独立子进程，
        # 不显式关会被 init 收养并一直占 85% 显存（感知侧因此泄过 3 张卡 249G）。
        self.close(exit_process=False)
        return False
