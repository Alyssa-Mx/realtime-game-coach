# 从 vLLM-Omni 抽 streaming runtime —— 改了什么、为什么

上游：`vllm-project/vllm-omni`，Apache-2.0，SHA 见 `../UPSTREAM_SHA`（`b81aeb7b`）。

## 为什么是"抽"而不是 `pip install -e .`

`video_stream_base.py` **首次出现在 v0.24.0**（v0.18.0 / v0.20.0 / v0.22.0 都没有），
而 vllm-omni 的发布对齐上游 vllm 的偶数小版本 —— v0.24.0 对齐 **vllm 0.24**。

本机天花板是 **vllm 0.19.1**（H20 驱动 535；vllm 0.20+ 钉 torch 2.11/cudnn-cu13，要驱动 ≥580）。
另外它还要求 `transformers >= 5.10.1`，我们是 4.57.6。

→ **装不上，也跑不起来。** 但要的东西（协议 / 缓冲 / 过滤 / 打断编排）全是纯 asyncio，
跟推理引擎无关（`engine_client` 是构造时注入的 `Any | None`），所以抽出来反而更干净。

## 抽了哪 4 个文件（共 1296 行）

| 文件 | 行数 | 动了什么 |
|---|---|---|
| `video_stream_base.py` | 1090 | 只改 5 处 import |
| `video_frame_filter.py` | 116 | 原样 |
| `video_stream_context.py` | 28 | 原样 |
| `video_stream_envs.py` | 62 | 原样 |

**没抽** `serving_video_stream.py`（152 行）—— 那是 Qwen-Omni 的 pipeline，正是要替换掉的那层。

## 改的 5 处 import

| 原 | 改成 |
|---|---|
| `from vllm.logger import init_logger` | `from ._compat import init_logger`（stdlib logging） |
| `from vllm_omni.entrypoints.openai import video_stream_envs` | `from . import video_stream_envs` |
| `...video_frame_filter import FrameSimilarityFilter` | `from .video_frame_filter import ...` |
| `...video_stream_context import (...)` | `from .video_stream_context import (...)` |
| `from vllm_omni.outputs import OmniRequestOutput` | `from ._compat import OmniRequestOutput` |
| `from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest`（第 623 行，函数内） | `from ._compat import ChatCompletionRequest` |

`_compat.py` 里的两个替身是**读源码得出的最小形状**：
- `OmniRequestOutput`：只用到 `.final_output_type == "text"` 和 `.outputs[0].text`（**累积**文本，
  增量由 handler 自己算）；音频路径用 `.multimodal_output['audio']`，本项目暂不用。
- `ChatCompletionRequest`：只读 `.messages` / `.add_generation_prompt` / `.add_special_tokens` /
  `.continue_final_message`，外加 `getattr(request, "chat_template", None)`。

## 没解决的一处

`_encode_audio_wav_b64`（第 980-981 行）仍 import `vllm_omni` 的 `AudioMixin` / `CreateAudio`。
**音频路径专用，文本模式跑不到。** 要开音频输出时再处理。

## 我们自己写的

| 文件 | 作用 |
|---|---|
| `runtime/_compat.py` | 上面两个替身 + logger |
| `runtime/pipeline.py` | 填三个 hook（目前都是最简实现）+ 覆盖 `_preprocess_to_engine_prompt` |
| `backends/http_backend.py` | 把 OpenAI 兼容 HTTP 服务包成 `engine_client`（`generate` + `abort`） |
| `app.py` | FastAPI，挂在 `/v1/video/chat/stream`，**只绑 127.0.0.1** |

**为什么要覆盖 `_preprocess_to_engine_prompt`**：上游用 `self._chat_service.renderer` +
`handler._preprocess_chat(...)`，那是 vllm `serving_chat` 的内部方法（做 tokenize 和多模态预处理）。
我们走 HTTP，这些在服务端做，所以直接透传 `request.messages`。

## 第一步验证（2026-08-31，后端 = 本地 Qwen3-VL-8B，OpenAI 兼容接口）

素材：一段对局录屏的 60 帧。

| 要验的 | 结果 |
|---|---|
| `video.frame` + ack | ✅ 60 帧 60 条 ack |
| EVS 过滤 | ✅ `accepted: False, reason: 'filtered'`，但**默认阈值 0.95 丢了约 93%**（60 帧只留 4~5 帧） |
| buffer 淘汰 | ✅ 关掉 EVS 后：`buffered_frames=16` 封顶，`dropped_frame_id='f0004'` → `'f0024'`，oldest-first |
| `num_frames` stride 采样 | ✅ `[CONSUMED] 4 帧: ['f0044','f0048','f0052','f0059']` |
| 手动 `video.query` → 出字 | ✅ `'玩家用GROZA击倒敌人。'` |
| **interrupt** | ✅ 见下 |

### interrupt 的证据

第一条 query 生成中插第二条：

```
21:44:33.905  POST /v1/chat/completions 200                 ← query#1
21:44:35.210  Interrupt signaled for video-fe4592220bbf     ← query#2 打断 #1
21:44:35.339  POST /v1/chat/completions 200                 ← query#2
21:44:35.679  [TIMING] total=0.37s first_text=0.04s         ← 只有这一条
```

**`[TIMING]` 每轮完成时打一次，两条 query 只出现一条 → query#1 从未走到完成路径。**
说明软打断 + task cancel 这条链**走 HTTP 也成立**，不需要进程内 engine 的 `abort()`。
（`first_text=0.04s` 是 8B 的 prefix cache 命中，两条 query 共享同一批图的前缀。）


## 对上游的有意分叉（2026-09-01）

**`build_engine_prompt` 多收一个 `frame_metadata` 参数**（带默认值 `None`，不破坏上游调用方式）。

改了三处签名（Protocol、基类、`_process_query_engine` 的调用点），原因：
每帧要配 `<X.X seconds>` 时间戳，而 `pts_ms` 只存在于 `frame_metadata`，
上游没有把它传进 hook。`frame_metadata` 与 `frame_buffer` 本来就同步 append / 同步淘汰，
按同一组下标取即可对齐。

**为什么要时间戳** —— 读 vllm `qwen3_vl.py` 得到的事实：

| 通道 | prompt 里的形态 |
|---|---|
| 图片 | `[image_token] * n`，**没有任何时间信息** |
| 视频 | 每帧前插文本 `<X.X seconds>`（`get_video_repl()`），再接 vision_start/end |

M-RoPE 的时间轴（`mrope_section: [24,20,20]` 第一段）只按帧递增
（`np.indices((1,h,w))` + `st_idx += 1`）——**编码次序，不编码间隔**。
所以送图片时模型不知道两帧隔了多久；写法逐字复刻视频通道，等于手工补上这条信息。

本地 8B 实测（12 帧覆盖 18 秒，问"击倒玩家47 在第几秒"，真值约 7 秒）：

| | prompt token | 回答 |
|---|---|---|
| 图片，无时间戳 | 10644 | `18:20 秒` ❌（去读 HUD 倒计时了） |
| 图片，加 `<X.X seconds>` | 10721 | `7秒` ✅ |
| 视频通道 mp4 | 6279 | `8秒` ✅ |

**为什么最终选"buffer 存图 + 时间戳"而不是封 mp4**：
在 `pubg-skill-router-v2` 上 A/B 各 3 次，两者在输出质量上**分辨不出差别**
（组间差 < 组内极差），token 差 1.2%。选它的真正理由是
**mp4 必须等一整段录完才能发，而帧可以随时从 buffer 取** —— 这正是 buffer 存在的意义。
实验记录见 `pubg-skill-router` 仓库的 `experiment/vision-mode-ab` 分支。

**时间戳基准**由 `TS_MODE` 控制，默认 `window`（buffer 里最老的帧记作 0.0s）。
理由：值域小、留在训练分布内；每次 query 是无状态请求，绝对会话时间对模型没有意义，
有意义的是间隔。`session`（会话至今秒数）会产生 `<900.0 seconds>` 这类值，
**是否仍在训练分布内未验证**。

验收（2026-09-01，后端 = 本机 Qwen3-VL-8B）：客户端按 5fps 发 `pts_ms=i*200`，
选中 f0024/f0028/f0032/f0039（pts 4800/5600/6400/7800ms），
服务端实际发出 `['<0.0 seconds>','<0.8 seconds>','<1.6 seconds>','<3.0 seconds>']` —— 间隔与真实帧距一致。


## 2026-09-01 又两处分叉

**① `enable_frame_filter` 默认值 True → False。**

不是否定 EVS。它的价值场景是**画面长时间不变** —— AI 教练面对的新手会卡住不动、
不知道该干什么，那种几十秒没变化的片段确实不必逐帧判断。
但当前测试素材全是激战，画面一直在动，过滤只会丢信息
（实测 5fps 游戏画面上默认阈值 0.95 丢 90%，`max_frames` 淘汰因此从未触发）。
**接口完整保留**，`session.config` 传 `enable_frame_filter: true` 即可开启。

**② 选帧改成时间等距，且 `_sample_frame_metadata` 与之共用同一个纯函数。**

上游有两处独立做采样：`build_engine_prompt`（决定真正喂给模型的帧）和
`_sample_frame_metadata`（决定 `video.frames.consumed` 里报哪些帧），
两边各写了一遍按下标 stride —— **覆盖 hook 之后它们就会不一致**，
`video.frames.consumed` 报的不是真正喂进去的帧。实测撞到过：
hook 选 idx [0,5,9,14,19]，consumed 报的却是 [0,4,8,12,19]。

改法：抽出模块级纯函数 `select_indices(all_pts, k)`，两处共用。
优先按时间等距（在 [最老,最新] 之间取 k 个等距时刻，各找最近帧），
拿不到 `pts_ms` 时退回按下标等距。

为什么要时间等距：客户端帧率抖动时按下标等距在时间上分布不均。
实测同一 buffer（20 帧，pts 4000–7800ms，取 5 帧）：

| | 选中 pts (ms) | 间隔 (ms) |
|---|---|---|
| 按下标等距 | 4000/4800/5600/6400/7800 | 800/800/800/**1400** |
| 时间等距 | 4000/5000/5800/6800/7800 | 1000/800/1000/1000 |

## 已知问题 / 下一步

1. ~~EVS 默认阈值太狠~~ → 已按上面②处理：默认关闭，接口保留。原记录：**阈值 0.95 在游戏画面上太狠**（丢 93%），会毁掉时序。要么调阈值，要么只当辅助、
   用 OCR 闸门当强 trigger。注意：离线单独跑这个算法我量到的是丢 50%，与线上 93% 不一致，
   原因未查（采样间隔不同？缩略图实现差异？）——**这条待复核，别直接引用任一数字**。
2. `should_trigger_turn` 仍是 `return False`（空壳），下一步换成读仪表盘的 OCR 闸门。
3. `build_engine_prompt` 仍是照抄上游的 stride 采样，下一步换 v2 的 `build_messages`。
4. 空闲 60 秒会 `Idle timeout` 关会话（上游默认），长跑要调 `idle_timeout`。
