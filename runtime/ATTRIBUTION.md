# 来源与许可

`runtime/` 目录下有 4 个文件抽自 **vLLM-Omni**，并做了修改。

| | |
|---|---|
| 上游 | https://github.com/vllm-project/vllm-omni |
| 许可 | Apache License 2.0（全文见 `LICENSE.upstream-vllm-omni`） |
| 抽取时的 SHA | `b81aeb7b86837f6fe8956f3aef83798ad26c5a26`（也记在 `UPSTREAM_SHA`） |
| 抽取日期 | 2026-08-31 |

## 抽了哪些、改了什么

| 文件 | 行数 | 修改 |
|---|---|---|
| `runtime/video_stream_base.py` | 1090 | **有修改** —— 6 处 import 改为本地实现，见 `runtime/EXTRACTION.md` |
| `runtime/video_frame_filter.py` | 116 | 未修改 |
| `runtime/video_stream_context.py` | 28 | 未修改 |
| `runtime/video_stream_envs.py` | 62 | 未修改 |

这些文件的 SPDX 头（`SPDX-License-Identifier: Apache-2.0` /
`SPDX-FileCopyrightText: Copyright contributors to the vLLM project`）**原样保留**。

## 本仓库自己写的部分

`runtime/_compat.py`、`runtime/pipeline.py`、`runtime/observer.py`、`runtime/backends/`、`runtime/app.py` 为本项目原创。

## 为什么是抽取而不是依赖

`video_stream_base.py` 首次出现在 vllm-omni v0.24.0，该版本线对齐上游 vLLM 0.24；
当时的机器受 H20 驱动 535 约束，vLLM 上限 0.19.1，装不上也跑不起来。
详见 `runtime/EXTRACTION.md`。
