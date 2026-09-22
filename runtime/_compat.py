"""把抽出来的 vLLM-Omni streaming runtime 与 vllm/vllm_omni 解耦。

原文件对上游只有两处硬依赖：
  1. `from vllm.logger import init_logger`      → 换 stdlib logging
  2. `from vllm_omni.outputs import OmniRequestOutput` → 换下面这个鸭子类型

`OmniRequestOutput` 在 base handler 里只用到三件事（读源码得到）：
  - `.final_output_type == "text"`   判断这条 output 是不是文本
  - `.outputs[0].text`               累积文本（不是增量，增量由 handler 自己算）
  - `.multimodal_output['audio']`    音频路径，本项目暂不用
所以复刻这个形状即可，不需要 vllm_omni 的任何代码。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any


def init_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@dataclass
class TextOutput:
    """对应 vllm RequestOutput.outputs[i]。text 是**累积**文本。"""
    text: str = ""
    finish_reason: str | None = None


@dataclass
class OmniRequestOutput:
    """base handler 唯一依赖的输出形状。backend 产出这个类型即可。"""
    request_id: str = ""
    final_output_type: str = "text"          # "text" | "audio" | ...
    outputs: list[TextOutput] = field(default_factory=list)
    finished: bool = False
    multimodal_output: dict[str, Any] = field(default_factory=dict)


class ChatCompletionRequest:
    """`ChatCompletionRequest` 的最小替身。

    base handler 只读这几个属性（grep 得到）：messages / add_generation_prompt /
    add_special_tokens / continue_final_message，外加 getattr(request, "chat_template", None)。
    上游用的是 vllm 的 pydantic 模型（为了送进 vllm 的 chat 预处理）；我们走 HTTP，
    预处理在服务端做，这里只需要一个属性袋。
    """

    def __init__(self, **kwargs: Any) -> None:
        self.messages = kwargs.pop("messages", [])
        self.model = kwargs.pop("model", "default")
        self.stream = kwargs.pop("stream", True)
        self.add_generation_prompt = kwargs.pop("add_generation_prompt", True)
        self.continue_final_message = kwargs.pop("continue_final_message", False)
        self.add_special_tokens = kwargs.pop("add_special_tokens", False)
        self.chat_template = kwargs.pop("chat_template", None)
        self.extra = kwargs          # modalities / sampling_params_list / mm_processor_kwargs 等
