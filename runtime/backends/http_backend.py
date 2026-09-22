"""ModelBackend：把 OpenAI 兼容的 HTTP 服务包装成 base handler 要的 engine_client。

base handler 对 engine_client 的全部要求（读源码得到，就两条）：
    async for out in engine_client.generate(prompt=..., request_id=..., output_modalities=...)
    await engine_client.abort(request_id)
`out` 只需要有 `.final_output_type == "text"` 和 `.outputs[0].text`（**累积**文本）。

打断说明：HTTP 上没有真正的 engine 侧 abort。这里的 abort 关掉流式响应，
让 token 不再回来；vLLM 的 api_server 检测到客户端断连**通常**也会 abort 掉请求，
但那是它的行为不是我们的保证 —— 未实测，别当成 GPU 一定被释放。
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

import httpx

from runtime._compat import OmniRequestOutput, TextOutput

logger = logging.getLogger(__name__)


class HttpOpenAIBackend:
    def __init__(self, base_url: str, model: str, timeout: float = 300.0,
                 max_tokens: int = 256, temperature: float = 0.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = httpx.AsyncClient(timeout=timeout)
        self._live: dict[str, bool] = {}     # request_id -> 是否还该继续

    async def generate(self, prompt: Any, request_id: str,
                       output_modalities: Any = None) -> AsyncGenerator[OmniRequestOutput, None]:
        """prompt 就是 messages（我们的 _preprocess_to_engine_prompt 直接透传）。"""
        del output_modalities
        self._live[request_id] = True
        try:
            _ts = [c.get("text") for m in prompt for c in (m.get("content") or [])
                   if isinstance(c, dict) and str(c.get("text","")).endswith(" seconds>")]
            _ni = sum(1 for m in prompt for c in (m.get("content") or [])
                      if isinstance(c, dict) and c.get("type") == "image_url")
            logger.debug("[PROMPT] 图 %d 张，时间戳 %s", _ni, _ts)
        except Exception:
            pass
        payload = {
            "model": self.model,
            "messages": prompt,
            "stream": True,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        acc = ""
        try:
            async with self._client.stream("POST", f"{self.base_url}/v1/chat/completions",
                                           json=payload) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")[:300]
                    raise RuntimeError(f"backend HTTP {resp.status_code}: {body}")
                async for line in resp.aiter_lines():
                    if not self._live.get(request_id):
                        logger.info("backend: request %s aborted, closing stream", request_id)
                        break
                    if not line.startswith("data: "):
                        continue
                    body = line[6:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        delta = json.loads(body)["choices"][0]["delta"].get("content")
                    except Exception:
                        continue
                    if not delta:
                        continue
                    acc += delta
                    yield OmniRequestOutput(
                        request_id=request_id, final_output_type="text",
                        outputs=[TextOutput(text=acc)], finished=False)
        finally:
            self._live.pop(request_id, None)
        yield OmniRequestOutput(request_id=request_id, final_output_type="text",
                                outputs=[TextOutput(text=acc, finish_reason="stop")], finished=True)

    async def abort(self, request_id: str) -> None:
        """标记停止；生成循环下一次迭代就会 break 并关闭 HTTP 流。"""
        if request_id in self._live:
            self._live[request_id] = False
            logger.info("backend: abort requested for %s", request_id)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def read_image(self, system: str, user: str, image_b64: str,
                         max_tokens: int = 64) -> str:
        """记录员用：对一张小裁剪图做一次性非流式问答。与 generate() 无关，不参与打断。"""
        payload = {
            "model": self.model, "max_tokens": max_tokens, "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": "data:image/jpeg;base64," + image_b64}},
                    {"type": "text", "text": user}]}],
        }
        r = await self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
