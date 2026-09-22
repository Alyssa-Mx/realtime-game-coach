"""最小服务：把抽出来的 streaming runtime 挂在 /v1/video/chat/stream 上。

模型服务是任意 OpenAI 兼容接口（默认 localhost:8321）：
    BACKEND_URL=http://127.0.0.1:8321 python runtime/app.py
"""
from __future__ import annotations

import logging
import os
import sys

from fastapi import FastAPI, WebSocket

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backends.http_backend import HttpOpenAIBackend        # noqa: E402
from runtime.pipeline import LocalStreamingVideoHandler    # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("stream_runtime")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8321")
BACKEND_MODEL = os.getenv("BACKEND_MODEL", "qwen3vl8b")
PORT = int(os.getenv("PORT", "8400"))

backend = HttpOpenAIBackend(BACKEND_URL, BACKEND_MODEL)
handler = LocalStreamingVideoHandler(chat_service=None, engine_client=backend)
app = FastAPI()


@app.websocket("/v1/video/chat/stream")
async def video_stream(ws: WebSocket) -> None:
    await handler.handle_session(ws)


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "backend": BACKEND_URL, "model": BACKEND_MODEL}


if __name__ == "__main__":
    import uvicorn
    log.info("backend=%s model=%s port=%d", BACKEND_URL, BACKEND_MODEL, PORT)
    uvicorn.run(app, host="127.0.0.1", port=PORT)
