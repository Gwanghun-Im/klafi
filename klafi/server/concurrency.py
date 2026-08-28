"""동시 실행 상한(백프레셔) — 순수 ASGI 미들웨어.

세마포어 획득/반납을 엔드포인트마다 복붙하지 않고 한 곳에서 관리한다.
- 실행 엔드포인트(에이전트 호출)만 슬롯을 센다. health·목록 조회는 세지 않는다.
- **스트리밍이 끝날 때까지 슬롯을 점유**해야 하므로 순수 ASGI 로 작성한다.
  starlette 의 BaseHTTPMiddleware(=@app.middleware)는 StreamingResponse 를 재포장해
  응답 완료 시점을 놓치므로 쓰지 않는다. 여기서는 send 이벤트의 more_body=False 로
  본문 전송 완료를 직접 감지해 그 순간 반납한다.
초과 시 429 + Retry-After 로 즉시 거절한다.
"""

from __future__ import annotations

import json
import threading
from typing import Any

# 슬롯을 세는 경로 판별 — 실행 계열만. (읽기 전용 조회는 백프레셔 대상 아님)
_COUNTED_SUFFIXES = ("/invoke", "/resume", "/stream", "/resume/stream")


def _is_counted(scope: dict) -> bool:
    if scope.get("type") != "http" or scope.get("method") != "POST":
        return False
    path = scope.get("path", "")
    return path.startswith("/agents/") and path.endswith(_COUNTED_SUFFIXES)


class _ConcurrencyLimiter:
    """ASGI 미들웨어: 실행 엔드포인트에 세마포어 백프레셔."""

    def __init__(self, app: Any, max_concurrency: int) -> None:
        self.app = app
        self.max = max_concurrency
        self._sem = threading.BoundedSemaphore(max_concurrency)

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if not _is_counted(scope):
            await self.app(scope, receive, send)
            return

        if not self._sem.acquire(blocking=False):
            await self._reject(send)
            return

        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                self._sem.release()

        async def send_wrapper(message: dict) -> None:
            # 본문 전송이 끝나는 순간(더 보낼 body 없음) 슬롯 반납. 스트리밍이면 마지막 청크 시점.
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                release()
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            release()  # 예외·조기 종료·클라이언트 끊김 등 모든 경로에서 반납 보장(멱등)

    async def _reject(self, send: Any) -> None:
        body = json.dumps({"error": "동시 실행 상한 도달", "limit": self.max}).encode()
        await send({
            "type": "http.response.start",
            "status": 429,
            "headers": [(b"content-type", b"application/json"), (b"retry-after", b"1")],
        })
        await send({"type": "http.response.body", "body": body})


def install_concurrency_limit(app: Any, max_concurrency: int | None) -> None:
    """max_concurrency>0 이면 실행 엔드포인트에 세마포어 백프레셔를 건다. None/0 이면 무제한(미설치)."""
    if not max_concurrency or max_concurrency <= 0:
        return
    app.add_middleware(_ConcurrencyLimiter, max_concurrency=max_concurrency)
