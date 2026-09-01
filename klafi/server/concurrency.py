"""동시 실행 상한(백프레셔) — 순수 ASGI 미들웨어. 2단계(전역 + 에이전트별).

세마포어 획득/반납을 엔드포인트마다 복붙하지 않고 한 곳에서 관리한다.
- 실행 엔드포인트(에이전트 호출)만 슬롯을 센다. health·목록 조회는 세지 않는다.
- **2단계**: 요청은 전역 총량 세마포어 AND 해당 에이전트 세마포어를 **둘 다** 획득해야 실행된다.
  전역은 서버 과부하를, 에이전트별은 특정 에이전트의 독식을 막는다. 어느 하나라도 실패하면
  이미 잡은 슬롯을 반납하고 429.
- **스트리밍이 끝날 때까지 슬롯을 점유**해야 하므로 순수 ASGI 로 작성한다.
  starlette 의 BaseHTTPMiddleware(=@app.middleware)는 StreamingResponse 를 재포장해
  응답 완료 시점을 놓치므로 쓰지 않는다. send 이벤트의 more_body=False 로 본문 전송 완료를
  직접 감지해 그 순간 반납한다.
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


def _agent_id(scope: dict) -> "str | None":
    # 경로 /agents/<id>/invoke|resume|stream|resume/stream → 두 번째 세그먼트가 agent_id.
    parts = scope.get("path", "").strip("/").split("/")
    return parts[1] if len(parts) >= 2 and parts[0] == "agents" else None


class _ConcurrencyLimiter:
    """ASGI 미들웨어: 실행 엔드포인트에 2단계(전역 + 에이전트별) 세마포어 백프레셔."""

    def __init__(self, app: Any, global_max: "int | None" = None, per_agent: "dict[str, int] | None" = None) -> None:
        self.app = app
        self.global_max = global_max
        self._global = threading.BoundedSemaphore(global_max) if global_max and global_max > 0 else None
        limits = {a: n for a, n in (per_agent or {}).items() if n and n > 0}
        self._agent_limits = limits
        self._agents = {a: threading.BoundedSemaphore(n) for a, n in limits.items()}

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if not _is_counted(scope):
            await self.app(scope, receive, send)
            return

        aid = _agent_id(scope)
        asem = self._agents.get(aid)
        acquired: list[threading.BoundedSemaphore] = []

        # 전역 먼저, 그다음 에이전트별. 어느 하나 실패하면 잡은 것 반납 후 429.
        if self._global is not None:
            if not self._global.acquire(blocking=False):
                await self._reject(send, "global", self.global_max)
                return
            acquired.append(self._global)
        if asem is not None:
            if not asem.acquire(blocking=False):
                for s in acquired:
                    s.release()
                await self._reject(send, aid, self._agent_limits.get(aid))
                return
            acquired.append(asem)

        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                for s in acquired:
                    s.release()

        async def send_wrapper(message: dict) -> None:
            # 본문 전송이 끝나는 순간(더 보낼 body 없음) 슬롯 반납. 스트리밍이면 마지막 청크 시점.
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                release()
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            release()  # 예외·조기 종료·클라이언트 끊김 등 모든 경로에서 반납 보장(멱등)

    async def _reject(self, send: Any, scope_label: "str | None", limit: "int | None") -> None:
        body = json.dumps(
            {"error": "동시 실행 상한 도달", "scope": scope_label, "limit": limit},
            ensure_ascii=False,
        ).encode()
        await send({
            "type": "http.response.start",
            "status": 429,
            "headers": [(b"content-type", b"application/json"), (b"retry-after", b"1")],
        })
        await send({"type": "http.response.body", "body": body})


def install_concurrency_limit(
    app: Any, global_max: "int | None", per_agent: "dict[str, int] | None" = None
) -> None:
    """실행 엔드포인트에 2단계 세마포어 백프레셔를 건다.

    global_max>0 이면 전역 총량 캡, per_agent 는 {agent_id: n} 에이전트별 캡. 둘 다 비면 무제한(미설치).
    """
    has_agent = any(n and n > 0 for n in (per_agent or {}).values())
    if (not global_max or global_max <= 0) and not has_agent:
        return
    app.add_middleware(_ConcurrencyLimiter, global_max=global_max, per_agent=per_agent)
