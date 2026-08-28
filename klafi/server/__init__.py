"""서버 레이어.

AgentServer(순수 런타임 레지스트리)는 FastAPI를 모른다 — `[server]` extra 없이도 import 된다.
create_app / mount_frontend 만 FastAPI 에 의존하므로, 그 둘은 접근 시점에 lazy 로 로드한다
(PEP 562). 이렇게 해야 HTTP 를 쓰지 않는 demo.py·CLI·테스트가 fastapi 미설치 환경에서 돈다.
"""

from typing import Any

from .registry import AgentNotFound, AgentServer

__all__ = ["AgentServer", "AgentNotFound", "create_app", "mount_frontend"]


def __getattr__(name: str) -> Any:  # PEP 562 — fastapi 의존 심볼만 지연 로드
    if name == "create_app":
        from .http import create_app

        return create_app
    if name == "mount_frontend":
        from .webui import mount_frontend

        return mount_frontend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
