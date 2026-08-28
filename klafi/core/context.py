"""ExecutionContext — KLAFI 공통 실행정보 전달 객체 (요구사항 §9, F04).

원칙: Global 변수 금지. ContextVar로 실행 Scope 기반 전파하여 async/thread 안전.
비즈니스 Node는 get_context()로 Logger·사용자·인증·Trace 정보를 꺼내 쓴다.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

# 실행 Scope 기반 전달 (CTX-08/09/10). 절대 module-global 상태로 쓰지 않는다.
_current: ContextVar["ExecutionContext | None"] = ContextVar(
    "klafi_execution_context", default=None
)


@dataclass
class ExecutionContext:
    execution_id: str
    trace_id: str | None = None
    agent_id: str | None = None
    agent_version: str | None = None
    project_id: str | None = None
    user_id: str | None = None
    tenant_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    security_context: dict[str, Any] = field(default_factory=dict)
    runtime_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    state: str = "CREATED"  # Execution 상태 (klafi.runtime.state.ExecutionState 값)

    @classmethod
    def new(cls, **kwargs: Any) -> "ExecutionContext":
        """execution_id/trace_id를 자동 발급하며 생성 (EXE-12/13)."""
        kwargs.setdefault("execution_id", uuid.uuid4().hex)
        kwargs.setdefault("trace_id", kwargs["execution_id"])
        return cls(**kwargs)


def get_context() -> ExecutionContext | None:
    """현재 실행 Scope의 Context. 실행 밖에서는 None."""
    return _current.get()


def require_context() -> ExecutionContext:
    ctx = _current.get()
    if ctx is None:
        from .exceptions import ContextException

        raise ContextException("활성 ExecutionContext가 없습니다")
    return ctx


@contextmanager
def bind_context(ctx: ExecutionContext) -> Iterator[ExecutionContext]:
    """with 블록 동안 ctx를 현재 실행 Context로 바인딩."""
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)
