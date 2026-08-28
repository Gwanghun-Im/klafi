"""Event Framework (요구사항 §24).

KLAFI 내부 Component 간 강한 결합을 줄이기 위한 실행 Event 모델.
Monitoring / Evaluation / Audit / Billing 등이 Event를 구독한다.

- 구독자 없으면 publish는 near-noop.
- 구독자 예외는 격리(fail-open): 한 구독자의 오류가 Agent나 다른 구독자를 막지 않는다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from klafi.core.context import get_context

_log = logging.getLogger("klafi.events")


class EventType(str, Enum):  # §24 대표 Event
    ExecutionStarted = "ExecutionStarted"
    ExecutionCompleted = "ExecutionCompleted"
    ExecutionFailed = "ExecutionFailed"
    AgentStarted = "AgentStarted"
    AgentCompleted = "AgentCompleted"
    AgentFailed = "AgentFailed"
    NodeStarted = "NodeStarted"
    NodeCompleted = "NodeCompleted"
    NodeFailed = "NodeFailed"
    ToolStarted = "ToolStarted"
    ToolCompleted = "ToolCompleted"
    ToolFailed = "ToolFailed"
    ModelCalled = "ModelCalled"
    ApprovalRequested = "ApprovalRequested"
    ApprovalCompleted = "ApprovalCompleted"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Event:
    type: EventType
    execution_id: str | None = None
    agent_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now)


Subscriber = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._subs: list[tuple[set[EventType] | None, Subscriber]] = []

    def subscribe(self, handler: Subscriber, types: list[EventType] | None = None) -> None:
        self._subs.append((set(types) if types else None, handler))

    def publish(self, event: Event) -> None:
        for types, handler in self._subs:
            if types is not None and event.type not in types:
                continue
            try:
                handler(event)
            except Exception as exc:  # noqa: BLE001 — 구독자 오류 격리(fail-open)
                _log.warning("event subscriber 오류 무시: %s", exc)

    def clear(self) -> None:
        self._subs.clear()


# 기본 전역 Bus
EVENTS = EventBus()


def subscribe(handler: Subscriber, types: list[EventType] | None = None) -> None:
    EVENTS.subscribe(handler, types)


def emit(type: EventType, **data: Any) -> None:
    """현재 ExecutionContext에서 execution_id/agent_id를 채워 Event를 발행."""
    ctx = get_context()
    EVENTS.publish(
        Event(
            type=type,
            execution_id=ctx.execution_id if ctx else None,
            agent_id=ctx.agent_id if ctx else None,
            data=data,
        )
    )
