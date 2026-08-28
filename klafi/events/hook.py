"""EventHook — Agent/Node 생명주기 Event 발행 (§24).

Agent에 붙이면 실행 흐름이 Event로 발행되어 Monitoring/Audit 등이 구독할 수 있다.
발행 자체가 fail-open(EventBus가 구독자 예외를 격리)이라 Hook도 fail_open.
"""

from __future__ import annotations

from typing import Any

from klafi.core.context import ExecutionContext
from klafi.core.hook import Hook

from .bus import EventType, emit


class EventHook(Hook):
    priority = 8
    fail_open = True

    def before_agent(self, input: Any, ctx: ExecutionContext | None) -> None:
        emit(EventType.ExecutionStarted)
        emit(EventType.AgentStarted)

    def after_agent(self, input: Any, result: Any, ctx: ExecutionContext | None) -> None:
        emit(EventType.AgentCompleted)
        emit(EventType.ExecutionCompleted)

    def on_agent_error(self, input: Any, exc: BaseException, ctx: ExecutionContext | None) -> None:
        emit(EventType.AgentFailed, error=str(exc))
        emit(EventType.ExecutionFailed, error=str(exc))

    def before_node(self, node: str, state: Any, ctx: ExecutionContext | None) -> None:
        emit(EventType.NodeStarted, node=node)

    def after_node(self, node: str, state: Any, result: Any, ctx: ExecutionContext | None) -> None:
        emit(EventType.NodeCompleted, node=node)

    def on_node_error(self, node: str, state: Any, exc: BaseException, ctx: ExecutionContext | None) -> None:
        emit(EventType.NodeFailed, node=node, error=str(exc))
