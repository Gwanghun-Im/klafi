"""ContextHook — 히스토리 자동 관리 (요구사항 §10.3, CNT-01~04).

Node 진입 시 State의 대화 히스토리가 Threshold를 넘으면 ContextManager로 압축한다.
압축은 **노드가 보는 view**에만 적용되므로 모델 입력 토큰이 줄고,
Checkpoint에는 원본 히스토리가 그대로 남는다(감사·재현 목적).

배선: config/context.yaml + config/hooks.yaml 의 `hooks: [context]` (공통개발자 영역).
업무개발자 코드에는 나타나지 않는다.
"""

from __future__ import annotations

from typing import Any, Callable

from klafi.core.context import ExecutionContext
from klafi.core.hook import Hook

from .manager import ContextManager


class ContextHook(Hook):
    priority = 15  # Guardrail(1)·Tracing(5)·Logging(10) 다음
    fail_open = True  # 압축 실패가 업무를 막지 않는다

    def __init__(
        self,
        max_tokens: int = 3000,
        keep_recent: int = 4,
        summarizer: Callable[[str], str] | None = None,
        token_counter: Callable[[str], int] | None = None,
        key: str = "messages",
    ) -> None:
        self._key = key
        self._cm = ContextManager(
            token_counter, max_tokens=max_tokens, keep_recent=keep_recent, summarizer=summarizer
        )

    def before_node(self, node: str, state: Any, ctx: ExecutionContext | None) -> None:
        if not isinstance(state, dict):
            return
        messages = state.get(self._key)
        if not messages or not self._cm.over_threshold(messages):
            return
        state[self._key] = self._cm.reduce(messages)  # in-place: 이 노드의 view만 축소
