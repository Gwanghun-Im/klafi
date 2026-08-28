"""기본 제공 LoggingHook — DoD "개발자가 Node에 로깅 코드를 안 써도 로그 생성".

stdlib logging만 사용. Correlation을 위해 execution_id를 함께 남긴다(§16).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .context import ExecutionContext
from .hook import Hook

_log = logging.getLogger("klafi.node")


class LoggingHook(Hook):
    priority = 10  # 가장 바깥에서 감싸도록 일찍
    fail_open = True  # 로깅 실패가 업무를 막지 않음

    def __init__(self) -> None:
        self._t: dict[tuple[str, int], float] = {}

    def _eid(self, ctx: ExecutionContext | None) -> str:
        return ctx.execution_id if ctx else "-"

    def before_node(self, node: str, state: Any, ctx: ExecutionContext | None) -> None:
        self._t[(self._eid(ctx), id(state))] = time.perf_counter()
        _log.info("node.start node=%s execution_id=%s", node, self._eid(ctx))

    def after_node(self, node: str, state: Any, result: Any, ctx: ExecutionContext | None) -> None:
        t0 = self._t.pop((self._eid(ctx), id(state)), None)
        dur = f"{(time.perf_counter() - t0) * 1000:.1f}ms" if t0 else "-"
        _log.info("node.end node=%s execution_id=%s duration=%s", node, self._eid(ctx), dur)

    def on_node_error(self, node: str, state: Any, exc: BaseException, ctx: ExecutionContext | None) -> None:
        self._t.pop((self._eid(ctx), id(state)), None)
        _log.error("node.error node=%s execution_id=%s error=%r", node, self._eid(ctx), exc)
