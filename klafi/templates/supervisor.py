"""T03. Supervisor Multi-Agent — Supervisor가 Worker를 라우팅."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START

from klafi.core.graph import KlafiGraph
from klafi.core.node import klafi_node
from klafi.core.spec import AgentSpec
from klafi.observability.tracing import span

FINISH = "FINISH"

Router = Callable[[dict], str]
Worker = Callable[[dict], Any]


def _merge(a: dict, b: dict) -> dict:
    return {**a, **b}


class SupervisorState(TypedDict):
    task: str
    next: str
    results: Annotated[dict, _merge]
    history: Annotated[list, operator.add]


class SupervisorAgent(KlafiGraph):
    state_schema = SupervisorState

    def __init__(
        self,
        router: Router,
        workers: dict[str, Worker],
        max_steps: int = 10,
        *,
        spec: AgentSpec | None = None,
        **kwargs: Any,
    ) -> None:
        if FINISH in workers:
            raise ValueError(f"worker 이름에 예약어 {FINISH!r}를 쓸 수 없습니다")
        self._router = router
        self._workers = workers
        self._max_steps = max_steps
        super().__init__(
            spec or AgentSpec(id="supervisor", name="Supervisor Agent", version="0.1.0", agent_type="supervisor"),
            **kwargs,
        )

    def define(self) -> None:
        @klafi_node("supervisor")
        def supervisor(state: SupervisorState) -> dict:
            if len(state.get("history", [])) >= self._max_steps:
                return {"next": FINISH}
            with span("supervisor.route"):
                return {"next": self._router(state)}

        self.add_node("supervisor", supervisor)
        for name, fn in self._workers.items():
            self.add_node(name, self._worker_node(name, fn))

        self.add_edge(START, "supervisor")
        routes = {name: name for name in self._workers}
        routes[FINISH] = END
        self.add_conditional_edges("supervisor", lambda s: s["next"], routes)
        for name in self._workers:
            self.add_edge(name, "supervisor")

    def _worker_node(self, name: str, fn: Worker) -> Callable[[dict], dict]:
        @klafi_node(name)
        def worker(state: SupervisorState) -> dict:
            with span(f"worker.{name}"):
                out = fn(state)
            return {"results": {name: out}, "history": [name]}

        return worker
