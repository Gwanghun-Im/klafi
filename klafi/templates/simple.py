"""T01. Simple Agent — User → LLM → Response."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START

from klafi.core.graph import KlafiGraph
from klafi.core.node import klafi_node
from klafi.core.spec import AgentSpec
from klafi.observability.tracing import span

from .base import Model


class SimpleState(TypedDict):
    question: str
    answer: str


class SimpleAgent(KlafiGraph):
    state_schema = SimpleState

    def __init__(self, model: Model, *, spec: AgentSpec | None = None, **kwargs: Any) -> None:
        super().__init__(
            spec or AgentSpec(id="simple", name="Simple Agent", version="0.1.0", agent_type="simple"),
            model=model,
            **kwargs,
        )

    def define(self) -> None:
        @klafi_node("llm")
        def llm(state: SimpleState) -> SimpleState:
            with span("model.llm"):
                return {"answer": self.model(state["question"])}

        self.add_node("llm", llm)
        self.add_edge(START, "llm")
        self.add_edge("llm", END)
