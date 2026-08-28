"""T02. RAG Agent — Question → Retriever → LLM → Response."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START

from klafi.core.graph import KlafiGraph
from klafi.core.node import klafi_node
from klafi.core.spec import AgentSpec
from klafi.observability.tracing import span

from .base import Model, Retriever


class RAGState(TypedDict):
    question: str
    context: list[str]
    answer: str


class RAGAgent(KlafiGraph):
    state_schema = RAGState

    def __init__(self, model: Model, retriever: Retriever, *, spec: AgentSpec | None = None, **kwargs: Any) -> None:
        self._retrieve = retriever
        super().__init__(
            spec or AgentSpec(id="rag", name="RAG Agent", version="0.1.0", agent_type="rag"),
            model=model,
            **kwargs,
        )

    def define(self) -> None:
        @klafi_node("retrieve")
        def retrieve(state: RAGState) -> RAGState:
            with span("tool.retriever"):
                return {"context": self._retrieve(state["question"])}

        @klafi_node("generate")
        def generate(state: RAGState) -> RAGState:
            with span("model.llm"):
                context = "\n".join(state["context"])
                prompt = f"참고자료:\n{context}\n\n질문: {state['question']}"
                return {"answer": self.model(prompt)}

        self.add_node("retrieve", retrieve)
        self.add_node("generate", generate)
        self.add_edge(START, "retrieve")
        self.add_edge("retrieve", "generate")
        self.add_edge("generate", END)
