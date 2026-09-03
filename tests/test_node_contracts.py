"""노드 전달 계약 — @klafi_node(visibility=..., output=Schema).

internal: 토큰·updates 미전달(상태는 유지). output: 스키마 검증·강제, 스트림에서 structured 청크 1회, 메타데이터 노출.
"""

from __future__ import annotations

import json
from typing import TypedDict

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import Annotated

from klafi import AgentSpec, KlafiGraph
from klafi.core import klafi_node
from klafi.core.exceptions import AgentExecutionException, ValidationError
from klafi.server import AgentServer, create_app


class Report(BaseModel):
    title: str
    score: int


class St(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    route: str
    report: Report


class Pipeline(KlafiGraph):
    """route(internal, LLM 이 'skip' 토큰을 냄) → extract(output=Report) → answer(external)."""

    spec = AgentSpec(id="pipe", name="Pipe")
    state_schema = St
    observability = False

    def define(self):
        router = FakeListChatModel(responses=["skip"])
        writer = FakeListChatModel(responses=["final answer"])

        @klafi_node("route", visibility="internal")
        def route(state):
            return {"route": router.invoke(state["messages"]).content}

        @klafi_node("extract", output=Report)
        def extract(state):
            return {"report": {"title": "T", "score": 3}}  # dict → Report 로 강제

        @klafi_node("answer")
        def answer(state):
            return {"messages": [writer.invoke(state["messages"])]}

        self.add_node("route", route)
        self.add_node("extract", extract)
        self.add_node("answer", answer)
        self.add_edge(START, "route")
        self.add_edge("route", "extract")
        self.add_edge("extract", "answer")
        self.add_edge("answer", END)


def _inp():
    return {"messages": [HumanMessage("q")]}


def test_invoke_keeps_state_and_coerces_output():
    out = Pipeline().invoke(_inp())
    assert out["route"] == "skip"  # internal 은 상태를 숨기지 않는다
    assert isinstance(out["report"], Report) and out["report"].score == 3


def test_stream_hides_internal_and_emits_structured_once():
    items = list(Pipeline().stream(_inp(), stream_mode=["updates", "messages"]))
    tokens = "".join(p[0].content for m, p in items if m == "messages")
    assert "skip" not in tokens and tokens == "final answer"  # 내부 노드 토큰 미전달, extract 토큰 없음
    assert not any(m == "updates" and "route" in p for m, p in items)  # internal updates 미전달
    structured = [p for m, p in items if m == "structured"]
    assert structured == [{"node": "extract", "key": "report", "data": Report(title="T", score=3)}]
    assert list(Pipeline().stream(_inp(), stream_mode="updates"))  # 단일 모드는 모양 유지(structured 미삽입)


def test_output_contract_validation_and_key_forms():
    class Bad(KlafiGraph):
        spec = AgentSpec(id="bad", name="Bad")
        state_schema = St
        observability = False

        def define(self):
            @klafi_node("extract", output=Report)
            def extract(state):
                return {"report": {"title": "T"}}  # score 누락

            self.add_node("extract", extract)
            self.add_edge(START, "extract")
            self.add_edge("extract", END)

    with pytest.raises(ValidationError):
        Bad().invoke(_inp())

    class Explicit(KlafiGraph):
        spec = AgentSpec(id="ex", name="Ex")
        state_schema = St
        observability = False

        def define(self):
            @klafi_node("extract", output=("report", Report))
            def extract(state):
                return {"report": Report(title="T", score=1), "route": "x"}

            self.add_node("extract", extract)
            self.add_edge(START, "extract")
            self.add_edge("extract", END)

    assert Explicit().invoke(_inp())["report"].score == 1
    with pytest.raises(AgentExecutionException):
        klafi_node("n", visibility="hidden")(lambda s: s)
    with pytest.raises(AgentExecutionException):
        klafi_node("n", output=dict)(lambda s: s)


def test_metadata_and_http_expose_contracts_and_structured_line():
    server = AgentServer()
    server.register(Pipeline())
    client = TestClient(create_app(server))
    nodes = client.get("/agents/pipe").json()["nodes"]
    assert nodes["route"]["visibility"] == "internal" and nodes["answer"]["visibility"] == "external"
    assert nodes["extract"]["output_schema"]["title"] == "Report" and nodes["extract"]["output_key"] is None

    r = client.post("/agents/pipe/invoke", json={"input": {"messages": [{"role": "user", "content": "q"}]}})
    assert r.json()["result"]["report"] == {"title": "T", "score": 3}  # pydantic → JSON

    with client.stream("POST", "/agents/pipe/stream", json={"input": {"messages": [{"role": "user", "content": "q"}]}}) as s:
        lines = [json.loads(l) for l in s.iter_lines() if l.strip()]
    assert "".join(l["token"] for l in lines if "token" in l) == "final answer"
    assert [l["structured"] for l in lines if "structured" in l] == [{"node": "extract", "key": "report", "data": {"title": "T", "score": 3}}]
    assert not any("route" in (l.get("chunk") or {}) for l in lines)
