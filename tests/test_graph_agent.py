"""KlafiGraph 검증 — 개발자가 상속해 그래프를 조립하는 표준 클래스."""

from typing import TypedDict

import pytest
from langgraph.graph import END, START
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from klafi import AgentSpec, ExecutionPolicy, KlafiGraph, klafi_node, setup_tracing
from klafi.core.exceptions import TimeoutException


class State(TypedDict):
    n: int


class LoopAgent(KlafiGraph):
    spec = AgentSpec(id="loop", name="Loop")
    state_schema = State

    def define(self):
        @klafi_node("step")
        def step(s):
            return {"n": s["n"] + 1}

        self.add_node("step", step)
        self.add_conditional_edges("step", lambda s: END if s["n"] >= 3 else "step", {"step": "step", END: END})
        self.add_edge(START, "step")


@pytest.fixture(scope="session")
def exporter():
    exp = InMemorySpanExporter()
    setup_tracing(exporter=exp, simple=True)
    return exp


@pytest.fixture(autouse=True)
def _clear(exporter):
    exporter.clear()
    yield


def test_subclass_define_runs_with_default_observability(exporter):
    agent = LoopAgent()  # spec은 클래스 속성
    out = agent.invoke({"n": 0})
    assert out["n"] == 3  # 커스텀 루프 동작
    names = {s.name for s in exporter.get_finished_spans()}
    assert "agent.loop" in names and "node.step" in names  # Logging/Tracing 기본 탑재


def test_enterprise_features_apply():
    agent = LoopAgent(checkpointer="memory", policy=ExecutionPolicy(timeout=30))
    assert agent.checkpointer is not None and agent.policy.timeout == 30
    out = agent.invoke({"n": 0}, thread_id="t1")
    assert agent.get_state(thread_id="t1").values["n"] == 3


def test_model_injection():
    class QA(KlafiGraph):
        spec = AgentSpec(id="qa", name="QA")
        state_schema = dict  # 자유형

        def define(self):
            @klafi_node("a")
            def a(s):
                return {"answer": self.model(s["q"])}

            self.add_node("a", a)
            self.add_edge(START, "a")
            self.add_edge("a", END)

    agent = QA(model=lambda p: f"r:{p}")
    assert agent.invoke({"q": "hi"})["answer"] == "r:hi"


def test_missing_state_schema_raises():
    class Bad(KlafiGraph):
        spec = AgentSpec(id="b", name="B")

        def define(self):
            pass

    with pytest.raises(Exception, match="state_schema"):
        Bad()


def test_observability_opt_out(exporter):
    class Quiet(KlafiGraph):
        spec = AgentSpec(id="q", name="Q")
        state_schema = State
        observability = False

        def define(self):
            @klafi_node("step")
            def step(s):
                return {"n": s["n"] + 1}

            self.add_node("step", step)
            self.add_edge(START, "step")
            self.add_edge("step", END)

    Quiet().invoke({"n": 0})
    assert "agent.q" not in {s.name for s in exporter.get_finished_spans()}
