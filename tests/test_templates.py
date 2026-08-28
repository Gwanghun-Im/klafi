"""WS8 Template 검증 (요구사항 §20, F14).

DoD: 신규 개발자가 Template을 복사해 첫 Agent를 만들 수 있을 것.
→ model/retriever만 주입하면 Logging+Tracing+Checkpoint가 자동으로 붙어 실행된다.
"""

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from klafi import RAGAgent, SimpleAgent, setup_tracing


@pytest.fixture(scope="session")
def exporter():
    exp = InMemorySpanExporter()
    setup_tracing(exporter=exp, simple=True)
    return exp


@pytest.fixture(autouse=True)
def _clear(exporter):
    exporter.clear()
    yield


def _names(exporter):
    return {s.name for s in exporter.get_finished_spans()}


def test_simple_agent_runs_with_only_a_model(exporter):
    agent = SimpleAgent(model=lambda p: f"echo:{p}")
    out = agent.invoke({"question": "hi"})
    assert out["answer"] == "echo:hi"
    # 개발자 코드 0줄로 Tracing 자동 부착
    assert {"agent.simple", "node.llm", "model.llm"} <= _names(exporter)


def test_rag_agent_runs_with_model_and_retriever(exporter):
    agent = RAGAgent(
        model=lambda p: "answer",
        retriever=lambda q: ["doc1", "doc2"],
    )
    out = agent.invoke({"question": "q"})
    assert out["context"] == ["doc1", "doc2"]
    assert out["answer"] == "answer"
    # 흐름: retriever → generate, 각각 tool/model span 중첩
    assert {"agent.rag", "node.retrieve", "tool.retriever", "node.generate", "model.llm"} <= _names(exporter)


def test_template_checkpointer_string_resolves():
    # DoD: checkpointer="memory" 문자열만으로 Resume 배선
    agent = SimpleAgent(model=lambda p: "x", checkpointer="memory")
    assert agent.checkpointer is not None
    agent.invoke({"question": "hi"}, thread_id="t1")
    snap = agent.get_state(thread_id="t1")
    assert snap.values["answer"] == "x"


def test_template_policy_passthrough():
    from klafi import ExecutionPolicy
    from klafi.core.exceptions import TimeoutException
    import time

    agent = SimpleAgent(
        model=lambda p: (time.sleep(0.2), "late")[1],
        policy=ExecutionPolicy(timeout=0.05),
    )
    with pytest.raises(TimeoutException):
        agent.invoke({"question": "hi"})
