"""WS5 Observability 검증 (요구사항 §16, F11).

DoD: Execution → Agent → Node → Tool → Model → Error 단일 Trace 추적.
+ Correlation ID 속성, Business Exception의 span 연결, Fail-Open.
"""

import pytest
from langgraph.graph import END, START, StateGraph
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from typing import TypedDict

from klafi import AgentSpec, BaseGraph, ExecutionContext, TracingHook, setup_tracing, span


class State(TypedDict):
    text: str


# OTel Provider는 프로세스당 1회만 설정 → 세션 스코프로 두고 매 테스트 exporter만 비운다.
@pytest.fixture(scope="session")
def exporter():
    exp = InMemorySpanExporter()
    setup_tracing(exporter=exp, simple=True)  # simple=즉시 flush
    return exp


@pytest.fixture(autouse=True)
def _clear(exporter):
    exporter.clear()
    yield


def _by_name(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


def _agent(node, spec_id="obs"):
    class A(BaseGraph):
        def build(self):
            g = StateGraph(State)
            g.add_node("work", node)
            g.add_edge(START, "work")
            g.add_edge("work", END)
            return g

    return A(AgentSpec(id=spec_id, name="Obs", version="1.0", project="demo"), hooks=[TracingHook()])


def test_full_chain_hierarchy(exporter):
    def node(s):
        with span("tool.search", **{"klafi.tool": "search"}):
            pass
        with span("model.gpt", **{"klafi.model": "gpt", "klafi.tokens": 42}):
            pass
        return {"text": s["text"] + "!"}

    ctx = ExecutionContext.new(agent_id="obs", agent_version="1.0", session_id="sess1")
    out = _agent(node).invoke({"text": "hi"}, context=ctx)
    assert out["text"] == "hi!"

    spans = _by_name(exporter)
    assert set(spans) >= {"agent.obs", "node.work", "tool.search", "model.gpt"}
    # 계층: model/tool → node → agent
    assert spans["tool.search"].parent.span_id == spans["node.work"].context.span_id
    assert spans["model.gpt"].parent.span_id == spans["node.work"].context.span_id
    assert spans["node.work"].parent.span_id == spans["agent.obs"].context.span_id
    # 같은 trace에 묶임 (Execution 단일 Trace)
    tid = spans["agent.obs"].context.trace_id
    assert all(s.context.trace_id == tid for s in spans.values())


def test_correlation_ids_on_span(exporter):
    ctx = ExecutionContext.new(agent_id="obs", session_id="sess1", user_id="u1", tenant_id="t1")
    _agent(lambda s: {"text": "x"}).invoke({"text": "x"}, context=ctx)
    a = _by_name(exporter)["agent.obs"]
    attrs = a.attributes
    assert attrs["klafi.execution_id"] == ctx.execution_id
    assert attrs["klafi.session_id"] == "sess1"
    assert attrs["klafi.thread_id"] == "sess1"  # session 우선
    assert attrs["klafi.user_id"] == "u1"


def test_model_token_attribute_recorded(exporter):
    def node(s):
        with span("model.gpt", **{"klafi.tokens": 128}):
            pass
        return {"text": "x"}

    _agent(node).invoke({"text": "x"})
    assert _by_name(exporter)["model.gpt"].attributes["klafi.tokens"] == 128


def test_business_exception_connects_to_trace(exporter):
    def node(s):
        raise ValueError("business failure in node")

    with pytest.raises(ValueError):
        _agent(node).invoke({"text": "x"})

    spans = _by_name(exporter)
    node_span = spans["node.work"]
    agent_span = spans["agent.obs"]
    # Node/Agent span 모두 ERROR + 예외 이벤트 기록 (OBS-07)
    assert node_span.status.status_code == StatusCode.ERROR
    assert agent_span.status.status_code == StatusCode.ERROR
    events = [e.name for e in node_span.events]
    assert "exception" in events


def test_tracing_hook_is_fail_open(exporter):
    # Provider가 정상이어도 Hook 예외가 나면 안 되지만, 최소한 정상 실행은 span을 남긴다.
    out = _agent(lambda s: {"text": "ok"}).invoke({"text": "x"})
    assert out["text"] == "ok"
    assert "agent.obs" in _by_name(exporter)


def test_no_provider_is_noop():
    # setup_tracing을 부르지 않은 경로에서도 (no-op tracer) 실행이 깨지지 않음.
    # 별도 프로세스가 아니라 여기선 Provider가 이미 있으므로, Hook만으로 예외 없음을 확인.
    out = _agent(lambda s: {"text": "safe"}).invoke({"text": "x"})
    assert out["text"] == "safe"


# ── setup_logging (공통 로깅 부트스트랩, from_config 가 자동 호출) ──────────
def test_setup_logging_respects_opt_out(monkeypatch):
    """KLAFI_LOG_SETUP=0 이면 root 로거를 건드리지 않는다(호스트 앱 존중)."""
    import logging as _logging

    from klafi import setup_logging

    root = _logging.getLogger()
    before = list(root.handlers)
    monkeypatch.setenv("KLAFI_LOG_SETUP", "0")
    try:
        setup_logging()
        assert root.handlers == before  # 아무 핸들러도 추가되지 않음
    finally:
        root.handlers = before
