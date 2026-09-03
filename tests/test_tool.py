"""Tool Framework 검증 (요구사항 §14.2, F09 / TOL-01~10)."""

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from klafi import ExecutionPolicy, Tool, ToolRegistry, setup_tracing, tool
from klafi.core.context import ExecutionContext, bind_context
from klafi.core.exceptions import ToolException


@pytest.fixture(scope="session")
def exporter():
    exp = InMemorySpanExporter()
    setup_tracing(exporter=exp, simple=True)
    return exp


@pytest.fixture(autouse=True)
def _clear(exporter):
    exporter.clear()
    yield


def _spans(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


# ── Metadata + 기본 실행 + span (TOL-01/02/07/10) ───────────────────────
def test_tool_metadata_and_span(exporter):
    @tool(name="search", description="검색", tags=["io"])
    def search(query: str) -> str:
        return f"결과:{query}"

    assert search.metadata.name == "search"
    assert search.metadata.description == "검색" and search.metadata.tags == ["io"]
    assert search(query="klafi") == "결과:klafi"
    sp = _spans(exporter)["tool.search"]
    assert sp.attributes["klafi.tool"] == "search" and sp.attributes["klafi.tool_ok"] is True


# ── Input Validation (TOL-08) ───────────────────────────────────────────
def test_input_validation_coerces_and_rejects():
    class In(BaseModel):
        n: int

    @tool(input_schema=In)
    def double(n: int) -> int:
        return n * 2

    assert double(n="21") == 42  # "21" → 21 강제 변환
    with pytest.raises(ToolException, match="입력 검증"):
        double(n="not-a-number")


# ── Output Validation (TOL-09) ──────────────────────────────────────────
def test_output_validation():
    class Out(BaseModel):
        answer: str

    @tool(output_schema=Out)
    def good() -> dict:
        return {"answer": "ok"}

    assert good().answer == "ok"

    @tool(output_schema=Out)
    def bad() -> dict:
        return {"wrong": 1}

    with pytest.raises(ToolException, match="출력 검증"):
        bad()


# ── 권한 (TOL-06, 최소권한) ─────────────────────────────────────────────
def test_permission_denied_without_grant():
    @tool(required_permission="db:write")
    def writer() -> str:
        return "wrote"

    with pytest.raises(ToolException, match="권한 없음"):
        writer()  # context 없음 → 거부


def test_permission_granted_via_security_context():
    @tool(required_permission="db:write")
    def writer() -> str:
        return "wrote"

    ctx = ExecutionContext.new(security_context={"permissions": ["db:write"]})
    with bind_context(ctx):
        assert writer() == "wrote"


# ── Timeout/Retry (TOL-04/05) ───────────────────────────────────────────
def test_tool_retry():
    calls = []

    @tool(policy=ExecutionPolicy(max_retries=3, backoff_base=0.0))
    def flaky() -> str:
        calls.append(1)
        if len(calls) < 2:
            raise ConnectionError("transient")
        return "ok"

    assert flaky() == "ok" and len(calls) == 2


def test_tool_timeout():
    import time

    from klafi.core.exceptions import TimeoutException

    @tool(policy=ExecutionPolicy(timeout=0.05))
    def slow() -> str:
        time.sleep(0.3)
        return "late"

    with pytest.raises(TimeoutException):
        slow()


# ── Registry (TOL-03 / FAC-05) ──────────────────────────────────────────
def test_registry_register_get_load():
    reg = ToolRegistry()
    reg.register(Tool(lambda: 1, name="a"))
    reg.register(Tool(lambda: 2, name="b"))
    assert reg.get("a").name == "a"
    assert {t.name for t in reg.all()} == {"a", "b"}
    assert [t.name for t in reg.load(["b", "a"])] == ["b", "a"]
    with pytest.raises(ToolException, match="미등록"):
        reg.get("z")
