"""예외 체계 검증 — 도메인 축 / 종류 축 두 방향으로 잡힌다 (요구사항 §23)."""

from typing import TypedDict

import pytest
from langgraph.graph import END, START
from pydantic import BaseModel

from klafi import (
    AgentSpec,
    ConfigException,
    ConfigNotFoundError,
    ConfigSchemaError,
    ExecutionContext,
    GuardrailException,
    GuardrailHook,
    GuardrailViolationError,
    KlafiApp,
    KlafiException,
    KlafiGraph,
    ModelException,
    ModelNotFoundError,
    ModelGateway,
    NotFoundError,
    PermissionDeniedError,
    ToolException,
    ToolNotFoundError,
    ToolPermissionError,
    ToolValidationError,
    ToolRegistry,
    ValidationError,
    ViolationError,
    guardrail,
    klafi_node,
)
from klafi.core.context import bind_context
from klafi.hookdefs import resolve_named_hooks
from klafi.core.exceptions import HookNotFoundError
from klafi.tool import tool


# ── config 미설정 vs 스키마 오류 ────────────────────────────────────────
def test_config_not_found_vs_schema_error(tmp_path):
    with pytest.raises(ConfigNotFoundError) as e1:
        KlafiApp.from_config("/tmp/klafi-nope-xyz")
    assert e1.value.error_code == "CONFIG_NOT_FOUND"
    assert isinstance(e1.value, (ConfigException, NotFoundError))  # 두 축 모두

    (tmp_path / "framework.yaml").write_text("service: t\n", encoding="utf-8")
    (tmp_path / "policy.yaml").write_text("timeoutt: 1\n", encoding="utf-8")
    with pytest.raises(ConfigSchemaError) as e2:
        KlafiApp.from_config(str(tmp_path))
    assert e2.value.error_code == "CONFIG_SCHEMA_ERROR"
    assert isinstance(e2.value, (ConfigException, ValidationError))


# ── 못찾음 (NotFoundError 로 한 번에) ───────────────────────────────────
def test_not_found_family():
    with pytest.raises(ToolNotFoundError) as t:
        ToolRegistry().get("nope")
    with pytest.raises(ModelNotFoundError) as m:
        ModelGateway().model("nope")("x")
    with pytest.raises(HookNotFoundError) as h:
        resolve_named_hooks(["nope"])

    # 종류 축: 전부 NotFoundError
    for exc in (t.value, m.value, h.value):
        assert isinstance(exc, NotFoundError)
    # 도메인 축: 각자 원래 타입으로도 잡힘 (하위호환)
    assert isinstance(t.value, ToolException)
    assert isinstance(m.value, ModelException)
    assert {t.value.error_code, m.value.error_code} == {"TOOL_NOT_FOUND", "MODEL_NOT_FOUND"}


# ── 실행 중 가드레일 위반 ───────────────────────────────────────────────
class S(TypedDict):
    q: str
    a: str


def _agent(hooks):
    class A(KlafiGraph):
        spec = AgentSpec(id="a", name="A")
        state_schema = S
        observability = False

        def define(self):
            @klafi_node("n")
            def n(s):
                return {"a": s["q"]}

            self.add_node("n", n)
            self.add_edge(START, "n")
            self.add_edge("n", END)

    return A(hooks=hooks)


def test_guardrail_violation_at_runtime():
    @guardrail("blk_test")
    def blk_test(text):
        return "금지" not in text

    with pytest.raises(GuardrailViolationError) as e:
        _agent([GuardrailHook(input=[blk_test])]).invoke({"q": "금지어", "a": ""})

    assert e.value.error_code == "GUARDRAIL_VIOLATION"
    assert e.value.context["stage"] == "input"  # 어느 단계에서 막혔는지
    assert e.value.context["guard"] == "blk_test"
    # 두 축: 가드레일 도메인 + 실행 중 위반
    assert isinstance(e.value, (GuardrailException, ViolationError))
    # 미등록(설정 오류)과는 다른 타입
    assert not isinstance(e.value, NotFoundError)


# ── Tool 권한 vs 검증 ───────────────────────────────────────────────────
def test_tool_permission_vs_validation():
    class In(BaseModel):
        n: int

    @tool(name="secure", required_permission="db:write", input_schema=In)
    def secure(n: int) -> int:
        return n

    with pytest.raises(ToolPermissionError) as p:  # 권한 없음
        secure.run(n=1)
    assert isinstance(p.value, (ToolException, PermissionDeniedError))
    assert p.value.error_code == "TOOL_PERMISSION_DENIED"

    ctx = ExecutionContext.new(security_context={"permissions": ["db:write"]})
    with bind_context(ctx):
        with pytest.raises(ToolValidationError) as v:  # 권한은 있으나 입력 불량
            secure.run(n="숫자아님")
    assert isinstance(v.value, (ToolException, ValidationError))
    assert v.value.error_code == "TOOL_VALIDATION_ERROR"


# ── 모든 KLAFI 예외는 KlafiException 하위 ───────────────────────────────
def test_all_are_klafi_exceptions():
    for exc_cls in (
        ConfigNotFoundError, ConfigSchemaError, ToolNotFoundError, ToolPermissionError,
        ToolValidationError, ModelNotFoundError,
        GuardrailViolationError, HookNotFoundError, NotFoundError, ViolationError,
    ):
        assert issubclass(exc_cls, KlafiException)
