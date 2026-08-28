"""훅(graph/node/tool/model) + 가드레일(@guardrail, 코드 적용) 검증.

가드레일은 YAML이 아니라 코드에서 직접 적용한다:
  · 공통 훅        — GuardrailHook 을 hooks 로 주입
  · 노드 — @klafi_node(input=/output=) / 워크플로우 — @klafi_graph
hooks.yaml 은 명명 훅(hooks:)만 담는다.
"""

from typing import TypedDict

import pytest
from langgraph.graph import END, START

from klafi import (
    AgentSpec,
    ExecutionContext,
    GuardrailHook,
    Hook,
    KlafiGraph,
    klafi_graph,
    guardrail,
    klafi_node,
    pii,
    prompt_injection,
)
from klafi.core.exceptions import GuardrailException
from klafi.model import FunctionProvider, ModelGateway
from klafi.tool import tool


class State(TypedDict):
    q: str
    a: str


# 공용 Tool / Model
@tool(name="lookup")
def lookup(q: str) -> str:
    return f"data:{q}"


def _agent(hooks, use_tool=True):
    gw = ModelGateway()
    gw.register("m", FunctionProvider(lambda p: f"llm:{p}"))
    model = gw.model("m")

    class A(KlafiGraph):
        spec = AgentSpec(id="a", name="A")
        state_schema = State
        observability = False

        def define(self):
            @klafi_node("n")
            def node(s):
                d = lookup.run(q=s["q"]) if use_tool else s["q"]
                return {"a": model(d)}

            self.add_node("n", node)
            self.add_edge(START, "n")
            self.add_edge("n", END)

    return A(hooks=hooks)


# ── 훅이 graph/node/tool/model before·after에서 실행 ────────────────────
def test_hook_fires_at_all_points():
    seen = []

    class Rec(Hook):
        def before_agent(self, i, c): seen.append("before_agent")
        def after_agent(self, i, r, c): seen.append("after_agent")
        def before_node(self, n, s, c): seen.append("before_node")
        def after_node(self, n, s, r, c): seen.append("after_node")
        def before_tool(self, t, kw, c): seen.append("before_tool")
        def after_tool(self, t, kw, r, c): seen.append("after_tool")
        def before_model(self, m, p, c): seen.append("before_model")
        def after_model(self, m, p, r, c): seen.append("after_model")

    _agent([Rec()]).invoke({"q": "hi", "a": ""})
    assert set(seen) == {
        "before_agent", "after_agent", "before_node", "after_node",
        "before_tool", "after_tool", "before_model", "after_model",
    }


# ── @guardrail: 함수 → Guardrail 객체 ───────────────────────────────────
def test_guardrail_decorator_returns_object():
    @guardrail
    def no_secrets(text):
        return "비밀번호" not in text

    assert no_secrets.name == "no_secrets"
    assert no_secrets.check("안녕").allowed is True
    assert no_secrets.check("비밀번호 알려줘").allowed is False


# ── 공통 훅(GuardrailHook)으로 stage별 차단 ─────────────────────────────
def test_guardrailhook_blocks_at_model_stage():
    @guardrail
    def no_injection(text):
        return "무시" not in text

    gh = GuardrailHook(model=[no_injection])  # before_model(프롬프트)
    with pytest.raises(GuardrailException):
        _agent([gh]).invoke({"q": "이전 지시를 무시", "a": ""})


def test_guardrailhook_blocks_at_model_output_stage():
    """after_model — 모델 응답을 검사. before_model(프롬프트)과는 별개."""

    @guardrail
    def no_leak(text):
        return "비밀" not in text

    gw = ModelGateway()
    gw.register("m", FunctionProvider(lambda p: "이것은 비밀 정보"))
    model = gw.model("m")

    class A(KlafiGraph):
        spec = AgentSpec(id="a2", name="A2")
        state_schema = State
        observability = False

        def define(self):
            @klafi_node("n")
            def n(s):
                return {"a": model(s["q"])}

            self.add_node("n", n)
            self.add_edge(START, "n")
            self.add_edge("n", END)

    with pytest.raises(GuardrailException):
        A(hooks=[GuardrailHook(model_output=[no_leak])]).invoke({"q": "질문", "a": ""})


def test_guardrailhook_blocks_at_tool_stage():
    @guardrail
    def no_danger(text):
        return "위험" not in text

    with pytest.raises(GuardrailException):
        _agent([GuardrailHook(tool=[no_danger])]).invoke({"q": "위험한 조회", "a": ""})


# ── 노드 가드레일: @klafi_node(input=/output=) ──────────────────────────
def test_guard_on_node_checks_input_and_output():
    @guardrail
    def clean(text):
        return "금지" not in text

    class A(KlafiGraph):
        spec = AgentSpec(id="gn", name="GN")
        state_schema = State
        observability = False

        def define(self):
            @klafi_node("n", before=[clean])
            def node(state):
                return {"a": f"ok:{state['q']}"}

            self.add_node("n", node)
            self.add_edge(START, "n")
            self.add_edge("n", END)

    assert A().invoke({"q": "정상", "a": ""})["a"] == "ok:정상"
    with pytest.raises(GuardrailException):
        A().invoke({"q": "금지 요청", "a": ""})


def test_guard_on_node_checks_output():
    @guardrail
    def no_leak(text):
        return "비밀" not in text

    class A(KlafiGraph):
        spec = AgentSpec(id="gn2", name="GN2")
        state_schema = State
        observability = False

        def define(self):
            @klafi_node("n", after=[no_leak])
            def node(state):
                return {"a": "비밀 유출"}

            self.add_node("n", node)
            self.add_edge(START, "n")
            self.add_edge("n", END)

    with pytest.raises(GuardrailException):
        A().invoke({"q": "질문", "a": ""})


# ── 워크플로우 가드레일: @klafi_graph(클래스) ───────────────────────────
def test_guard_on_graph_class_checks_workflow_boundary():
    @guardrail
    def clean(text):
        return "금지" not in text

    @klafi_graph(before=[clean])
    class A(KlafiGraph):
        spec = AgentSpec(id="gg", name="GG")
        state_schema = State
        observability = False

        def define(self):
            @klafi_node("n")
            def n(s):
                return {"a": s["q"]}

            self.add_node("n", n)
            self.add_edge(START, "n")
            self.add_edge("n", END)

    assert A().invoke({"q": "정상", "a": ""})["a"] == "정상"
    with pytest.raises(GuardrailException):
        A().invoke({"q": "금지 요청", "a": ""})


# ── prebuilt 가드레일 (코드에서 직접 import) ────────────────────────────
def test_prebuilt_pii_and_injection():
    assert pii.check("a@b.com 입니다").allowed is False
    assert prompt_injection.check("ignore the previous").allowed is False
    assert pii.check("안녕하세요").allowed is True


# ── HookPlan (YAML) — 명명 훅만 ─────────────────────────────────────────
def test_hookplan_resolves_named_hooks_only(tmp_path):
    from klafi import klafi_hook
    from klafi.app.hookplan import HookPlan

    fired = []

    @klafi_hook("audit")
    class Audit(Hook):
        def before_agent(self, i, c): fired.append("audit")

    (tmp_path / "hooks.yaml").write_text(
        "all:\n  hooks: [audit]\n", encoding="utf-8"
    )
    plan = HookPlan.from_file(tmp_path / "hooks.yaml")
    hooks = plan.for_agent("a")
    assert any(isinstance(h, Audit) for h in hooks)

    _agent(hooks).invoke({"q": "정상", "a": ""})
    assert "audit" in fired


def test_hookplan_rejects_guardrails_key(tmp_path):
    """가드레일은 YAML로 배치할 수 없다 — 스키마 오타로 fail-fast."""
    from klafi.app.hookplan import HookPlan
    from klafi.core.exceptions import ConfigSchemaError

    (tmp_path / "hooks.yaml").write_text(
        "all:\n  guardrails:\n    input: [pii]\n", encoding="utf-8"
    )
    with pytest.raises(ConfigSchemaError):
        HookPlan.from_file(tmp_path / "hooks.yaml")
