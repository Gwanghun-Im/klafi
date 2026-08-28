"""@klafi_node — 노드 전용 미들웨어(before/after/on_error) + 가드레일(input/output) 검증."""

from typing import TypedDict

import pytest
from langgraph.graph import END, START

from klafi import AgentSpec, ExecutionContext, KlafiGraph, guardrail, klafi_node
from klafi.core.exceptions import GuardrailViolationError


class State(TypedDict):
    q: str
    a: str
    verified: bool


def _agent(build, spec_id="nd"):
    class A(KlafiGraph):
        spec = AgentSpec(id=spec_id, name=spec_id)
        state_schema = State
        observability = False

        def define(self):
            build(self)

    return A


# ── before/after 는 임의 미들웨어 (가드레일 아님) ──────────────────────
def test_before_middleware_can_modify_state():
    """before 가 반환하면 body 로 넘어가는 state 를 교체한다(세션 보강 등)."""
    seen = {}

    def enrich(state):
        return {**state, "verified": True}  # state 교체

    def build(g):
        @klafi_node("work", before=[enrich])
        def work(state):
            seen["verified"] = state["verified"]  # body 가 보강된 state 를 본다
            return {"a": "ok"}

        g.add_node("work", work)
        g.add_edge(START, "work")
        g.add_edge("work", END)

    _agent(build)().invoke({"q": "x", "a": "", "verified": False})
    assert seen["verified"] is True


def test_before_middleware_uses_ctx_and_can_raise():
    """세션/로그인 검증 예: ctx 를 받아 검증하고 실패 시 예외(fail-close)."""

    def require_login(state, ctx):
        if not (ctx and ctx.security_context.get("user_id")):
            raise PermissionError("로그인 필요")
        return None  # 통과, state 변경 없음

    def build(g):
        @klafi_node("secure", before=[require_login])
        def secure(state):
            return {"a": "secret"}

        g.add_node("secure", secure)
        g.add_edge(START, "secure")
        g.add_edge("secure", END)

    A = _agent(build, "nd_login")
    # 로그인 있음 → 통과
    ok = ExecutionContext.new(security_context={"user_id": "u1"})
    assert A().invoke({"q": "x", "a": "", "verified": False}, context=ok)["a"] == "secret"
    # 로그인 없음 → 차단
    with pytest.raises(PermissionError):
        A().invoke({"q": "x", "a": "", "verified": False}, context=ExecutionContext.new())


def test_after_middleware_can_modify_result():
    def redact(result):
        return {**result, "a": "***"}

    def build(g):
        @klafi_node("work", after=[redact])
        def work(state):
            return {"a": "민감정보"}

        g.add_node("work", work)
        g.add_edge(START, "work")
        g.add_edge("work", END)

    out = _agent(build, "nd_after")().invoke({"q": "x", "a": "", "verified": False})
    assert out["a"] == "***"  # after 가 노드 출력을 교체(persist)


def test_on_error_middleware_observes_exception():
    seen = []

    def build(g):
        @klafi_node("boom", on_error=[lambda exc: seen.append(type(exc).__name__)])
        def boom(state):
            raise ValueError("터짐")

        g.add_node("boom", boom)
        g.add_edge(START, "boom")
        g.add_edge("boom", END)

    with pytest.raises(ValueError):
        _agent(build, "nd_err")().invoke({"q": "x", "a": "", "verified": False})
    assert seen == ["ValueError"]


# ── 노드 가드레일 (input/output) 통합 ───────────────────────────────────
def test_node_input_guardrail_blocks():
    @guardrail
    def clean(text):
        return "금지" not in text

    def build(g):
        @klafi_node("work", before=[clean])
        def work(state):
            return {"a": state["q"]}

        g.add_node("work", work)
        g.add_edge(START, "work")
        g.add_edge("work", END)

    A = _agent(build, "nd_gin")
    assert A().invoke({"q": "정상", "a": "", "verified": False})["a"] == "정상"
    with pytest.raises(GuardrailViolationError):
        A().invoke({"q": "금지 요청", "a": "", "verified": False})


def test_node_output_guardrail_blocks():
    @guardrail
    def no_leak(text):
        return "비밀" not in text

    def build(g):
        @klafi_node("work", after=[no_leak])
        def work(state):
            return {"a": "비밀 유출"}

        g.add_node("work", work)
        g.add_edge(START, "work")
        g.add_edge("work", END)

    with pytest.raises(GuardrailViolationError):
        _agent(build, "nd_gout")().invoke({"q": "x", "a": "", "verified": False})


def _collapse(seq):
    """연속 중복 제거 — 가드레일은 문자열 리프마다 실행되므로 리프 수만큼 반복될 수 있다.
    순서 검증에는 스테이지 전이만 의미가 있다."""
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


# ── 순서: before 리스트 순서 → fn → after 리스트 순서 ────────────────────
def test_pipeline_order():
    """가드레일과 미들웨어는 한 리스트에 섞이며 **리스트에 적은 순서대로** 적용된다."""
    order = []

    @guardrail
    def gin(text):
        order.append("input_guard")
        return True

    @guardrail
    def gout(text):
        order.append("output_guard")
        return True

    def build(g):
        @klafi_node(
            "work",
            before=[gin, lambda s: order.append("before") or None],
            after=[lambda r: order.append("after") or None, gout],
        )
        def work(state):
            order.append("fn")
            return {"a": "ok"}

        g.add_node("work", work)
        g.add_edge(START, "work")
        g.add_edge("work", END)

    _agent(build, "nd_order")().invoke({"q": "x", "a": "", "verified": False})
    assert _collapse(order) == ["input_guard", "before", "fn", "after", "output_guard"]


def test_pipeline_order_is_explicit():
    """분리돼 있던 시절과 달리, 미들웨어를 가드레일보다 **앞에** 둘 수도 있다(정규화 후 검사)."""
    order = []

    @guardrail
    def g(text):
        order.append("guard")
        return True

    def build(gr):
        @klafi_node("work", before=[lambda s: order.append("normalize") or None, g])
        def work(state):
            return {"a": "ok"}

        gr.add_node("work", work)
        gr.add_edge(START, "work")
        gr.add_edge("work", END)

    _agent(build, "nd_order2")().invoke({"q": "x", "a": "", "verified": False})
    assert _collapse(order) == ["normalize", "guard"]
