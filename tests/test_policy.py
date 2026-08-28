"""WS2 Execution Policy 검증 (요구사항 §12, F07 / EXE-07·08).

DoD: Agent 코드 변경 없이 Config로 실행정책 변경.
+ Timeout, Retry+Backoff, Max Retry, 결정적 예외 재시도 제외, 상태 전이.
"""

import time

import pytest
from langgraph.graph import END, START, StateGraph
from typing import TypedDict

from klafi import AgentSpec, BaseGraph, ExecutionContext, ExecutionPolicy, ExecutionState, get_context
from klafi.core.exceptions import GuardrailException, TimeoutException


class State(TypedDict):
    text: str


def _agent(build_node, policy=None, config=None):
    class A(BaseGraph):
        def build(self):
            g = StateGraph(State)
            g.add_node("n", build_node)
            g.add_edge(START, "n")
            g.add_edge("n", END)
            return g

    return A(AgentSpec(id="a", name="A", config=config or {}), policy=policy)


# ── ExecutionPolicy 단위 ────────────────────────────────────────────────
def test_backoff_is_exponential_and_capped():
    p = ExecutionPolicy(backoff_base=1.0, backoff_factor=2.0, backoff_max=5.0)
    assert [p.backoff_delay(i) for i in range(4)] == [1.0, 2.0, 4.0, 5.0]  # 8→cap 5


def test_should_retry_respects_max_and_deterministic():
    p = ExecutionPolicy(max_retries=2)
    assert p.should_retry(ValueError(), 0) is True
    assert p.should_retry(ValueError(), 2) is False  # attempt >= max
    assert p.should_retry(GuardrailException("x"), 0) is False  # no_retry_on


# ── Timeout ─────────────────────────────────────────────────────────────
def test_sync_timeout_raises_and_sets_state():
    def slow(s):
        time.sleep(0.3)
        return {"text": "done"}

    ctx = ExecutionContext.new()
    with pytest.raises(TimeoutException):
        _agent(slow, policy=ExecutionPolicy(timeout=0.05)).invoke({"text": "x"}, context=ctx)
    assert ctx.state == ExecutionState.TIMEOUT.value


def test_timeout_thread_still_sees_context():
    # ContextVar가 timeout worker thread로 전파되는지 (전파 실패 시 get_context()=None)
    seen = {}

    def node(s):
        seen["ctx"] = get_context()
        return {"text": "ok"}

    _agent(node, policy=ExecutionPolicy(timeout=1.0)).invoke({"text": "x"})
    assert seen["ctx"] is not None and len(seen["ctx"].execution_id) == 32


async def test_async_timeout_cancels():
    import asyncio

    async def slow(s):
        await asyncio.sleep(0.3)
        return {"text": "done"}

    ctx = ExecutionContext.new()
    with pytest.raises(TimeoutException):
        await _agent(slow, policy=ExecutionPolicy(timeout=0.05)).ainvoke({"text": "x"}, context=ctx)
    assert ctx.state == ExecutionState.TIMEOUT.value


# ── Retry ───────────────────────────────────────────────────────────────
def test_retry_then_success():
    calls = []

    def flaky(s):
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("transient")
        return {"text": "ok"}

    ctx = ExecutionContext.new()
    out = _agent(flaky, policy=ExecutionPolicy(max_retries=5, backoff_base=0.0)).invoke(
        {"text": "x"}, context=ctx
    )
    assert out["text"] == "ok"
    assert len(calls) == 3  # 2 실패 + 1 성공
    assert ctx.state == ExecutionState.COMPLETED.value


def test_retry_exhausted_fails():
    calls = []

    def always(s):
        calls.append(1)
        raise ValueError("nope")

    ctx = ExecutionContext.new()
    with pytest.raises(ValueError):
        _agent(always, policy=ExecutionPolicy(max_retries=2, backoff_base=0.0)).invoke(
            {"text": "x"}, context=ctx
        )
    assert len(calls) == 3  # 최초 1 + 재시도 2
    assert ctx.state == ExecutionState.FAILED.value


def test_guardrail_not_retried():
    calls = []

    def blocked(s):
        calls.append(1)
        raise GuardrailException("blocked")

    with pytest.raises(GuardrailException):
        _agent(blocked, policy=ExecutionPolicy(max_retries=5, backoff_base=0.0)).invoke({"text": "x"})
    assert len(calls) == 1  # 결정적 실패 → 재시도 없음


# ── Config 주입 (DoD) ───────────────────────────────────────────────────
def test_policy_from_config_no_code_change():
    def slow(s):
        time.sleep(0.3)
        return {"text": "done"}

    # 코드가 아니라 Config(dict)로 정책 지정
    agent = _agent(slow, config={"policy": {"timeout": 0.05, "max_retries": 0}})
    with pytest.raises(TimeoutException):
        agent.invoke({"text": "x"})


def test_no_policy_zero_overhead():
    # 정책 미지정 시 runtime 없이도 동작(직접 호출 경로)
    out = _agent(lambda s: {"text": "ok"}).invoke({"text": "x"})
    assert out["text"] == "ok"


# ── Retry × Checkpoint 상호작용 (부수효과 중복 방지) ────────────────────
def test_retry_with_checkpointer_resumes_instead_of_replaying():
    """체크포인터가 있으면 정책 Retry가 완료된 Node를 재실행하지 않는다.

    원본 input을 재차 넘기면 그래프가 처음부터 돌아 결제 등 부수효과가 중복된다.
    """
    import operator
    from typing import Annotated

    class S(TypedDict):
        steps: Annotated[list, operator.add]

    calls = {"a": 0, "b": 0}
    fail = {"on": True}

    class Flow(BaseGraph):
        def build(self):
            g = StateGraph(S)

            def node_a(s):
                calls["a"] += 1
                return {"steps": ["a"]}

            def node_b(s):
                calls["b"] += 1
                if fail["on"]:
                    raise ConnectionError("외부 시스템 장애")
                return {"steps": ["b"]}

            g.add_node("a", node_a)
            g.add_node("b", node_b)
            g.add_edge(START, "a")
            g.add_edge("a", "b")
            g.add_edge("b", END)
            return g

    agent = Flow(
        AgentSpec(id="flow", name="Flow"),
        checkpointer="memory",
        policy=ExecutionPolicy(max_retries=2, backoff_base=0.0),
    )
    with pytest.raises(ConnectionError):
        agent.invoke({"steps": []}, thread_id="t1")

    assert calls["a"] == 1  # 완료된 node는 재시도에도 1회만
    assert calls["b"] == 3  # 실패 node만 재시도 (최초 1 + retry 2)
    assert agent.get_state(thread_id="t1").values["steps"] == ["a"]

    # 장애 해소 후 Resume → b부터 이어서
    fail["on"] = False
    assert agent.invoke(None, thread_id="t1")["steps"] == ["a", "b"]
    assert calls["a"] == 1  # 여전히 1회


def test_retry_without_checkpointer_replays_input():
    """체크포인터가 없으면 재개할 상태가 없으므로 원본 input으로 재시도한다."""
    calls = []

    class S(TypedDict):
        x: str

    class Flow(BaseGraph):
        def build(self):
            g = StateGraph(S)

            def node(s):
                calls.append(s["x"])
                raise ValueError("실패")

            g.add_node("n", node)
            g.add_edge(START, "n")
            g.add_edge("n", END)
            return g

    agent = Flow(AgentSpec(id="f2", name="F2"), policy=ExecutionPolicy(max_retries=1, backoff_base=0.0))
    with pytest.raises(ValueError):
        agent.invoke({"x": "입력"})
    assert calls == ["입력", "입력"]  # 매번 원본 input으로 재실행
