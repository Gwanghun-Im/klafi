"""v0.1.5 결함 수정 검증 — 테스트 ID 는 결함 대장(LangGraph 대조 검증) 기준.

각 테스트는 "raw StateGraph 에서는 되는데 KlafiGraph 를 거치면 안 되던 것" 또는 재현된 오동작 하나를 고정한다.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TypedDict

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import get_runtime
from langgraph.types import CachePolicy

from klafi import AgentSpec, KlafiGraph
from klafi.core import klafi_node
from klafi.core.context import ExecutionContext
from klafi.core.exceptions import ViolationError
from klafi.core.hook import Hook
from klafi.guardrail import GuardrailHook, GuardrailResult, guardrail


class S(TypedDict):
    x: int


def _spec() -> AgentSpec:
    return AgentSpec(id="t", name="T")


@guardrail
def mask_secret(text: str) -> GuardrailResult:
    return GuardrailResult("SECRET" not in text, "secret", replacement=text.replace("SECRET", "***"))


@guardrail
def block_secret(text: str) -> GuardrailResult:
    return GuardrailResult("SECRET" not in text, "secret 차단")


# ── C1: add_node(cache_policy=) 가 compile(cache=) 까지 닿는다 ─────────────────
def test_c1_cache_policy_is_honoured_with_cache_backend():
    calls = []

    class A(KlafiGraph):
        state_schema = S
        observability = False

        def define(self):
            @klafi_node("n")
            def n(state):
                calls.append(1)
                return {"x": state["x"] + 1}

            self.add_node("n", n, cache_policy=CachePolicy(ttl=60))
            self.add_edge(START, "n")
            self.add_edge("n", END)

    a = A(_spec(), cache="memory")
    assert a.invoke({"x": 0}) == {"x": 1} and a.invoke({"x": 0}) == {"x": 1}
    assert calls == [1]  # 두 번째는 캐시 적중 — 이전엔 cache=None 이라 2회 실행


# ── C2: interrupt_before / durability 가 Pregel 인자로 전달된다 ────────────────
def test_c2_pregel_kwargs_reach_langgraph():
    class A(KlafiGraph):
        state_schema = S
        observability = False

        def define(self):
            @klafi_node("a")
            def a(state):
                return {"x": state["x"] + 1}

            @klafi_node("b")
            def b(state):
                return {"x": state["x"] + 10}

            self.add_node("a", a)
            self.add_node("b", b)
            self.add_edge(START, "a")
            self.add_edge("a", "b")
            self.add_edge("b", END)

    ag = A(_spec(), checkpointer="memory")
    out = ag.invoke({"x": 0}, thread_id="t", interrupt_before=["b"], durability="sync")
    assert out["x"] == 1  # b 앞에서 멈춤 — 이전엔 config 로 접혀 무시되어 11
    assert ag.get_state(thread_id="t").next == ("b",)


# ── C3: context_schema + runtime_context= 로 LangGraph Runtime.context 공급 ─────
def test_c3_runtime_context_reaches_nodes():
    @dataclass
    class Ctx:
        user: str

    class Out(TypedDict):
        who: str

    class A(KlafiGraph):
        state_schema = Out
        context_schema = Ctx
        observability = False

        def define(self):
            @klafi_node("n")
            def n(state):
                return {"who": get_runtime(Ctx).context.user}

            self.add_node("n", n)
            self.add_edge(START, "n")
            self.add_edge("n", END)

    assert A(_spec()).invoke({"who": ""}, runtime_context=Ctx("bob")) == {"who": "bob"}


# ── C4: 동기 stream() 도 stream_mode 를 존중한다 (astream 과 동일) ──────────────
def test_c4_sync_stream_honours_stream_mode():
    class A(KlafiGraph):
        state_schema = S
        observability = False

        def define(self):
            @klafi_node("n")
            def n(state):
                return {"x": state["x"] + 1}

            self.add_node("n", n)
            self.add_edge(START, "n")
            self.add_edge("n", END)

    a = A(_spec())
    assert list(a.stream({"x": 0}, stream_mode="updates")) == [{"n": {"x": 1}}]
    assert list(a.stream({"x": 0}, stream_mode="values"))[-1] == {"x": 1}
    multi = list(a.stream({"x": 0}, stream_mode=["updates", "values"]))
    assert ("updates", {"n": {"x": 1}}) in multi and multi[-1] == ("values", {"x": 1})
    assert list(a.stream({"x": 0}))[-1] == {"n": {"x": 1}}  # 기본(updates, 컴파일 그래프 기본값) 모양 유지


# ── C5: 컴파일된 서브그래프 노드에도 노드 훅이 붙고, 서브그래프 탐지는 유지 ──────
def test_c5_subgraph_node_gets_hooks_and_stays_detectable():
    seen = []

    class Rec(Hook):
        def before_node(self, node, state, ctx):
            seen.append(node)

    sub = StateGraph(S)
    sub.add_node("inner", lambda s: {"x": s["x"] + 100})
    sub.add_edge(START, "inner")
    sub.add_edge("inner", END)
    compiled_sub = sub.compile()

    class A(KlafiGraph):
        state_schema = S
        observability = False

        def define(self):
            @klafi_node("a")
            def a(state):
                return {"x": state["x"] + 1}

            self.add_node("a", a)
            self.add_node("sub", compiled_sub)
            self.add_edge(START, "a")
            self.add_edge("a", "sub")
            self.add_edge("sub", END)

    ag = A(_spec(), hooks=[Rec()])
    assert ag.invoke({"x": 0}) == {"x": 101}
    assert seen == ["a", "sub"]  # 이전엔 'sub' 누락
    assert "sub" in dict(ag.compiled.get_subgraphs())  # find_subgraph_pregel 유지


# ── G4: 스트림 토큰은 after 가드레일 적용 후 본문만, 출력 가드레일 차단은 스트림에도 ─
def _masked_llm_agent(hooks=None):
    class A(KlafiGraph):
        state_schema = MessagesState
        observability = False

        def define(self):
            llm = FakeListChatModel(responses=["hello SECRET world"])

            @klafi_node("llm", after=[mask_secret])
            def node(state):
                return {"messages": [llm.invoke(state["messages"])]}

            self.add_node("llm", node)
            self.add_edge(START, "llm")
            self.add_edge("llm", END)

    return A(_spec(), hooks=hooks)


def test_g4_stream_tokens_are_post_guardrail():
    items = list(_masked_llm_agent().stream({"messages": [HumanMessage("q")]}, stream_mode=["updates", "messages"]))
    tokens = "".join(p[0].content for m, p in items if m == "messages")
    assert "SECRET" not in tokens and tokens == "hello *** world"  # 원문 토큰 미유출, 최종 본문 1회
    upd = [p for m, p in items if m == "updates"][0]
    assert upd["llm"]["messages"][-1].content == "hello *** world"


@pytest.mark.asyncio
async def test_g4_astream_tokens_are_post_guardrail():
    items = [it async for it in _masked_llm_agent().astream({"messages": [HumanMessage("q")]}, stream_mode=["updates", "messages"])]
    tokens = "".join(p[0].content for m, p in items if m == "messages")
    assert tokens == "hello *** world"


def test_g4_output_guardrail_block_applies_at_stream_end():
    class T(TypedDict):
        text: str

    class A(KlafiGraph):
        state_schema = T
        observability = False

        def define(self):
            @klafi_node("n")
            def n(state):
                return {"text": state["text"]}

            self.add_node("n", n)
            self.add_edge(START, "n")
            self.add_edge("n", END)

    ag = A(_spec(), hooks=[GuardrailHook(output=[block_secret])])
    assert list(ag.stream({"text": "ok"}))  # 통과
    with pytest.raises(ViolationError):
        list(ag.stream({"text": "has SECRET"}))  # 이전엔 스트림에서 after_agent 미발화 → 통과


# ── R4: 클라이언트 취소는 실패가 아니다 ────────────────────────────────────────
@pytest.mark.asyncio
async def test_r4_cancel_is_not_an_error():
    errs = []

    class Rec(Hook):
        def on_agent_error(self, input, exc, ctx):
            errs.append(("agent", type(exc).__name__))

        def on_node_error(self, node, state, exc, ctx):
            errs.append(("node", type(exc).__name__))

    class A(KlafiGraph):
        state_schema = S
        observability = False

        def define(self):
            @klafi_node("slow")
            async def slow(state):
                await asyncio.sleep(1)
                return {"x": 1}

            self.add_node("slow", slow)
            self.add_edge(START, "slow")
            self.add_edge("slow", END)

    ctx = ExecutionContext.new()

    async def run():
        async for _ in A(_spec(), hooks=[Rec()]).astream({"x": 0}, context=ctx):
            pass

    task = asyncio.create_task(run())
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert errs == [] and ctx.state == "CANCELLED"
