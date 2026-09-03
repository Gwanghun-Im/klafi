"""v0.1.5 결함 수정 검증 (2) — runtime·guardrail·context·model·tool·observability·evaluation.

테스트 ID 는 결함 대장 기준. 각각 재현된 오동작 하나를 고정한다.
"""

from __future__ import annotations

import logging
import time
from typing import TypedDict

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_core.tools import ToolException as LCToolException
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState
from pydantic import BaseModel

from klafi import AgentSpec, KlafiGraph
from klafi.context.checkpoint import SyncSaverAsyncAdapter, resolve_checkpointer
from klafi.context.manager import ContextManager
from klafi.core import klafi_node
from klafi.core.exceptions import ConfigNotFoundError, ConfigSchemaError, TimeoutException, ToolException
from klafi.guardrail import GuardrailHook, GuardrailResult, enforce, guardrail, pii
from klafi.model.gateway import ChatModel, FunctionProvider, ModelGateway
from klafi.runtime.policy import ExecutionPolicy
from klafi.tool.skill import Skill
from klafi.tool.tool import tool


class S(TypedDict):
    x: int


def _spec() -> AgentSpec:
    return AgentSpec(id="t", name="T")


@guardrail
def mask_secret(text: str) -> GuardrailResult:
    return GuardrailResult("SECRET" not in text, "secret", replacement=text.replace("SECRET", "***"))


# ── R1/R2/R3: 런타임 정책 ──────────────────────────────────────────────────────
def test_r1_sync_timeout_is_not_retried_concurrently():
    from klafi.runtime.engine import run_sync

    starts: list[float] = []

    def slow():
        starts.append(time.monotonic())
        time.sleep(0.3)
        return "done"

    pol = ExecutionPolicy(timeout=0.05, max_retries=2, backoff_base=0.0, jitter=False)
    with pytest.raises(TimeoutException):
        run_sync(slow, pol, lambda s: None)
    time.sleep(0.4)
    assert len(starts) == 1  # 이전엔 timeout 마다 재시도해 같은 ctx 위에서 3개가 동시에 돌았다


def test_r2_deterministic_errors_not_retried_and_backoff_has_jitter():
    p = ExecutionPolicy(max_retries=3, backoff_base=1.0, backoff_factor=1.0, backoff_max=10)
    assert p.should_retry(TypeError(), 0) is False and p.should_retry(ValueError(), 0) is False
    assert p.should_retry(ConnectionError(), 0) is True  # OSError 하위지만 일시 장애
    assert 1.0 <= p.backoff_delay(0) <= 2.0  # 0~min(delay,1s) 무작위 가산
    assert ExecutionPolicy(jitter=False, backoff_base=1.0).backoff_delay(0) == 1.0


def test_r3_tool_timeout_message_says_it_is_not_a_cancel():
    @tool("slow", policy=ExecutionPolicy(timeout=0.05))
    def slow() -> int:
        time.sleep(0.15)
        return 1

    with pytest.raises(TimeoutException) as ei:
        slow.run()
    assert "취소" in str(ei.value)


# ── G1/G2/G3/G5: 가드레일 ──────────────────────────────────────────────────────
def test_g1_parallel_tool_messages_are_all_scanned():
    ai = AIMessage("", tool_calls=[{"name": "a", "args": {}, "id": "1"}, {"name": "b", "args": {}, "id": "2"}])
    state = {"messages": [HumanMessage("q"), ai, ToolMessage("phone SECRET", tool_call_id="1"),
                          ToolMessage("clean", tool_call_id="2")]}
    out = enforce([mask_secret], state, "input")
    assert out["messages"][2].content == "phone ***"  # 이전엔 마지막(Tool2)만 봐서 통과
    assert out["messages"][3].content == "clean" and out["messages"][0] is state["messages"][0]


def test_g2_mask_guardrail_on_model_stage_is_rejected_and_scan_is_last_message_only():
    from klafi.guardrail import LLMGuardrail
    from klafi.model.callback import _messages_to_text

    g = LLMGuardrail("mod", lambda p: "SAFE", "no secrets", action="mask")
    with pytest.raises(ConfigSchemaError):
        GuardrailHook(model=[g])  # 콜백 경로는 판정 전용 — 조용히 무시되던 것을 등록 시점에 거부
    assert _messages_to_text([[HumanMessage("old SECRET"), HumanMessage("new")]]) == "new"


def test_g3_pydantic_tool_output_is_masked_not_rejected():
    class Out(BaseModel):
        phone: str
        n: int = 1

    out = enforce([mask_secret], Out(phone="SECRET-1"), "tool_output")
    assert isinstance(out, Out) and out.phone == "***-1" and out.n == 1
    same = Out(phone="clean")
    assert enforce([mask_secret], same, "tool_output") is same


def test_g5_card_number_requires_luhn_and_email_regex_is_strict():
    assert pii.check("주문번호 1234567890123456").allowed  # Luhn 실패 → 카드번호 아님
    assert not pii.check("카드 4111111111111111").allowed
    assert not pii.check("메일 a@b.com 으로").allowed and pii.check("a@b.c 는 메일 아님").allowed


# ── X1/X2/X3/X4: 컨텍스트·체크포인트 ─────────────────────────────────────────────
def test_x1_summary_is_cached_across_nodes_and_turns():
    calls: list[str] = []
    cm = ContextManager(keep_recent=1, summarizer=lambda p: calls.append(p) or "요약")
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]
    cm.reduce(msgs)
    cm.reduce(msgs)  # 같은 턴의 두 번째 노드
    assert len(calls) == 1


def test_x2_default_counter_counts_tool_calls_and_cjk():
    cm = ContextManager()
    ai = AIMessage("", tool_calls=[{"name": "search", "args": {"q": "x" * 200}, "id": "1"}])
    assert cm.count_tokens([ai]) > 20  # 공백 계수는 0 이었다
    assert cm.count_tokens([HumanMessage("한국어 문장입니다 " * 10)]) > 20


def test_x3_important_messages_keep_original_order():
    msgs = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2", "important": True}, {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"}, {"role": "assistant", "content": "a3"}]
    out = ContextManager(keep_recent=2).reduce(msgs)
    assert [m["content"] for m in out] == ["q2", "q3", "a3"]  # important 를 맨 앞으로 끌어올리지 않는다


@pytest.mark.asyncio
async def test_x4_sync_only_saver_works_under_ainvoke():
    class SyncOnly(InMemorySaver):  # PostgresSaver 처럼 async 메서드가 기본(NotImplementedError)인 saver
        aget_tuple = BaseCheckpointSaver.aget_tuple
        aput = BaseCheckpointSaver.aput
        aput_writes = BaseCheckpointSaver.aput_writes
        alist = BaseCheckpointSaver.alist

    saver = resolve_checkpointer(SyncOnly())
    assert isinstance(saver, SyncSaverAsyncAdapter)

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

    ag = A(_spec(), checkpointer=saver)
    assert (await ag.ainvoke({"x": 1}, thread_id="t"))["x"] == 2  # 이전엔 NotImplementedError
    assert ag.invoke({"x": 5}, thread_id="t2")["x"] == 6
    assert ag.get_state(thread_id="t").values["x"] == 2


# ── M1/M2/M3/M4/M5/M6: 모델·설정 ────────────────────────────────────────────────
def test_m1_alias_policy_reaches_chat_model():
    seen: dict = {}

    class P:
        def __call__(self, p):
            return "x"

        def chat_model(self, callbacks=None, **kw):
            seen.update(kw)
            return FakeListChatModel(responses=["ok"], callbacks=callbacks)

    gw = ModelGateway()
    gw.register("m", P(), policy=ExecutionPolicy(timeout=3, max_retries=1))
    assert gw.chat_model("m") is not None and seen == {"timeout": 3, "max_retries": 1}


def test_m2_mutual_fallback_does_not_recurse_and_chat_path_has_fallback():
    def down(p):
        raise RuntimeError("down")

    gw = ModelGateway()
    gw.register("a", FunctionProvider(down), fallback="b")
    gw.register("b", FunctionProvider(down), fallback="a")
    with pytest.raises(RuntimeError):  # 이전엔 RecursionError
        gw.model("a")("x")

    gw2 = ModelGateway()
    gw2.register("primary", FunctionProvider(down), fallback="backup")
    gw2.register("backup", FunctionProvider(lambda p: "from backup"))
    llm = ChatModel(gw2.chat_model("primary"), fallbacks=gw2.chat_fallbacks("primary"))
    assert llm.invoke([HumanMessage("hi")]).content == "from backup"  # init_chat_model 경로에도 폴백


def test_m3_mcp_env_expansion_fails_fast_like_layered_config(monkeypatch):
    from klafi.tool.mcp import _expand_env

    monkeypatch.delenv("KLAFI_UNSET_X", raising=False)
    with pytest.raises(ConfigNotFoundError):  # 이전엔 빈 문자열로 통과 → 키 없는 서버가 떴다
        _expand_env({"env": {"K": "${KLAFI_UNSET_X}"}})
    assert _expand_env({"k": "${KLAFI_UNSET_X:dflt}"}) == {"k": "dflt"}


def test_m4_derived_methods_apply_to_model_not_prompt_sequence():
    fake = FakeListChatModel(responses=["ok"])
    llm = ChatModel(fake).bind_skills([Skill(name="s", tools=[], prompt="P")]).bind(stop=["zzz"]).with_retry(stop_after_attempt=2)
    assert llm.invoke([HumanMessage("hi")]).content == "ok"  # 이전엔 bind(stop=) → 앞단 람다로 가서 TypeError
    assert callable(llm.with_structured_output)  # 시퀀스가 아니라 모델에 적용되는 파생


def test_m5_provider_key_is_secret():
    from klafi.model.providers import AnthropicProvider

    p = AnthropicProvider("m", api_key="sk-secret")
    assert "sk-secret" not in repr(vars(p)) and p._key.get_secret_value() == "sk-secret"


def test_m6_model_params_are_accepted_from_config():
    from klafi.app.application import _build_gateway
    from klafi.model.providers import AnthropicProvider

    assert AnthropicProvider("m", temperature=0.2)._kwargs == {"temperature": 0.2}
    gw = _build_gateway({"providers": {"main": {"type": "echo", "model": "m", "params": {"temperature": 0.2}}}})
    assert gw.has("main")  # 이전엔 ConfigSchemaError '알 수 없는 항목: [params]'


# ── T1/T2/T4b: 툴 ──────────────────────────────────────────────────────────────
def test_t1_tool_permission_error_becomes_error_tool_message():
    @tool("secure", required_permission="admin")
    def secure() -> str:
        return "ok"

    class A(KlafiGraph):
        state_schema = MessagesState
        observability = False

        def define(self):
            self.add_node("tools", self.make_tool_node([secure]))
            self.add_edge(START, "tools")
            self.add_edge("tools", END)

    out = A(_spec()).invoke({"messages": [AIMessage("", tool_calls=[{"name": "secure", "args": {}, "id": "c1"}])]})
    tm = out["messages"][-1]
    assert tm.type == "tool" and tm.status == "error" and "권한" in tm.content  # 이전엔 그래프 실행 중단


def test_t2_langchain_tool_error_status_is_not_swallowed():
    from klafi.tool.mcp import from_langchain_tool

    def boom(q: str) -> str:
        raise LCToolException("mcp failure")

    lc = StructuredTool.from_function(boom, name="boom", description="d", handle_tool_error=True)
    with pytest.raises(ToolException, match="mcp failure"):  # 이전엔 status=error 가 성공 content 로 둔갑
        from_langchain_tool(lc).run(q="x")
    ok = StructuredTool.from_function(lambda q: f"got {q}", name="ok", description="d")
    assert from_langchain_tool(ok).run(q="x") == "got x"


def test_t4b_async_function_is_rejected_by_tool_decorator():
    with pytest.raises(ToolException):

        @tool("asy")
        async def asy() -> int:  # 이전엔 '<coroutine object>' 문자열이 결과로 나갔다
            return 1


# ── O1/O3: 관측성·평가 ───────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_o1_callback_under_async_invoke_has_no_context_detach_error(caplog):
    from klafi.model.callback import KlafiCallbackHandler

    llm = FakeListChatModel(responses=["hi"], callbacks=[KlafiCallbackHandler("fake")])
    with caplog.at_level(logging.ERROR, logger="opentelemetry.context"):
        await llm.ainvoke("q")
    assert not [r for r in caplog.records if "detach" in r.getMessage()]


def test_o3_judge_score_parsing():
    from klafi.evaluation.evaluator import _parse_score

    assert _parse_score("8/10") == 0.8 and _parse_score("Score: 7") == 0.7
    assert _parse_score("There are 2 factual errors. Score: 0.2") == 0.2  # 이전엔 첫 숫자 2 → 1.0
    assert _parse_score('{"score": 0.35}') == 0.35 and _parse_score("0.9") == 0.9
