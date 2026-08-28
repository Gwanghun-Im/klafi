"""SOLID 감사에서 실증된 결함들의 회귀 방지 테스트.

각 테스트는 수정 전에는 실패했고, 수정 후 통과한다. 테스트 공백이 원래 결함을 가렸으므로
(특히 #1은 보안 영향) 재발을 막는다.
"""

import pytest


# ── #1 warn_only 가 마스킹을 삼키지 않는다 (보안) ──────────────────────────
def test_warn_only_preserves_masking():
    from klafi.guardrail import GuardrailResult, enforce, guardrail, warn_only

    @guardrail
    def mask_email(text):
        if "@" not in text:
            return GuardrailResult(True)
        return GuardrailResult(False, "PII", replacement=text.replace("@", "[X]"))

    # warn_only 는 '차단→경고'만 바꾼다. 마스킹(치환)은 그대로 적용돼야 한다.
    assert enforce([warn_only(mask_email)], "a@b.com", "output") == "a[X]b.com"
    # 감싸지 않은 원본과 동일한 치환 결과
    assert enforce([mask_email], "a@b.com", "output") == "a[X]b.com"


# ── #2 재시도 시 사용자 config 가 유실되지 않는다 ─────────────────────────
def test_config_not_consumed_on_repeated_calls():
    from klafi.core.base_graph import BaseGraph
    from klafi.core.context import ExecutionContext

    ctx = ExecutionContext.new(agent_id="t")
    kwargs = {"config": {"recursion_limit": 50}, "debug": True}
    g = object.__new__(BaseGraph)

    first = BaseGraph._config(g, ctx, kwargs, None)
    second = BaseGraph._config(g, ctx, kwargs, None)  # 재시도 시뮬레이션 (같은 kwargs)
    assert first["recursion_limit"] == 50
    assert second["recursion_limit"] == 50  # pop 이면 여기서 KeyError/None
    assert kwargs == {"config": {"recursion_limit": 50}, "debug": True}  # 원본 불변


# ── #3 AgentServer 는 fastapi 없이도 import 된다 (순수 런타임) ─────────────
def test_agent_server_import_is_fastapi_free():
    # server 패키지 __init__ 이 http/webui 를 eager import 하면 안 된다.
    import ast
    from pathlib import Path

    src = Path("klafi/server/__init__.py").read_text()
    tree = ast.parse(src)
    module_level = [
        n
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for n in [getattr(node, "module", None)]
    ]
    # http/webui(=fastapi 의존)는 모듈 최상단 import 에 없어야 한다 (lazy __getattr__).
    assert all(m not in ("klafi.server.http", "klafi.server.webui", ".http", ".webui") for m in module_level)
    # AgentServer 자체는 정상 노출
    from klafi.server import AgentServer

    assert AgentServer().ids() == []


# ── #4 provider registry: 확장 가능 + 미지 항목 fail-fast ─────────────────
def test_register_provider_is_extensible():
    from klafi.model import register_provider, resolve_provider

    register_provider("dummy", lambda spec: f"provider:{spec.get('model')}")
    assert resolve_provider({"type": "dummy", "model": "x"}) == "provider:x"


def test_unknown_provider_type_fails_fast():
    from klafi.core.exceptions import ModelException
    from klafi.model import resolve_provider

    with pytest.raises(ModelException):
        resolve_provider({"type": "no-such-provider"})


def test_gateway_config_wires_fallback_and_rejects_typos():
    from klafi.app.application import _build_gateway
    from klafi.core.exceptions import ConfigSchemaError

    gw = _build_gateway(
        {"providers": {"main": {"type": "echo", "fallback": "backup"}, "backup": {"type": "echo"}}}
    )
    assert gw._entry("main").fallback == "backup"  # MOD-08 이 config 경로에서 도달 가능

    with pytest.raises(ConfigSchemaError):  # 오타를 조용히 무시하지 않는다
        _build_gateway({"providers": {"main": {"type": "echo", "fallbck": "oops"}}})


# ── #7 두 factory 가 서로 다른 gateway 로 충돌하지 않는다 ──────────────────
def test_multiple_factories_do_not_collide():
    """factory 2개를 서로 다른 gateway 로 만들어도, 각 에이전트는 자기 factory 의 gateway 를 본다.
    (전역 _ACTIVE 이던 시절엔 나중에 만든 factory 가 조용히 이겼다.)"""
    from typing import TypedDict

    from langgraph.graph import END, START

    from klafi import AgentSpec, ExecutionFactory, KlafiGraph, ModelGateway
    from klafi.core import klafi_node
    from klafi.model import init_chat_model

    class S(TypedDict):
        q: str
        a: str

    def make_gw(tag):
        gw = ModelGateway()

        class P:
            def __call__(self, p):
                return tag

            def chat_model(self, **kw):
                class M:
                    def invoke(self, *a, **k):
                        return tag

                return M()

        gw.register("main", P())
        return gw

    class Agent(KlafiGraph):
        spec = AgentSpec(id="a", name="A", version="1.0.0", model="main")
        state_schema = S
        observability = False

        def define(self):
            llm = init_chat_model("main")

            @klafi_node("n")
            def n(state):
                return {"a": llm.invoke([])}

            self.add_node("n", n)
            self.add_edge(START, "n")
            self.add_edge("n", END)

    fa = ExecutionFactory(gateway=make_gw("A"))
    fb = ExecutionFactory(gateway=make_gw("B"))  # 나중에 생성 (예전엔 이게 전역을 이겼다)

    assert fa.create(Agent).invoke({"q": "x", "a": ""})["a"] == "A"
    assert fb.create(Agent).invoke({"q": "x", "a": ""})["a"] == "B"


def test_gateway_binding_does_not_leak_globally():
    """using_gateway 블록 밖에서는 init_chat_model 이 미구성이어야 한다 (전역 오염 없음)."""
    from klafi.core.exceptions import ModelNotConfiguredError
    from klafi.model import ModelGateway, init_chat_model, using_gateway

    gw = ModelGateway()
    with using_gateway(gw):
        pass
    with pytest.raises(ModelNotConfiguredError):
        init_chat_model("main")


# ── #8 factory.create 가 조립 불가 클래스를 친절히 안내한다 (LSP 힌트 정직화) ──
def test_factory_gives_helpful_error_for_non_assemblable_templates():
    """RAGAgent/SupervisorAgent 처럼 생성자에 필수 의존성이 있는 템플릿은 factory 로 못 만든다.
    이전엔 `missing positional argument: retriever` 로 죽었다 → 이제 안내 메시지."""
    from klafi import AgentSpec, ExecutionFactory, ModelGateway
    from klafi.core.exceptions import AgentExecutionException
    from klafi.templates import RAGAgent

    class DocsRAG(RAGAgent):  # spec 은 줬지만 retriever 가 필수
        spec = AgentSpec(id="docs", name="Docs", version="1.0.0", model="main")

    f = ExecutionFactory(gateway=ModelGateway())
    with pytest.raises(AgentExecutionException) as ei:
        f.create(DocsRAG)
    msg = str(ei.value)
    assert "조립할 수 없습니다" in msg
    assert "server.register" in msg  # 대안 경로 안내


def test_factory_spec_missing_is_friendly():
    from klafi import ExecutionFactory, ModelGateway
    from klafi.core.exceptions import AgentExecutionException
    from klafi.templates import SimpleAgent

    with pytest.raises(AgentExecutionException, match="spec"):
        ExecutionFactory(gateway=ModelGateway()).create(SimpleAgent)


# ── #6 Tool 권한 거부가 감사 이벤트를 남긴다 ──────────────────────────────
def test_tool_permission_denial_is_audited():
    from klafi.core.context import ExecutionContext, bind_context
    from klafi.events import EventType, subscribe
    from klafi.events.bus import EVENTS
    from klafi.tool import tool

    seen = []
    subscribe(lambda e: seen.append(e.type))
    try:
        @tool(required_permission="trades:write")
        def buy(x: int) -> int:
            return x

        ctx = ExecutionContext.new(agent_id="t", security_context={"permissions": []})
        with bind_context(ctx):
            with pytest.raises(Exception):
                buy.run(x=1)
        # 권한 거부도 span/이벤트 경계 안 → 감사 흔적이 남아야 한다
        assert EventType.ToolStarted in seen
        assert EventType.ToolFailed in seen
    finally:
        EVENTS.clear()


# ── #9 echo(키 없는) provider 가 init_chat_model 표준 경로에서 동작한다 (ISP) ──
def test_function_provider_supports_chat_model_path():
    """FunctionProvider 가 chat_model() 을 지원 → 키 없이 예제 스타일(init_chat_model.bind_tools)
    에이전트를 테스트/데모할 수 있다. 이전엔 ModelNotConfiguredError 로 막혔다."""
    from typing import TypedDict

    from langchain_core.messages import SystemMessage
    from langgraph.graph import END, START, MessagesState

    from klafi import AgentSpec, ExecutionFactory, KlafiGraph, ModelGateway
    from klafi.core import klafi_node
    from klafi.model import FunctionProvider, init_chat_model

    gw = ModelGateway()
    gw.register("main", FunctionProvider(lambda p: f"echo:{p[-6:]}"))

    class Agent(KlafiGraph):
        spec = AgentSpec(id="a", name="A", version="1.0.0", model="main")
        state_schema = MessagesState
        observability = False

        def define(self):
            llm = init_chat_model("main").bind_tools([])  # 예제와 동일한 표준 스타일

            @klafi_node("agent")
            def agent(state):
                return {"messages": [llm.invoke([SystemMessage("s"), *state["messages"]])]}

            self.add_node("agent", agent)
            self.add_edge(START, "agent")
            self.add_edge("agent", END)

    out = ExecutionFactory(gateway=gw).create(Agent).invoke(
        {"messages": [{"role": "user", "content": "안녕"}]}
    )
    assert out["messages"][-1].content.startswith("echo:")
