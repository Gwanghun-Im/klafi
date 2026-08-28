"""Tool bind_tools (LangGraph 네이티브) 검증 — KLAFI Tool을 ToolNode로 실행."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import START, MessagesState
from langgraph.prebuilt import tools_condition
from pydantic import BaseModel

from klafi import AgentSpec, ExecutionContext, Hook, KlafiGraph, klafi_node
from klafi.core.context import bind_context
from klafi.core.exceptions import ToolException
from klafi.tool import tool


class LookupIn(BaseModel):
    order_id: str


@tool(name="lookup", description="주문 조회", required_permission="orders:read", input_schema=LookupIn)
def lookup(order_id: str) -> dict:
    return {"order_id": order_id, "status": "배송중"}


# ── as_langchain: 변환해도 권한·검증 그대로 ─────────────────────────────
def test_as_langchain_keeps_permission():
    lc = lookup.as_langchain()
    assert lc.name == "lookup"
    # 권한 컨텍스트 없으면 차단
    with pytest.raises(ToolException):
        lc.invoke({"order_id": "A1"})
    # 권한 있으면 실행
    with bind_context(ExecutionContext.new(security_context={"permissions": ["orders:read"]})):
        assert lc.invoke({"order_id": "A1"})["status"] == "배송중"


# ── bind_tools + ToolNode: 네이티브 tool-calling 루프 ───────────────────
def _tool_calls_seen():
    return []


def test_bind_tools_toolnode_loop_runs_klafi_tool():
    seen_tool = []

    class TraceHook(Hook):
        def before_tool(self, t, kw, c):
            seen_tool.append(t)

    class ToolAgent(KlafiGraph):
        spec = AgentSpec(id="ta", name="ToolAgent")
        state_schema = MessagesState
        observability = False

        def define(self):

            @klafi_node("agent")
            def agent(state):
                last = state["messages"][-1]
                if isinstance(last, ToolMessage):  # tool 결과 받으면 종료
                    return {"messages": [AIMessage(content="처리완료")]}
                # 첫 턴: tool 호출 요청 (실제로는 chat model이 생성)
                return {"messages": [AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"order_id": "A1"}, "id": "c1"}])]}

            self.add_node("agent", agent)
            self.add_node("tools", self.make_tool_node([lookup]))  # LangGraph 네이티브 ToolNode
            self.add_edge(START, "agent")
            self.add_conditional_edges("agent", tools_condition)
            self.add_edge("tools", "agent")

    agent = ToolAgent(hooks=[TraceHook()])
    ctx = ExecutionContext.new(security_context={"permissions": ["orders:read"]})
    out = agent.invoke({"messages": [HumanMessage("A1 주문 조회")]}, context=ctx)

    # ToolNode가 KLAFI lookup을 실행 → before_tool Hook 발화(= 권한·검증·훅 경로)
    assert "lookup" in seen_tool
    # tool 결과가 대화에 들어가고 최종 응답 생성
    contents = [m.content for m in out["messages"] if isinstance(m, (AIMessage, ToolMessage))]
    assert "처리완료" in contents
    assert any("배송중" in str(c) for c in contents)  # ToolMessage에 tool 결과


# ── 노드별 다른 툴셋 (make_tool_node) ───────────────────────────────────
def test_per_node_different_toolsets():
    from langgraph.graph import MessagesState

    @tool(name="ta")
    def ta(x: str) -> str:
        return f"A:{x}"

    @tool(name="tb")
    def tb(x: str) -> str:
        return f"B:{x}"

    class Multi(KlafiGraph):
        spec = AgentSpec(id="multi", name="Multi")
        state_schema = MessagesState
        observability = False

        def define(self):
            @klafi_node("cA")
            def cA(s):
                return {"messages": [AIMessage(content="", tool_calls=[{"name": "ta", "args": {"x": "1"}, "id": "a"}])]}

            @klafi_node("cB")
            def cB(s):
                return {"messages": [AIMessage(content="", tool_calls=[{"name": "tb", "args": {"x": "2"}, "id": "b"}])]}

            self.add_node("cA", cA)
            self.add_node("toolsA", self.make_tool_node([ta]))  # 노드별 툴셋
            self.add_node("cB", cB)
            self.add_node("toolsB", self.make_tool_node([tb]))
            self.add_edge(START, "cA")
            self.add_edge("cA", "toolsA")
            self.add_edge("toolsA", "cB")
            self.add_edge("cB", "toolsB")

    out = Multi().invoke({"messages": [HumanMessage("go")]})
    results = [m.content for m in out["messages"] if isinstance(m, ToolMessage)]
    assert "A:1" in results and "B:2" in results  # 각 노드가 서로 다른 툴 실행


def test_per_node_models_declared_with_init_chat_model():
    """노드별 다른 모델은 define() 안에서 init_chat_model(alias)로 선언한다."""
    from langgraph.graph import MessagesState

    from klafi import ExecutionFactory, ModelGateway, init_chat_model

    class P:  # chat_model을 지원하는 테스트 provider
        def __init__(self, tag):
            self.tag = tag

        def __call__(self, prompt):
            return self.tag

        def chat_model(self, **kw):
            return self  # .tag 로 어떤 alias가 왔는지 확인

    gw = ModelGateway()
    gw.register("fast", P("f"))
    gw.register("strong", P("s"))
    seen = {}

    class A(KlafiGraph):
        spec = AgentSpec(id="a", name="A", model="fast")
        state_schema = MessagesState

        def define(self):
            seen["fast"] = init_chat_model("fast")
            seen["strong"] = init_chat_model("strong")

            @klafi_node("n")
            def n(s):
                return {"messages": [AIMessage(content="ok")]}

            self.add_node("n", n)
            self.add_edge(START, "n")

    ExecutionFactory(gateway=gw).create(A)
    assert {k: v.tag for k, v in seen.items()} == {"fast": "f", "strong": "s"}
