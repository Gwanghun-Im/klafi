"""Skill(툴 묶음 + 프롬프트) 등록·바인딩 검증."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from klafi import AgentSpec, KlafiGraph, Skill, ToolRegistry, tool
from klafi.model.gateway import ChatModel


@tool(name="lookup", description="주문 조회")
def lookup(order_id: str) -> dict:
    return {"order_id": order_id, "status": "배송중"}


@tool(name="refund", description="환불 처리")
def refund(order_id: str) -> str:
    return f"{order_id} 환불완료"


class _FakeModel:
    """bind_tools를 지원하는 최소 chat model — 받은 메시지를 그대로 기록."""

    def __init__(self) -> None:
        self.tools: list = []
        self.seen: list = []

    def bind_tools(self, tools):
        self.tools = tools
        return RunnableLambda(lambda msgs: self.seen.append(msgs) or AIMessage(content="ok"))


class State(dict):
    pass


class _G(KlafiGraph):
    spec = AgentSpec(id="g", name="G")
    state_schema = State
    observability = False

    def define(self):  # 그래프는 이 테스트의 관심사가 아님
        pass


def _graph():
    g = _G.__new__(_G)  # define() 없이 bind_tools만 확인
    from langgraph.graph.state import StateGraph

    g._sg = StateGraph(State)
    return g


# ── 구성 ────────────────────────────────────────────────────────────────
def test_skill_holds_tools_and_prompt():
    s = Skill(name="cs", tools=[lookup, refund], prompt="주문 확인 후 환불하라")
    assert [t.name for t in s.bind_tools()] == ["lookup", "refund"]


def test_from_registry_selects_by_name_or_all():
    reg = ToolRegistry()
    reg.register(lookup)
    reg.register(refund)
    assert [t.name for t in Skill.from_registry(reg, "cs", "refund").tools] == ["refund"]
    assert len(Skill.from_registry(reg, "all").tools) == 2


# ── 바인딩 ──────────────────────────────────────────────────────────────
def test_bind_skills_binds_tools_and_injects_prompt():
    model = _FakeModel()
    g = _graph()
    skill = Skill(name="cs", tools=[lookup, refund], prompt="주문 확인 후 환불하라")

    llm = ChatModel(model).bind_skills([skill])
    assert [t.name for t in model.tools] == ["lookup", "refund"]  # 툴 바인딩

    llm.invoke([HumanMessage("환불해줘")])
    sent = model.seen[0]
    assert sent[0].type == "system" and sent[0].content == "주문 확인 후 환불하라"  # 프롬프트 주입
    assert sent[1].content == "환불해줘"


def test_skill_and_plain_tool_mix_without_prompt():
    model = _FakeModel()
    llm = ChatModel(model).bind_skills([Skill(name="cs", tools=[lookup]), refund])
    assert [t.name for t in model.tools] == ["lookup", "refund"]

    llm.invoke([HumanMessage("안녕")])
    assert model.seen[0][0].type == "human"  # prompt 없으면 SystemMessage 미주입


def test_make_tool_node_flattens_skill():
    """ToolNode에는 Skill을 그대로 넘겨도 툴만 펼쳐 들어간다."""
    node = _graph().make_tool_node([Skill(name="cs", tools=[lookup, refund])])
    assert set(node.tools_by_name) == {"lookup", "refund"}


# ── 모델 선언 표준 ───────────────────────────────────────────────────────
def test_init_chat_model_resolves_alias_from_active_gateway():
    """모델 선언 표준은 init_chat_model("<alias>") 하나 — Factory가 조립 구간에 활성 Gateway를 잡아준다."""
    from klafi import ModelGateway, init_chat_model
    from klafi.core.exceptions import ModelNotConfiguredError, ModelNotFoundError
    from klafi.model.gateway import using_gateway

    class P:
        def __call__(self, prompt):
            return "x"

        def chat_model(self, **kw):
            return _FakeModel()

    # 활성 Gateway 바인딩 밖에서는 미구성
    with pytest.raises(ModelNotConfiguredError, match="ModelGateway 미구성"):
        init_chat_model("main")

    gw = ModelGateway()
    gw.register("main", P())
    with using_gateway(gw):  # factory.create() 가 조립 동안 하는 것과 동일
        llm = init_chat_model("main")
        assert hasattr(llm, "bind_skills")  # KLAFI 래퍼
        assert llm.seen == []  # 나머지 속성은 원본 chat model에 위임

        with pytest.raises(ModelNotFoundError):
            init_chat_model("없는alias")

    # 바인딩이 끝나면 다시 미구성으로 돌아온다 (전역 오염 없음)
    with pytest.raises(ModelNotConfiguredError):
        init_chat_model("main")
