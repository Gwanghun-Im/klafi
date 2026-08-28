"""업무개발자 영역 — 고객지원 Agent (bind_tools, LangGraph 네이티브 tool-calling).

LLM이 lookup_order Tool을 스스로 호출하는 ReAct 루프. Tool은 bind_tools로 연결하고
LangGraph ToolNode가 실행한다(권한·검증·Hook은 KLAFI Tool.run에서 그대로 적용).
"""

from langchain_core.messages import SystemMessage
from langgraph.config import get_store
from langgraph.graph import START, MessagesState
from langgraph.prebuilt import tools_condition

from klafi.core import AgentSpec, KlafiGraph, get_context, klafi_node
from klafi.context.memory import user_scope
from klafi.guardrail import pii, warn_only
from klafi.model import init_chat_model

from common.middleware import require_orders_read
from common.guardrails import mask_phone
from ..tools import lookup_order


class SupportAgent(KlafiGraph):
    spec = AgentSpec(
        id="support",
        name="Support Agent",
        version="1.0.0",
        agent_type="react",
        model="main",
        owner="EAMES",
    )
    state_schema = MessagesState

    def define(self):
        llm = init_chat_model("main").bind_tools([lookup_order])

        # @klafi_node — before/after 한 리스트에 미들웨어와 가드레일을 섞어 넣는다.
        # (before: 권한 확인 미들웨어 / after: 전화번호 마스킹 → PII 경고)
        # 상담 답변에는 주문자 이메일 등이 정상적으로 등장할 수 있어 차단 대신 경고 등급으로 둔다.
        # (차단이 필요하면 warn_only 를 벗기고 pii 를 그대로 쓰면 된다)
        @klafi_node("agent", before=[require_orders_read], after=[mask_phone, warn_only(pii)])
        def agent(state):
            ctx = get_context()
            pref = get_store().get(
                user_scope(ctx.user_id or "anon"), "pref"
            )  # 공통 Memory
            lang = pref.value.get("lang") if pref else "ko"
            sys = SystemMessage(
                f"너는 고객지원 상담원이다. 주문 문의는 lookup_order로 조회해 {lang}로 답하라."
            )
            return {"messages": [llm.invoke([sys, *state["messages"]])]}

        self.add_node("agent", agent)
        self.add_node(
            "tools", self.make_tool_node([lookup_order])
        )  # LangGraph ToolNode
        self.add_edge(START, "agent")
        self.add_conditional_edges("agent", tools_condition)  # tool_call 있으면 tools로
        self.add_edge("tools", "agent")
