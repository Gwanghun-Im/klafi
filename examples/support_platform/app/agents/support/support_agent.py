"""업무개발자 영역 — 고객지원 Agent (bind_tools, LangGraph 네이티브 tool-calling).

LLM이 lookup_order Tool을 스스로 호출하는 ReAct 루프. Tool은 bind_tools로 연결하고
LangGraph ToolNode가 실행한다(권한·검증·Hook은 KLAFI Tool.run에서 그대로 적용).

구조: spec → agentSpec.py · state → state.py · 그래프 → 이 파일 (에이전트별 패키지).
"""

from langchain_core.messages import SystemMessage
from langgraph.config import get_store
from langgraph.graph import START
from langgraph.prebuilt import tools_condition

from klafi.core import KlafiGraph, get_context, klafi_node
from klafi.context.memory import user_scope
from klafi.model import init_chat_model

from common.middleware import require_orders_read
from common.mcp import search  # MCP 외부 도구(Tavily 웹검색). 미설정/미설치면 [] → lookup_order 만.
from ...tools import lookup_order
from .agentSpec import spec
from .prompt import SYSTEM
from .state import MessagesState


class SupportAgent(KlafiGraph):
    spec = spec
    state_schema = MessagesState

    def define(self):
        # MCP 도구(Tavily)도 KLAFI Tool 로 감싸져(from_langchain_tool) 권한(web:search)·검증·audit·가드레일을 탄다.
        tools = [lookup_order, *search]
        llm = init_chat_model("main").bind_tools(tools)

        # 노드 미들웨어 — before 에 권한 확인. (출력 마스킹/PII 경고는 플랫폼 공통 훅으로 올렸다:
        # common/hooks.py 의 GuardrailHook(output=[mask_phone, warn_only(pii)]) → 전 에이전트 적용)
        @klafi_node("agent", before=[require_orders_read])
        def agent(state):
            ctx = get_context()
            pref = get_store().get(
                user_scope(ctx.user_id or "anon"), "pref"
            )  # 공통 Memory
            lang = pref.value.get("lang") if pref else "ko"
            sys = SystemMessage(SYSTEM.format(lang=lang))
            return {"messages": [llm.invoke([sys, *state["messages"]])]}

        self.add_node("agent", agent)
        self.add_node("tools", self.make_tool_node(tools))  # LangGraph ToolNode (lookup_order + MCP fs)
        self.add_edge(START, "agent")
        self.add_conditional_edges("agent", tools_condition)  # tool_call 있으면 tools로
        self.add_edge("tools", "agent")
