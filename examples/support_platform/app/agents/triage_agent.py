"""업무개발자 영역 — 문의 분류 Agent (노드별 다른 모델·다른 툴).

    triage(fast)  ──simple──▶ quick(fast + lookup_order)   ──▶ toolsQ
                  └─complex──▶ expert(expert + search_policy) ──▶ toolsE

- 노드별 모델: init_chat_model("fast" / "expert")
- 노드별 툴셋: self.make_tool_node([...]) 로 각각 다른 ToolNode
같은 그래프 안에서 값싼 모델로 분류하고, 어려운 건만 상위 모델로 보낸다(비용 최적화).
"""

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState
from langgraph.prebuilt import tools_condition

from klafi.core import AgentSpec, KlafiGraph, klafi_graph, klafi_node
from klafi.model import init_chat_model

from ...platform.middleware import audit_log
from ...platform.guardrails import refund_policy
from ..tools import lookup_order, search_policy


class TriageState(MessagesState):
    route: str  # "simple" | "complex"


@klafi_graph(before=[refund_policy])  # triage 전용 — 워크플로우 입력 가드레일
class TriageAgent(KlafiGraph):
    spec = AgentSpec(id="triage", name="Triage Agent", version="1.0.0", agent_type="router", model="fast")
    state_schema = TriageState

    def define(self):
        # ── 노드별로 다른 모델 + 다른 툴셋 ────────────────────────────────
        fast = init_chat_model("fast")  # 분류용 (툴 없음)
        quick_llm = init_chat_model("fast").bind_tools([lookup_order])
        expert_llm = init_chat_model("expert").bind_tools([search_policy])

        @klafi_node("triage", before=[audit_log])  # 진입 노드에서 audit 미들웨어
        def triage(state: TriageState) -> dict:
            sys = SystemMessage("문의를 분류하라. 주문/배송 조회면 'simple', 규정·정책 해석이면 'complex'. 한 단어만 답하라.")
            verdict = fast.invoke([sys, *state["messages"]]).content.strip().lower()
            return {"route": "complex" if "complex" in verdict else "simple"}

        @klafi_node("quick")
        def quick(state: TriageState) -> dict:
            sys = SystemMessage("주문 조회 담당. 필요하면 lookup_order로 조회해 간결히 답하라.")
            return {"messages": [quick_llm.invoke([sys, *state["messages"]])]}

        @klafi_node("expert")
        def expert(state: TriageState) -> dict:
            sys = SystemMessage("정책 상담 담당. search_policy로 규정을 확인하고 근거와 함께 답하라.")
            return {"messages": [expert_llm.invoke([sys, *state["messages"]])]}

        self.add_node("triage", triage)
        self.add_node("quick", quick)
        self.add_node("expert", expert)
        self.add_node("toolsQ", self.make_tool_node([lookup_order]))  # quick 전용 툴셋
        self.add_node("toolsE", self.make_tool_node([search_policy]))  # expert 전용 툴셋

        self.add_edge(START, "triage")
        self.add_conditional_edges("triage", lambda s: s["route"], {"simple": "quick", "complex": "expert"})
        self.add_conditional_edges("quick", tools_condition, {"tools": "toolsQ", END: END})
        self.add_conditional_edges("expert", tools_condition, {"tools": "toolsE", END: END})
        self.add_edge("toolsQ", "quick")
        self.add_edge("toolsE", "expert")
