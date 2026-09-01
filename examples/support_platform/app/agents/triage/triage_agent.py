"""업무개발자 영역 — 문의 분류 Agent (노드별 다른 모델·다른 툴).

    triage(fast)  ──simple──▶ quick(fast + lookup_order)   ──▶ toolsQ
                  └─complex──▶ expert(expert + search_policy) ──▶ toolsE

- 노드별 모델: init_chat_model("fast" / "expert")
- 노드별 툴셋: self.make_tool_node([...]) 로 각각 다른 ToolNode
같은 그래프 안에서 값싼 모델로 분류하고, 어려운 건만 상위 모델로 보낸다(비용 최적화).

구조: spec → agentSpec.py · state → state.py · 그래프 → 이 파일 (에이전트별 패키지).
"""

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START
from langgraph.prebuilt import tools_condition

from klafi.core import KlafiGraph, klafi_graph, klafi_node
from klafi.model import init_chat_model

from common.middleware import audit_log
from common.guardrails import refund_policy
from ...tools import lookup_order, search_policy
from .agentSpec import spec
from .prompt import EXPERT, QUICK, TRIAGE
from .state import TriageState


@klafi_graph(before=[refund_policy])  # triage 전용 — 워크플로우 입력 가드레일
class TriageAgent(KlafiGraph):
    spec = spec
    state_schema = TriageState

    def define(self):
        # ── 노드별로 다른 모델 + 다른 툴셋 ────────────────────────────────
        fast = init_chat_model("fast")  # 분류용 (툴 없음)
        quick_llm = init_chat_model("fast").bind_tools([lookup_order])
        expert_llm = init_chat_model("expert").bind_tools([search_policy])

        @klafi_node("triage", before=[audit_log])  # 진입 노드에서 audit 미들웨어
        def triage(state: TriageState) -> dict:
            sys = SystemMessage(TRIAGE)
            verdict = fast.invoke([sys, *state["messages"]]).content.strip().lower()
            return {"route": "complex" if "complex" in verdict else "simple"}

        @klafi_node("quick")
        def quick(state: TriageState) -> dict:
            sys = SystemMessage(QUICK)
            return {"messages": [quick_llm.invoke([sys, *state["messages"]])]}

        @klafi_node("expert")
        def expert(state: TriageState) -> dict:
            sys = SystemMessage(EXPERT)
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
