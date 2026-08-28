"""업무개발자 영역 — 일정 안내 Agent (Skill 바인딩 예제).

Skill(clock_kst)을 bind_skills로 주입하면 툴은 ToolNode로, prompt("한국 시각이 필요하면
kst_now 툴을 사용한다")는 SystemMessage로 자동 주입된다. 업무 코드에 툴 사용법을
다시 적을 필요가 없다 — 지침은 Skill이 들고 다닌다.
"""

from langchain_core.messages import SystemMessage
from langgraph.graph import START, MessagesState
from langgraph.prebuilt import tools_condition

from klafi.core import AgentSpec, KlafiGraph, klafi_node
from klafi.model import init_chat_model

from ..skills import clock_kst


class ScheduleAgent(KlafiGraph):
    spec = AgentSpec(id="schedule", name="Schedule Agent", version="1.0.0", agent_type="react", model="main")
    state_schema = MessagesState

    def define(self):
        llm = init_chat_model("main").bind_skills([clock_kst])  # 툴 + 지침

        @klafi_node("agent")
        def agent(state):
            sys = SystemMessage("너는 일정 안내 비서다. 간결하게 한국어로 답하라.")
            return {"messages": [llm.invoke([sys, *state["messages"]])]}

        self.add_node("agent", agent)
        self.add_node("tools", self.make_tool_node([clock_kst]))
        self.add_edge(START, "agent")
        self.add_conditional_edges("agent", tools_condition)
        self.add_edge("tools", "agent")
