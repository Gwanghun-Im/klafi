"""채팅 호환 에이전트 — MessagesState + 단일 LLM 노드 (툴 없음).

템플릿 SimpleAgent는 {question}→{answer} 스키마라 채팅 UI(messages)와 맞지 않는다.
채팅 프론트에서 바로 쓰려면 이렇게 state_schema=MessagesState 로 define() 한다 —
가장 단순한 대화 에이전트(툴/HITL 없음). 툴이 필요하면 support_agent 처럼 bind_tools 를 얹으면 된다.

(참고) 정형 입출력이면 템플릿이 더 짧다:  class Faq(SimpleAgent): spec = AgentSpec(..., model="main")
       단, 그 경우 입력은 {"question": ...} 이고 Swagger/API 로 호출한다.
"""

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState

from klafi.core import AgentSpec, KlafiGraph, klafi_node
from klafi.model import init_chat_model


class FaqAgent(KlafiGraph):
    spec = AgentSpec(
        id="faq", name="FAQ Agent", version="1.0.0", agent_type="chat", model="main", owner="team-cs"
    )
    state_schema = MessagesState

    def define(self):
        llm = init_chat_model("main")

        @klafi_node("agent")
        def agent(state):
            sys = SystemMessage("너는 친절한 FAQ 도우미다. 간결하게 한국어로 답하라.")
            return {"messages": [llm.invoke([sys, *state["messages"]])]}

        self.add_node("agent", agent)
        self.add_edge(START, "agent")
        self.add_edge("agent", END)
