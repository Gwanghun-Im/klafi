"""채팅 호환 에이전트 — MessagesState + 단일 LLM 노드 (툴 없음).

템플릿 SimpleAgent는 {question}→{answer} 스키마라 채팅 UI(messages)와 맞지 않는다.
채팅 프론트에서 바로 쓰려면 이렇게 state_schema=MessagesState 로 define() 한다 —
가장 단순한 대화 에이전트(툴/HITL 없음). 툴이 필요하면 support_agent 처럼 bind_tools 를 얹으면 된다.

구조: spec → agentSpec.py · state → state.py · 그래프 → 이 파일 (에이전트별 패키지).
"""

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START

from klafi.core import KlafiGraph, klafi_node
from klafi.model import init_chat_model

from .agentSpec import spec
from .prompt import SYSTEM
from .state import MessagesState


class FaqAgent(KlafiGraph):
    spec = spec
    state_schema = MessagesState

    def define(self):
        llm = init_chat_model("main")

        @klafi_node("agent")
        def agent(state):
            sys = SystemMessage(SYSTEM)
            return {"messages": [llm.invoke([sys, *state["messages"]])]}

        self.add_node("agent", agent)
        self.add_edge(START, "agent")
        self.add_edge("agent", END)
