"""Event Framework 검증 (요구사항 §24)."""

import pytest
from langgraph.graph import END, START, StateGraph
from typing import TypedDict

from klafi import (
    klafi_node,
    AgentSpec,
    EventHook,
    EventType,
    ModelGateway,
    SimpleAgent,
    Tool,
    subscribe,
)
from klafi.events import EVENTS
from klafi.hitl import request_approval, resume_approval
from klafi.model import FunctionProvider


@pytest.fixture(autouse=True)
def _clean_bus():
    EVENTS.clear()
    yield
    EVENTS.clear()


def _collector():
    seen = []
    subscribe(lambda e: seen.append(e.type))
    return seen


# ── Agent/Node 생명주기 (EventHook) ─────────────────────────────────────
def test_agent_and_node_lifecycle_events():
    seen = _collector()
    SimpleAgent(model=lambda p: p, hooks=[EventHook()]).invoke({"question": "hi"})
    assert EventType.ExecutionStarted in seen
    assert EventType.AgentStarted in seen
    assert EventType.NodeStarted in seen
    assert EventType.NodeCompleted in seen
    assert EventType.AgentCompleted in seen
    assert EventType.ExecutionCompleted in seen


def test_node_failed_event():
    def boom(s):
        raise ValueError("x")

    class A(SimpleAgent):
        def build(self):
            from klafi.templates.simple import SimpleState

            g = StateGraph(SimpleState)
            g.add_node("llm", boom)
            g.add_edge(START, "llm")
            g.add_edge("llm", END)
            return g

    seen = _collector()
    with pytest.raises(ValueError):
        A(model=lambda p: p, hooks=[EventHook()]).invoke({"question": "x"})
    assert EventType.NodeFailed in seen
    assert EventType.AgentFailed in seen


# ── 구독 필터 ───────────────────────────────────────────────────────────
def test_subscribe_with_type_filter():
    only_nodes = []
    subscribe(lambda e: only_nodes.append(e.type), types=[EventType.NodeStarted])
    SimpleAgent(model=lambda p: p, hooks=[EventHook()]).invoke({"question": "x"})
    assert set(only_nodes) == {EventType.NodeStarted}


# ── 구독자 예외 격리 (fail-open) ────────────────────────────────────────
def test_subscriber_error_isolated():
    subscribe(lambda e: 1 / 0)  # 항상 터지는 구독자
    good = []
    subscribe(lambda e: good.append(e.type))
    # 구독자 예외가 Agent를 막지 않음
    out = SimpleAgent(model=lambda p: p, hooks=[EventHook()]).invoke({"question": "ok"})
    assert out["answer"] == "ok"
    assert good  # 정상 구독자는 계속 받음


# ── Tool / Model / Approval Event (leaf) ────────────────────────────────
def test_tool_events():
    seen = _collector()
    Tool(lambda: "r", name="t1")()
    assert EventType.ToolStarted in seen and EventType.ToolCompleted in seen


def test_model_called_event():
    seen = []
    subscribe(lambda e: seen.append((e.type, e.data)), types=[EventType.ModelCalled])
    gw = ModelGateway()
    gw.register("m", FunctionProvider(lambda p: "hello world"))
    gw.model("m")("hi")
    assert seen[0][0] == EventType.ModelCalled
    assert seen[0][1]["model"] == "m" and seen[0][1]["tokens"] == 3  # prompt 1 + completion 2


def test_approval_events():
    from klafi import KlafiGraph

    class HState(TypedDict):
        x: str
        d: str

    class Approver(KlafiGraph):
        spec = AgentSpec(id="ap", name="Ap")
        state_schema = HState

        def define(self):
            @klafi_node("gate")
            def gate(s):
                dec = request_approval("act")
                return {"d": "y" if dec.approved else "n"}

            self.add_node("gate", gate)
            self.add_edge(START, "gate")
            self.add_edge("gate", END)

    seen = _collector()
    agent = Approver(checkpointer="memory")
    agent.invoke({"x": "1", "d": ""}, thread_id="t1")
    assert EventType.ApprovalRequested in seen
    resume_approval(agent, "t1", approved=True)
    assert EventType.ApprovalCompleted in seen
