"""P0 Foundation 게이트 검증: 순수 LangGraph Agent가 BaseGraph로 실행되고,
Node 내부에서 ExecutionContext(execution_id)가 자동으로 보인다.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from klafi import AgentSpec, BaseGraph, ExecutionContext, get_context


class State(TypedDict):
    text: str
    seen_execution_id: str


class EchoAgent(BaseGraph):
    def build(self) -> StateGraph:
        g = StateGraph(State)

        def node(state: State) -> State:
            ctx = get_context()  # 개발자가 Context를 직접 전달하지 않아도 보인다
            assert ctx is not None, "Node 안에서 ExecutionContext가 바인딩돼야 함"
            return {"text": state["text"] + "!", "seen_execution_id": ctx.execution_id}

        g.add_node("echo", node)
        g.add_edge(START, "echo")
        g.add_edge("echo", END)
        return g


def _agent() -> EchoAgent:
    return EchoAgent(AgentSpec(id="echo", name="Echo", version="1.0.0", project="demo"))


def test_invoke_sets_execution_id():
    out = _agent().invoke({"text": "hi"})
    assert out["text"] == "hi!"
    assert len(out["seen_execution_id"]) == 32  # uuid4 hex 자동 발급됨


def test_explicit_context_is_used():
    ctx = ExecutionContext.new(agent_id="echo", user_id="u1")
    out = _agent().invoke({"text": "x"}, context=ctx)
    assert out["seen_execution_id"] == ctx.execution_id


def test_stream_keeps_context():
    chunks = list(_agent().stream({"text": "go"}))
    assert chunks, "stream이 최소 1개 이벤트를 내야 함"
    assert chunks[-1]["echo"]["text"] == "go!"


def test_context_not_leaked_after_run():
    _agent().invoke({"text": "hi"})
    assert get_context() is None  # 실행 종료 후 Scope 정리


async def test_ainvoke():
    out = await _agent().ainvoke({"text": "async"})
    assert out["text"] == "async!"
    assert len(out["seen_execution_id"]) == 32
