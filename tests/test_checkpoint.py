"""WS3 Checkpoint/Resume 검증 (요구사항 §10.1, F05 / MEM-01~07).

DoD: 실행 중단 후 동일 Thread를 Resume할 수 있을 것.
+ Config 기반 Checkpointer 주입, Thread 격리, Adapter Registry.
"""

import pytest
from langgraph.graph import END, START, StateGraph
from typing import TypedDict

from klafi import AgentSpec, BaseGraph, register_checkpointer, resolve_checkpointer
from klafi.core.exceptions import CheckpointException


class State(TypedDict):
    a: str
    b: str


def _resumable_agent(fail_flag, a_runs, checkpointer=None, config=None):
    class A(BaseGraph):
        def build(self):
            g = StateGraph(State)

            def node_a(s):
                a_runs.append(1)
                return {"a": "A"}

            def node_b(s):
                if fail_flag["on"]:
                    raise ValueError("boom in B")
                return {"b": "B"}

            g.add_node("a", node_a)
            g.add_node("b", node_b)
            g.add_edge(START, "a")
            g.add_edge("a", "b")
            g.add_edge("b", END)
            return g

    return A(AgentSpec(id="a", name="A", config=config or {}), checkpointer=checkpointer)


# ── DoD: 중단 후 Resume ─────────────────────────────────────────────────
def test_resume_after_failure_does_not_rerun_a():
    from langgraph.checkpoint.memory import InMemorySaver

    fail = {"on": True}
    a_runs: list = []
    agent = _resumable_agent(fail, a_runs, checkpointer=InMemorySaver())

    with pytest.raises(ValueError):
        agent.invoke({"a": "", "b": ""}, thread_id="t1")
    assert a_runs == [1]  # A 1회 실행 후 B에서 실패

    # 체크포인트 조회 (MEM-07): A 결과 저장, 다음 대기 노드 = b
    snap = agent.get_state(thread_id="t1")
    assert snap.values["a"] == "A"
    assert snap.next == ("b",)

    # B 고치고 동일 Thread Resume: A는 재실행되지 않고 B만 재개
    fail["on"] = False
    out = agent.invoke(None, thread_id="t1")
    assert out == {"a": "A", "b": "B"}
    assert a_runs == [1]  # A 재실행 없음 → Resume 성공


def test_threads_are_isolated():
    from langgraph.checkpoint.memory import InMemorySaver

    fail = {"on": True}
    a_runs: list = []
    agent = _resumable_agent(fail, a_runs, checkpointer=InMemorySaver())
    with pytest.raises(ValueError):
        agent.invoke({"a": "", "b": ""}, thread_id="t1")
    # 다른 Thread는 t1 체크포인트 영향 없음
    snap = agent.get_state(thread_id="t2")
    assert snap.next == ()  # t2는 빈 상태


# ── Config 기반 주입 (FAC-02 / DoD) ─────────────────────────────────────
def test_checkpointer_from_config_string():
    fail = {"on": True}
    a_runs: list = []
    # 코드로 saver를 넘기지 않고 Config로 "memory" 지정
    agent = _resumable_agent(fail, a_runs, config={"checkpoint": "memory"})
    assert agent.checkpointer is not None

    with pytest.raises(ValueError):
        agent.invoke({"a": "", "b": ""}, thread_id="c1")
    fail["on"] = False
    out = agent.invoke(None, thread_id="c1")
    assert out == {"a": "A", "b": "B"}
    assert a_runs == [1]


def test_no_checkpoint_config_means_none():
    agent = _resumable_agent({"on": False}, [])
    assert agent.checkpointer is None


# ── Adapter Registry ────────────────────────────────────────────────────
def test_register_custom_checkpointer():
    from langgraph.checkpoint.memory import InMemorySaver

    sentinel = InMemorySaver()
    register_checkpointer("mycustom", lambda cfg: sentinel)
    assert resolve_checkpointer("mycustom") is sentinel
    assert resolve_checkpointer({"type": "mycustom"}) is sentinel


def test_unknown_checkpointer_raises():
    with pytest.raises(CheckpointException, match="알 수 없는"):
        resolve_checkpointer("nope-db")


def test_postgres_missing_conn_string_gives_clear_error():
    # conn_string 누락 → 명확한 에러 (조용한 실패 금지). 실 DB 연동은 test_postgres.py 참고
    with pytest.raises(CheckpointException, match="conn_string"):
        resolve_checkpointer({"type": "postgres"})


def test_passing_saver_instance_is_passthrough():
    from langgraph.checkpoint.memory import InMemorySaver

    s = InMemorySaver()
    assert resolve_checkpointer(s) is s
