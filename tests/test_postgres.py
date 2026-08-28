"""PostgreSQL Checkpoint/Store 운영 배선 검증 (MEM-03).

실 DB가 필요하므로 접속 불가 시 skip. 기본 접속 정보는 KLAFI_TEST_PG 환경변수로 덮어쓴다.
"""

import os
import socket
from typing import TypedDict
from urllib.parse import urlparse

import pytest
from langgraph.graph import END, START

from klafi import AgentSpec, KlafiGraph, klafi_node, user_scope

CONN = os.environ.get("KLAFI_TEST_PG", "postgresql://klafi:klafi@127.0.0.1:5433/klafi")
PG = {"type": "postgres", "conn_string": CONN}


def _reachable() -> bool:
    u = urlparse(CONN)
    try:
        with socket.create_connection((u.hostname, u.port or 5432), timeout=1):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason=f"PostgreSQL 접속 불가: {CONN}")


class State(TypedDict):
    a: str
    b: str


def _agent(fail: bool, thread_suffix: str):
    class Agent(KlafiGraph):
        spec = AgentSpec(id=f"pg-{thread_suffix}", name="PG")
        state_schema = State
        observability = False

        def define(self):
            @klafi_node("a")
            def node_a(s):
                return {"a": "A완료"}

            @klafi_node("b")
            def node_b(s):
                if fail:
                    raise ValueError("B 실패")
                return {"b": "B완료"}

            self.add_node("a", node_a)
            self.add_node("b", node_b)
            self.add_edge(START, "a")
            self.add_edge("a", "b")
            self.add_edge("b", END)

    return Agent(checkpointer=PG, store=PG)


def test_checkpoint_resume_across_instances():
    """별도 Agent 인스턴스(=재시작)에서도 체크포인트로 Resume."""
    tid = "pytest-resume"
    first = _agent(fail=True, thread_suffix="r")
    with pytest.raises(ValueError):
        first.invoke({"a": "", "b": ""}, thread_id=tid)
    assert first.get_state(thread_id=tid).values["a"] == "A완료"

    # 새 인스턴스(새 pool)에서 이어받아 Resume
    second = _agent(fail=False, thread_suffix="r")
    assert second.get_state(thread_id=tid).values == {"a": "A완료", "b": ""}
    assert second.invoke(None, thread_id=tid) == {"a": "A완료", "b": "B완료"}


def test_long_term_memory_persists():
    agent = _agent(fail=False, thread_suffix="m")
    agent.memory().remember(user_scope("pytest-u1"), "pref", {"lang": "ko"})

    other = _agent(fail=False, thread_suffix="m")  # 새 인스턴스
    assert other.memory().recall(user_scope("pytest-u1"), "pref") == {"lang": "ko"}
    other.memory().forget(user_scope("pytest-u1"), "pref")
    assert other.memory().recall(user_scope("pytest-u1"), "pref") is None


def test_pool_settings_applied():
    from klafi.context.checkpoint import resolve_checkpointer

    saver = resolve_checkpointer({**PG, "min_size": 2, "max_size": 4})
    pool = saver.conn
    assert pool.min_size == 2 and pool.max_size == 4  # Connection Pool 재사용(NFR)


def test_missing_conn_string_raises():
    from klafi.context.checkpoint import resolve_checkpointer
    from klafi.core.exceptions import CheckpointException

    with pytest.raises(CheckpointException, match="conn_string"):
        resolve_checkpointer({"type": "postgres"})
