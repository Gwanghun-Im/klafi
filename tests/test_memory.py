"""Long-Term Memory 검증 (요구사항 §10.2, F05 / MEM-11~17).

Checkpoint와 독립: 세션(Thread)을 넘어 지속되는 사용자/Agent 지식.
"""

import pytest
from langgraph.config import get_store
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from typing import TypedDict

from klafi import (
    AgentSpec,
    BaseGraph,
    MemoryStore,
    agent_scope,
    redact_pii,
    register_store,
    resolve_store,
    user_scope,
)
from klafi.core.exceptions import KlafiException


# ── MemoryStore 래퍼 (Scope/삭제/검색) ──────────────────────────────────
def test_scopes_and_crud():
    m = MemoryStore(InMemoryStore())
    m.remember(user_scope("u1"), "pref", {"lang": "ko"})  # MEM-12
    m.remember(agent_scope("a1"), "fact", {"note": "hello"})  # MEM-13

    assert m.recall(user_scope("u1"), "pref") == {"lang": "ko"}
    assert m.recall(user_scope("u1"), "missing") is None
    # scope 격리: user와 agent는 다른 namespace
    assert m.recall(user_scope("u1"), "fact") is None

    m.forget(user_scope("u1"), "pref")  # MEM-16
    assert m.recall(user_scope("u1"), "pref") is None


def test_search():
    m = MemoryStore(InMemoryStore())
    m.remember(user_scope("u1"), "a", {"v": 1})
    m.remember(user_scope("u1"), "b", {"v": 2})
    assert len(m.search(user_scope("u1"))) == 2


# ── PII 정책 (MEM-17) ───────────────────────────────────────────────────
def test_pii_redaction_on_store():
    m = MemoryStore(InMemoryStore(), pii_filter=redact_pii)
    m.remember(user_scope("u1"), "profile", {"email": "a@b.com", "memo": "카드 1234567812345678"})
    saved = m.recall(user_scope("u1"), "profile")
    assert "[REDACTED]" in saved["email"]
    assert "1234567812345678" not in saved["memo"]


# ── TTL (MEM-15): 메모리 백엔드는 미지원 → 명확한 에러(정직) ─────────────
def test_ttl_on_memory_backend_raises_clearly():
    m = MemoryStore(InMemoryStore())
    m.remember(user_scope("u1"), "k", {"v": 1})  # ttl=None은 정상
    with pytest.raises(NotImplementedError):
        m.remember(user_scope("u1"), "k2", {"v": 1}, ttl=60)  # TTL은 Postgres/Redis 백엔드 필요


# ── Store Adapter 해석 ──────────────────────────────────────────────────
def test_resolve_store_string_and_registry():
    assert isinstance(resolve_store("memory"), InMemoryStore)
    sentinel = InMemoryStore()
    register_store("mine", lambda cfg: sentinel)
    assert resolve_store("mine") is sentinel
    assert resolve_store(None) is None


def test_unknown_store_raises():
    with pytest.raises(KlafiException, match="알 수 없는"):
        resolve_store("nope")


def test_postgres_store_missing_conn_string():
    with pytest.raises(KlafiException, match="conn_string"):
        resolve_store({"type": "postgres"})


# ── BaseGraph Store 주입 + 세션 넘는 지속성 (핵심 DoD) ───────────────────
class S(TypedDict):
    text: str


def test_memory_persists_across_threads():
    """Checkpoint와 무관하게, Thread(세션)를 바꿔도 Long-Term Memory는 유지된다."""

    class A(BaseGraph):
        def build(self):
            g = StateGraph(S)

            def node(state):
                store = get_store()  # LangGraph Native 접근 (BaseGraph가 주입)
                ns = ("user", "u1")
                prev = store.get(ns, "seen")
                store.put(ns, "seen", {"count": (prev.value["count"] + 1) if prev else 1})
                return {"text": state["text"]}

            g.add_node("n", node)
            g.add_edge(START, "n")
            g.add_edge("n", END)
            return g

    agent = A(AgentSpec(id="mem", name="Mem"), store="memory", checkpointer="memory")
    assert agent.store is not None

    agent.invoke({"text": "1"}, thread_id="session-1")
    agent.invoke({"text": "2"}, thread_id="session-2")  # 다른 세션

    # 두 세션이 같은 user memory를 공유 → count=2
    assert agent.memory().recall(user_scope("u1"), "seen") == {"count": 2}


def test_no_store_config_means_none():
    from klafi import SimpleAgent

    a = SimpleAgent(model=lambda p: p)  # store 미지정
    assert a.store is None
    assert a.memory() is None
