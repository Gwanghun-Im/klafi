"""v0.1.5 결함 수정 검증 (3) — 서버·HITL (S1~S4). TestClient 로 실제 HTTP 경로를 탄다."""

from __future__ import annotations

import json
import logging
import operator
from typing import Annotated, TypedDict

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState
from langgraph.types import Command

from klafi import AgentSpec, KlafiGraph
from klafi.core import klafi_node
from klafi.guardrail import GuardrailResult, guardrail
from klafi.hitl import register_approval_adapter, request_approval
from klafi.server import AgentServer, create_app


class St(TypedDict):
    log: Annotated[list, operator.add]


@guardrail
def mask_secret(text: str) -> GuardrailResult:
    return GuardrailResult("SECRET" not in text, "secret", replacement=text.replace("SECRET", "***"))


class TwoApprovals(KlafiGraph):
    """병렬 브랜치 두 개가 각각 승인을 요청한다 → pending interrupt 2개."""

    spec = AgentSpec(id="two", name="Two")
    state_schema = St
    observability = False

    def define(self):
        @klafi_node("a")
        def a(state):
            d = request_approval("a-action", payload={"n": 1})
            return {"log": [f"a:{d.approved}"]}

        @klafi_node("b")
        def b(state):
            d = request_approval("b-action", payload={"n": 2})
            return {"log": [f"b:{d.approved}"]}

        self.add_node("a", a)
        self.add_node("b", b)
        self.add_edge(START, "a")
        self.add_edge(START, "b")
        self.add_edge("a", END)
        self.add_edge("b", END)


class Streamy(KlafiGraph):
    spec = AgentSpec(id="streamy", name="Streamy")
    state_schema = MessagesState
    observability = False

    def define(self):
        llm = FakeListChatModel(responses=["hello SECRET world"])

        @klafi_node("llm", after=[mask_secret])
        def node(state):
            get_stream_writer()({"progress": "thinking"})  # custom 모드로만 보이는 진행 신호
            return {"messages": [llm.invoke(state["messages"])]}

        self.add_node("llm", node)
        self.add_edge(START, "llm")
        self.add_edge("llm", END)


@pytest.fixture
def client():
    server = AgentServer()
    server.register(TwoApprovals(checkpointer="memory"))
    server.register(Streamy(checkpointer="memory"))
    return TestClient(create_app(server), raise_server_exceptions=False)


# ── S1: 승인 요청 부수효과는 재개 시 다시 나가지 않고, decided 는 같은 id ───────────
def test_s1_approval_side_effects_fire_once_across_resume(caplog):
    class One(KlafiGraph):
        spec = AgentSpec(id="one", name="One")
        state_schema = St
        observability = False

        def define(self):
            @klafi_node("pay")
            def pay(state):
                d = request_approval("pay", payload={"amt": 10})
                return {"log": [f"pay:{d.approved}"]}

            self.add_node("pay", pay)
            self.add_edge(START, "pay")
            self.add_edge("pay", END)

    pushed = []
    register_approval_adapter(lambda req: pushed.append(req.approval_id))
    try:
        ag = One(checkpointer="memory")
        with caplog.at_level(logging.INFO, logger="klafi.approval"):
            first = ag.invoke({"log": []}, thread_id="s1-thread")
            assert first["__interrupt__"]
            out = ag.invoke(Command(resume={"approved": True}), thread_id="s1-thread")
    finally:
        register_approval_adapter(None)
    assert out["log"] == ["pay:True"]
    assert len(pushed) == 1  # 이전엔 재개 시 노드 재실행으로 어댑터 push 2회(서로 다른 id)
    requested = [r.getMessage() for r in caplog.records if r.getMessage().startswith("approval.requested")]
    decided = [r.getMessage() for r in caplog.records if r.getMessage().startswith("approval.decided")]
    assert len(requested) == 1 and len(decided) == 1 and pushed[0] in decided[0]


# ── S2: __interrupt__ 에 id 가 있고, id-키 resume 으로 다중 interrupt 재개 ─────────
def test_s2_multiple_interrupts_resume_by_id_over_http(client):
    r = client.post("/agents/two/invoke", json={"input": {"log": []}, "thread_id": "s2"})
    assert r.status_code == 200
    pend = r.json()["result"]["__interrupt__"]
    assert len(pend) == 2 and all(p["id"] for p in pend)  # 이전엔 id 가 없어 재개 불가(500)
    decision = {p["id"]: {"approved": p["value"]["action"] == "a-action"} for p in pend}
    r2 = client.post("/agents/two/resume", json={"thread_id": "s2", "decision": decision})
    assert r2.status_code == 200, r2.text
    assert sorted(r2.json()["result"]["log"]) == ["a:True", "b:False"]


# ── S3: stream_mode 선택(custom 도달) + 토큰은 after 가드레일 적용 후 본문 ──────────
def test_s3_stream_mode_custom_and_masked_tokens(client):
    body = {"input": {"messages": [{"role": "user", "content": "q"}]}, "thread_id": "s3",
            "stream_mode": ["updates", "messages", "custom"]}
    with client.stream("POST", "/agents/streamy/stream", json=body) as r:
        lines = [json.loads(l) for l in r.iter_lines() if l.strip()]
    tokens = "".join(l["token"] for l in lines if "token" in l)
    assert tokens == "hello *** world"  # 이전엔 원문 'hello SECRET world' 가 토큰으로 유출
    assert any(l.get("custom") == {"progress": "thinking"} for l in lines)  # 이전엔 custom 도달 불가


# ── S4: 대기 중 interrupt 없는 thread 의 resume 은 409, decision 누락은 422 ─────────
def test_s4_resume_without_pending_interrupt_is_409(client):
    r = client.post("/agents/two/resume", json={"thread_id": "no-such-thread", "decision": {"approved": True}})
    assert r.status_code == 409 and r.json()["error_code"] == "NO_PENDING_INTERRUPT"  # 이전엔 처음부터 새로 실행
    assert client.post("/agents/two/resume", json={"thread_id": "x"}).status_code == 422  # decision 필수


# ── 스레드 상태 조회 — 재접속 클라이언트가 승인 카드를 복원한다 ──────────────────────
def test_thread_state_exposes_pending_interrupts_with_ids(client):
    client.post("/agents/two/invoke", json={"input": {"log": []}, "thread_id": "ts1"})
    r = client.get("/agents/two/threads/ts1")
    body = r.json()
    assert r.status_code == 200 and body["state"] == "WAITING_APPROVAL"
    assert len(body["interrupts"]) == 2 and all(p["id"] and p["value"]["action"] for p in body["interrupts"])
    empty = client.get("/agents/two/threads/never-used").json()
    assert empty["state"] == "EMPTY" and empty["interrupts"] == []
