"""WS-HITL + Supervisor 검증 (요구사항 §13 F08, §20 T03, §35 Reference Agent).

- HITL: interrupt 승인 요청 → 승인/반려 재개, audit log, error 오발화 없음.
- Supervisor: 여러 Worker 라우팅 후 FINISH.
- Reference Agent: Supervisor + Human Approval 통합 (M-F 축소판).
"""

import logging

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from typing import TypedDict

from klafi import SupervisorAgent, klafi_node, setup_tracing
from klafi.hitl import pending_approvals, request_approval, resume_approval
from klafi.templates.supervisor import FINISH


@pytest.fixture(scope="session")
def exporter():
    exp = InMemorySpanExporter()
    setup_tracing(exporter=exp, simple=True)
    return exp


@pytest.fixture(autouse=True)
def _clear(exporter):
    exporter.clear()
    yield


def _spans(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


# ── Supervisor ──────────────────────────────────────────────────────────
def test_supervisor_routes_workers_then_finishes():
    order = ["research", "analysis", "report", FINISH]

    def router(state):
        return order[len(state.get("history", []))]

    workers = {
        "research": lambda s: "found",
        "analysis": lambda s: "analyzed",
        "report": lambda s: "reported",
    }
    agent = SupervisorAgent(router=router, workers=workers)
    out = agent.invoke({"task": "do", "next": "", "results": {}, "history": []})
    assert out["history"] == ["research", "analysis", "report"]
    assert out["results"] == {"research": "found", "analysis": "analyzed", "report": "reported"}


def test_supervisor_max_steps_guard():
    # 항상 같은 worker로 보내는 라우터 → max_steps에서 강제 종료
    agent = SupervisorAgent(
        router=lambda s: "loop", workers={"loop": lambda s: "x"}, max_steps=3
    )
    out = agent.invoke({"task": "t", "next": "", "results": {}, "history": []})
    assert len(out["history"]) == 3  # 무한루프 방지


def test_supervisor_reserved_finish_name_rejected():
    with pytest.raises(ValueError):
        SupervisorAgent(router=lambda s: FINISH, workers={FINISH: lambda s: 1})


# ── HITL ────────────────────────────────────────────────────────────────
class HState(TypedDict):
    amount: int
    decision: str


def _approval_agent(exporter=None):
    from langgraph.graph import END, START

    from klafi import KlafiGraph
    from klafi.core.spec import AgentSpec

    class Approver(KlafiGraph):
        spec = AgentSpec(id="approver", name="Approver", agent_type="hitl")
        state_schema = HState

        def define(self):
            @klafi_node("gate")
            def gate(state):
                d = request_approval("transfer", payload={"amount": state["amount"]}, approver="manager")
                return {"decision": "approved" if d.approved else "rejected"}

            self.add_node("gate", gate)
            self.add_edge(START, "gate")
            self.add_edge("gate", END)

    return Approver(checkpointer="memory")


def test_hitl_interrupt_then_approve(caplog):
    agent = _approval_agent()
    with caplog.at_level(logging.INFO, logger="klafi.approval"):
        first = agent.invoke({"amount": 100, "decision": ""}, thread_id="a1")
    # 첫 실행은 중단되어 승인요청이 대기
    reqs = pending_approvals(first)
    assert reqs and reqs[0].value["action"] == "transfer"
    assert reqs[0].value["approver"] == "manager"
    # audit: 요청 기록됨 (HIT-09)
    assert any("approval.requested" in r.message for r in caplog.records)

    # 승인으로 재개
    out = resume_approval(agent, "a1", approved=True, decided_by="boss")
    assert out["decision"] == "approved"


def test_hitl_reject():
    agent = _approval_agent()
    agent.invoke({"amount": 100, "decision": ""}, thread_id="r1")
    out = resume_approval(agent, "r1", approved=False, comment="한도초과")
    assert out["decision"] == "rejected"


def test_hitl_state_is_waiting_approval():
    from klafi.core.context import ExecutionContext

    agent = _approval_agent()
    ctx = ExecutionContext.new(session_id="w1")
    agent.invoke({"amount": 1, "decision": ""}, context=ctx, thread_id="w1")
    assert ctx.state == "WAITING_APPROVAL"


def test_interrupt_does_not_mark_span_error(exporter):
    agent = _approval_agent(exporter)
    agent.invoke({"amount": 1, "decision": ""}, thread_id="s1")
    spans = _spans(exporter)
    # gate 노드/agent span이 interrupt로 ERROR 처리되면 안 된다
    assert spans["node.gate"].status.status_code != StatusCode.ERROR
    assert spans["agent.approver"].status.status_code != StatusCode.ERROR


# ── Reference Agent: Supervisor + Human Approval (§35, M-F 축소판) ────────
def test_reference_supervisor_with_approval():
    """report worker 실행 전 사람 승인을 받는 통합 시나리오."""
    order = ["research", "report", FINISH]

    def router(state):
        return order[len(state.get("history", []))]

    def report_worker(state):
        d = request_approval("publish_report", payload={"draft": "..."}, approver="editor")
        return "published" if d.approved else "held"

    workers = {"research": lambda s: "data", "report": report_worker}
    agent = SupervisorAgent(router=router, workers=workers, checkpointer="memory")

    first = agent.invoke(
        {"task": "make report", "next": "", "results": {}, "history": []}, thread_id="ref1"
    )
    # research는 끝나고 report의 승인에서 중단
    assert pending_approvals(first)
    assert first["history"] == ["research"]

    out = resume_approval(agent, "ref1", approved=True)
    assert out["results"]["report"] == "published"
    assert out["history"] == ["research", "report"]
