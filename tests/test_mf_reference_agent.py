"""M-F 통합검증 — Reference Agent (요구사항 §35).

하나의 Supervisor Agent에 KLAFI의 전 기능을 결합해 Integration Test로 삼는다:
  Supervisor(라우팅) + Model Gateway(Alias/Token/Cost) + Guardrail(fail-close)
  + Checkpoint + HITL(승인/Resume) + Observability(단일 Trace) + Policy(Retry)
  + Evaluation(Version 비교) + API(invoke/resume).

아키텍처(§35):  Supervisor → research → report → Human Approval → Resume
"""

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from klafi import (
    BlocklistGuardrail,
    ExecutionPolicy,
    GuardrailHook,
    ModelGateway,
    RuleEvaluator,
    SupervisorAgent,
    run_offline,
    setup_tracing,
)
from klafi.core.context import ExecutionContext
from klafi.core.exceptions import GuardrailException
from klafi.hitl import pending_approvals, request_approval, resume_approval
from klafi.model import FunctionProvider
from klafi.server import AgentServer, create_app
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


def _gateway():
    gw = ModelGateway()
    gw.register(
        "quality-high",
        FunctionProvider(lambda p: f"작성됨: {p[:20]}"),
        policy=ExecutionPolicy(max_retries=2, backoff_base=0.0),  # Model Retry
        cost=(1.0, 3.0),
    )
    return gw


def build_reference_agent(version: str = "1.0.0") -> SupervisorAgent:
    """§35 Reference Agent — 전 기능 결합."""
    gw = _gateway()
    model = gw.model("quality-high")  # Alias만 노출

    plan = ["research", "report", FINISH]

    def router(state):
        return plan[len(state.get("history", []))]

    def research(state):
        return model(f"조사: {state['task']}")  # Model Gateway 사용 → Token/Cost 기록

    def report(state):
        decision = request_approval("publish", payload={"task": state["task"]}, approver="editor")  # HITL
        return "발행됨" if decision.approved else "보류"

    agent = SupervisorAgent(
        router=router,
        workers={"research": research, "report": report},
        checkpointer="memory",  # Checkpoint/Resume
        policy=ExecutionPolicy(timeout=30, max_retries=1),  # Agent 정책
        hooks=[GuardrailHook(input=[BlocklistGuardrail(["기밀유출"])])],  # Guardrail fail-close
    )
    agent.spec.version = version
    return agent


def _init(task="분기 리포트"):
    return {"task": task, "next": "", "results": {}, "history": []}


# ── 1. 전 기능 결합 실행 (Direct) ───────────────────────────────────────
def test_reference_agent_full_flow(exporter):
    agent = build_reference_agent()
    ctx = ExecutionContext.new(session_id="mf1")

    # 1st: research 실행 후 report의 승인에서 중단
    first = agent.invoke(_init(), context=ctx, thread_id="mf1")
    assert first["history"] == ["research"]
    assert pending_approvals(first)  # HITL 대기
    assert ctx.state == "WAITING_APPROVAL"

    # Observability: 단일 Trace에 agent/node/model span + token 기록
    spans = _spans(exporter)
    assert "agent.supervisor" in spans
    assert "worker.research" in spans
    assert "model.quality-high" in spans
    m = spans["model.quality-high"]
    assert m.attributes["klafi.tokens"] > 0  # Token 수집
    assert "klafi.cost_usd" in m.attributes  # Cost 수집

    # 승인 → Resume → 완주
    final = resume_approval(agent, "mf1", approved=True)
    assert final["history"] == ["research", "report"]
    assert final["results"]["report"] == "발행됨"


# ── 2. Guardrail fail-close ─────────────────────────────────────────────
def test_reference_guardrail_blocks_bad_input():
    agent = build_reference_agent()
    with pytest.raises(GuardrailException):
        agent.invoke(_init(task="기밀유출 문서 작성"), thread_id="bad1")


# ── 3. API로 서비스 + HITL Resume 엔드포인트 (API-01/06) ────────────────
def test_reference_agent_served_over_api():
    server = AgentServer()
    server.register(build_reference_agent(), agent_id="ref")
    client = TestClient(create_app(server))

    r1 = client.post("/agents/ref/invoke", json={"input": _init(), "thread_id": "api1"})
    assert r1.status_code == 200
    body = r1.json()
    assert body["state"] == "WAITING_APPROVAL"  # HITL 중단
    assert "__interrupt__" in body["result"]

    # Resume 엔드포인트로 승인 재개
    r2 = client.post("/agents/ref/resume", json={"thread_id": "api1", "decision": {"approved": True}})
    assert r2.status_code == 200
    assert r2.json()["result"]["results"]["report"] == "발행됨"


# ── 4. Evaluation + Version 비교 (§35 품질 검증) ────────────────────────
def test_reference_agent_evaluation_version_compare():
    # report 승인이 필요 없는 평가용 데이터셋 흐름: research 결과 품질만 평가
    def build_eval_agent(version, quality_word):
        gw = ModelGateway()
        gw.register("m", FunctionProvider(lambda p: quality_word))
        model = gw.model("m")

        def router(state):
            return "research" if not state.get("history") else FINISH

        return _versioned(
            SupervisorAgent(router=router, workers={"research": lambda s: model(s["task"])}),
            version,
        )

    dataset = [{"input": _init(task="x")}]
    ev = RuleEvaluator(lambda s: "좋음" in str(s.output["results"]), metric="quality")

    report = run_offline(build_eval_agent("1.0.0", "나쁨"), dataset, [ev])
    for r in run_offline(build_eval_agent("2.0.0", "좋음"), dataset, [ev]).results:
        report.add(r)

    cmp = report.compare_versions("quality")
    assert cmp["1.0.0"] == 0.0 and cmp["2.0.0"] == 1.0  # v2 품질 개선을 정량 비교


def _versioned(agent, version):
    agent.spec.version = version
    return agent
