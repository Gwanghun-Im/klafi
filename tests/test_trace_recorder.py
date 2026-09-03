"""ExecutionRecorder — 실행 타임라인(노드·모델·툴·가드레일·승인) 기록과 GET /executions/{id}."""

from __future__ import annotations

from typing import TypedDict

from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState
from langgraph.types import Command

from klafi import AgentSpec, KlafiGraph
from klafi.core import klafi_node
from klafi.guardrail import GuardrailResult, guardrail
from klafi.hitl import request_approval
from klafi.model.gateway import FunctionProvider, ModelGateway
from klafi.server import AgentServer, create_app
from klafi.tool.tool import tool


@guardrail
def mask_secret(text: str) -> GuardrailResult:
    return GuardrailResult("SECRET" not in text, "secret", replacement=text.replace("SECRET", "***"))


@tool("lookup")
def lookup(q: str) -> str:
    return f"found {q}"


class St(TypedDict, total=False):
    messages: list
    log: list


class Traced(KlafiGraph):
    spec = AgentSpec(id="traced", name="Traced")
    state_schema = MessagesState
    observability = False

    def define(self):
        gw = ModelGateway()
        gw.register("m", FunctionProvider(lambda p: "answer SECRET"), cost=(1.0, 2.0))
        llm = gw.model("m")

        @klafi_node("think", after=[mask_secret])
        def think(state):
            lookup.run(q="x")
            return {"messages": [HumanMessage(llm(state["messages"][-1].content))]}

        self.add_node("think", think)
        self.add_edge(START, "think")
        self.add_edge("think", END)


class Approving(KlafiGraph):
    spec = AgentSpec(id="approving", name="Approving")
    state_schema = MessagesState
    observability = False

    def define(self):
        @klafi_node("gate")
        def gate(state):
            d = request_approval("pay", payload={"amt": 1})
            return {"messages": [HumanMessage(f"approved={d.approved}")]}

        self.add_node("gate", gate)
        self.add_edge(START, "gate")
        self.add_edge("gate", END)


def _client():
    server = AgentServer()
    server.register(Traced())
    server.register(Approving(checkpointer="memory"))
    return TestClient(create_app(server), raise_server_exceptions=False), server


def test_trace_has_node_model_tool_and_guardrail_rows():
    client, server = _client()
    r = client.post("/agents/traced/invoke", json={"input": {"messages": [{"role": "user", "content": "hello world q"}]}})
    eid = r.json()["execution_id"]
    t = client.get(f"/agents/traced/executions/{eid}").json()
    kinds = [(e["kind"], e["name"]) for e in t["events"]]
    assert ("tool", "lookup") in kinds and ("model", "m") in kinds and ("guardrail", "mask_secret") in kinds
    assert kinds[-1] == ("node", "think")  # 노드는 finally 에 완료 행으로
    model = next(e for e in t["events"] if e["kind"] == "model")
    assert model["duration_ms"] is not None and model["tokens"] > 0 and model["cost_usd"] > 0  # 토큰·비용 부착
    g = next(e for e in t["events"] if e["kind"] == "guardrail")
    assert g["stage"] == "output" and g["severity"] == "mask"
    assert t["totals"]["violations"] == 1 and t["totals"]["tokens"] == model["tokens"] and t["duration_ms"] is not None
    assert t["state"] == "COMPLETED" and t["agent_id"] == "traced"
    assert client.get("/agents/traced/executions/nope").status_code == 404
    assert client.get(f"/agents/approving/executions/{eid}").status_code == 404  # 다른 agent 의 실행은 숨김


def test_trace_records_approval_request_and_decision():
    client, _ = _client()
    r = client.post("/agents/approving/invoke", json={"input": {"messages": [{"role": "user", "content": "go"}]}, "thread_id": "tr1"})
    eid1 = r.json()["execution_id"]
    t1 = client.get(f"/agents/approving/executions/{eid1}").json()
    assert [e["status"] for e in t1["events"] if e["kind"] == "approval"] == ["requested"]
    assert t1["state"] == "WAITING_APPROVAL"
    r2 = client.post("/agents/approving/resume", json={"thread_id": "tr1", "decision": {"approved": True}})
    t2 = client.get(f"/agents/approving/executions/{r2.json()['execution_id']}").json()
    assert [e["status"] for e in t2["events"] if e["kind"] == "approval"] == ["approved"]  # 재개 실행엔 결정만(요청 중복 없음)
