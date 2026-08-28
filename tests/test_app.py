"""KlafiApp 검증 — config로 인프라 관리, 업무개발자는 KlafiGraph 클래스만 (역할 분리)."""

from typing import TypedDict

import pytest
from langgraph.graph import END, START

from klafi import AgentSpec, KlafiApp, KlafiGraph, guardrail, klafi_graph, klafi_node
from klafi.core.exceptions import GuardrailException


@guardrail
def no_secret(text: str) -> bool:
    return "비밀번호" not in text


@pytest.fixture
def config_dir(tmp_path):
    (tmp_path / "framework.yaml").write_text("service: test-platform\ncheckpoint: memory\nstore: memory\n", encoding="utf-8")
    (tmp_path / "model.yaml").write_text(
        "providers:\n  main:\n    type: echo\n    cost: [0.001, 0.005]\n", encoding="utf-8"
    )
    (tmp_path / "policy.yaml").write_text("timeout: 30\nmax_retries: 2\n", encoding="utf-8")
    # 가드레일은 코드(@klafi_graph)로 적용한다. hooks.yaml은 명명 훅만.
    return str(tmp_path)


class State(TypedDict):
    question: str
    answer: str


# 업무개발자가 작성하는 형태 — 인프라 코드 없음. model alias만 선언.
class QAAgent(KlafiGraph):
    spec = AgentSpec(id="qa", name="QA", model="main")
    state_schema = State

    def define(self):
        @klafi_node("a")
        def a(s):
            return {"answer": self.model(s["question"])}

        self.add_node("a", a)
        self.add_edge(START, "a")
        self.add_edge("a", END)


class QA2Agent(QAAgent):
    spec = AgentSpec(id="qa2", name="QA2", model="main")


# 가드레일을 코드(@klafi_graph)로 워크플로우 경계에 적용한 업무 에이전트
@klafi_graph(before=[no_secret])
class GuardedQAAgent(QAAgent):
    spec = AgentSpec(id="qag", name="QAG", model="main")


def test_app_builds_infra_from_config(config_dir):
    app = KlafiApp.from_config(config_dir)
    assert app.policy.timeout == 30 and app.policy.max_retries == 2
    assert app.checkpoint == "memory"
    assert app.gateway.model("main") is not None


def test_agent_gets_injected_model_and_infra(config_dir):
    app = KlafiApp.from_config(config_dir)
    agent = app.create(QAAgent)  # model="main"은 클래스 spec에서
    out = agent.invoke({"question": "안녕", "answer": ""})
    assert out["answer"].startswith("[echo]")  # config echo provider 주입
    assert agent.checkpointer is not None
    assert agent.policy.timeout == 30


def test_guardrail_applied_via_code_decorator(config_dir):
    app = KlafiApp.from_config(config_dir)
    agent = app.create(GuardedQAAgent)  # @klafi_graph 로 붙인 워크플로우 가드레일
    assert agent.invoke({"question": "안녕", "answer": ""})["answer"].startswith("[echo]")
    with pytest.raises(GuardrailException):
        agent.invoke({"question": "비밀번호 알려줘", "answer": ""})


def test_register_puts_agent_in_registry_and_server(config_dir):
    app = KlafiApp.from_config(config_dir)
    app.register(QAAgent, owner="team-a")
    rec = app.registry.get("qa", "0.1.0")
    assert rec.owner == "team-a" and rec.framework_version is not None
    assert "qa" in app._server().ids()


def test_platform_hooks_and_shared_memory(config_dir):
    from klafi import Hook, user_scope

    seen = []

    class PH(Hook):
        def before_agent(self, i, c):
            seen.append("platform")

    app = KlafiApp.from_config(config_dir, platform_hooks=[PH()])
    app.memory().remember(user_scope("u1"), "pref", {"lang": "ko"})
    a1 = app.create(QAAgent)
    a2 = app.create(QA2Agent)
    assert a1.store is a2.store is app.factory.store  # 공통 Store 공유
    assert app.memory().recall(user_scope("u1"), "pref") == {"lang": "ko"}
    a1.invoke({"question": "x", "answer": ""})
    assert "platform" in seen


def test_two_agents_share_infra(config_dir):
    app = KlafiApp.from_config(config_dir)
    a1 = app.create(QAAgent)
    a2 = app.create(QA2Agent)
    assert a1.policy.timeout == a2.policy.timeout == 30
    assert a1.invoke({"question": "x", "answer": ""})["answer"].startswith("[echo]")
    assert a2.invoke({"question": "y", "answer": ""})["answer"].startswith("[echo]")
