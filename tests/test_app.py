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


# ── register_package: app/agents/* 자동 등록 (convention) ──────────────────
_AGENT_SRC = '''
from typing import TypedDict
from langgraph.graph import START, END
from klafi import AgentSpec, KlafiGraph, klafi_node

class _S(TypedDict):
    question: str
    answer: str

class {cls}(KlafiGraph):
    spec = AgentSpec(id="{aid}", name="{cls}", model="main", owner="{owner}")
    state_schema = _S
    def define(self):
        @klafi_node("a")
        def a(s): return {{"answer": self.model(s["question"])}}
        self.add_node("a", a); self.add_edge(START, "a"); self.add_edge("a", END)
'''


def test_register_package_discovers_and_skips_underscore(config_dir, tmp_path, monkeypatch):
    """app.register_package 는 서브패키지의 KlafiGraph 를 자동 등록하되 `_` 폴더는 건너뛰고,
    owner 는 spec.owner 로 폴백한다."""
    pkg = tmp_path / "agpkg"
    (pkg).mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "good").mkdir()
    (pkg / "good" / "__init__.py").write_text(
        _AGENT_SRC.format(cls="GoodAgent", aid="good", owner="team-x"), encoding="utf-8"
    )
    (pkg / "_wip").mkdir()  # 밑줄 → 스킵돼야 함
    (pkg / "_wip" / "__init__.py").write_text(
        _AGENT_SRC.format(cls="WipAgent", aid="wip", owner="team-x"), encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    for m in [k for k in list(__import__("sys").modules) if k == "agpkg" or k.startswith("agpkg.")]:
        del __import__("sys").modules[m]  # 세션 오염 방지

    app = KlafiApp.from_config(config_dir)
    registered = app.register_package("agpkg")

    assert {a.spec.id for a in registered} == {"good"}  # _wip 는 자동 등록 제외
    assert app.registry.get("good", "0.1.0").owner == "team-x"  # spec.owner 폴백


def test_register_reads_colocated_config_yaml(config_dir, tmp_path, monkeypatch):
    """에이전트 폴더의 config.yaml 이 있으면 policy 를 전역 위에 머지해 그 에이전트에만 적용하고,
    명시된 concurrency 는 app._agent_concurrency 에 기록된다."""
    import sys

    pkg = tmp_path / "agpkg2"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "good").mkdir()
    (pkg / "good" / "__init__.py").write_text(
        _AGENT_SRC.format(cls="CfgAgent", aid="cfg", owner="team-x"), encoding="utf-8"
    )
    (pkg / "good" / "config.yaml").write_text("policy:\n  timeout: 99\n  concurrency: 2\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for m in [k for k in list(sys.modules) if k == "agpkg2" or k.startswith("agpkg2.")]:
        del sys.modules[m]

    app = KlafiApp.from_config(config_dir)  # 전역 policy.yaml: timeout 30 · max_retries 2
    [agent] = app.register_package("agpkg2")

    assert agent.policy.timeout == 99  # per-agent override
    assert agent.policy.max_retries == 2  # 전역 상속
    assert agent.policy.concurrency == 2
    assert app._agent_concurrency == {"cfg": 2}  # 명시된 것만 미들웨어 맵에


def test_spec_print_draws_graph_on_register(config_dir, capsys):
    """spec.print=True 면 등록(부팅) 시 컴파일된 그래프를 stdout 에 그린다. 기본(False)은 안 그린다."""

    class DrawAgent(QAAgent):
        spec = AgentSpec(id="draw", name="Draw", model="main", print=True)

    app = KlafiApp.from_config(config_dir)
    app.register(QAAgent)   # print=False(기본) → 안 그림
    app.register(DrawAgent)  # print=True → 그림
    out = capsys.readouterr().out

    assert "agent graph: draw" in out
    assert "agent graph: qa" not in out  # 기본은 그리지 않음
    assert "__start__" in out and "__end__" in out  # ascii(grandalf)·mermaid 공통 노드
