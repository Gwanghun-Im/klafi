"""Agent Registry 검증 (요구사항 §15, F10 / REG-01~10)."""

import pytest

from klafi import (
    AgentLifecycle,
    AgentRecord,
    AgentRegistry,
    AgentSpec,
    SimpleAgent,
    __version__,
)
from klafi.registry import AgentNotRegistered, InvalidTransition


# ── 등록/조회/버전 (REG-01/02/03/06) ────────────────────────────────────
def test_register_agent_records_metadata_and_framework_version():
    reg = AgentRegistry()
    agent = SimpleAgent(model=lambda p: p, spec=AgentSpec(id="qa", name="QA", version="1.0.0", project="p1", agent_type="simple"))
    rec = reg.register_agent(agent, owner="team-a", endpoint="/agents/qa/invoke", security_level="internal")

    assert rec.owner == "team-a"  # REG-04
    assert rec.endpoint == "/agents/qa/invoke"  # REG-07
    assert rec.framework_version == __version__  # REG-06
    assert rec.status == AgentLifecycle.DEVELOPMENT
    assert reg.get("qa", "1.0.0").name == "QA"


def test_versions_and_latest():
    reg = AgentRegistry()
    for v in ("1.0.0", "1.2.0", "1.10.0"):
        reg.register(AgentRecord(agent_id="a", name="A", version=v))
    assert reg.versions("a") == ["1.0.0", "1.2.0", "1.10.0"]  # 숫자 정렬
    assert reg.latest("a").version == "1.10.0"


def test_get_unregistered_raises():
    with pytest.raises(AgentNotRegistered):
        AgentRegistry().get("nope", "1.0.0")


# ── Lifecycle 전이 (REG-05/08/10) ───────────────────────────────────────
def test_valid_lifecycle_flow_and_approved_date():
    reg = AgentRegistry()
    reg.register(AgentRecord(agent_id="a", name="A", version="1.0.0"))

    reg.transition("a", "1.0.0", AgentLifecycle.TEST)
    rec = reg.approve("a", "1.0.0")  # TEST → APPROVED
    assert rec.status == AgentLifecycle.APPROVED
    assert rec.approved_date is not None  # REG-08

    reg.transition("a", "1.0.0", AgentLifecycle.PRODUCTION)
    rec = reg.deprecate("a", "1.0.0")  # REG-10
    assert rec.status == AgentLifecycle.DEPRECATED
    reg.transition("a", "1.0.0", AgentLifecycle.RETIRED)
    assert reg.get("a", "1.0.0").status == AgentLifecycle.RETIRED


def test_invalid_transition_rejected():
    reg = AgentRegistry()
    reg.register(AgentRecord(agent_id="a", name="A", version="1.0.0"))
    # DEVELOPMENT → PRODUCTION은 건너뛰기 → 불가
    with pytest.raises(InvalidTransition):
        reg.transition("a", "1.0.0", AgentLifecycle.PRODUCTION)
    # RETIRED는 종착
    for to in (AgentLifecycle.TEST, AgentLifecycle.APPROVED, AgentLifecycle.PRODUCTION, AgentLifecycle.DEPRECATED, AgentLifecycle.RETIRED):
        reg.get("a", "1.0.0").status = AgentLifecycle.RETIRED
        with pytest.raises(InvalidTransition):
            reg.transition("a", "1.0.0", to)


# ── 목록 필터 (REG-02) ──────────────────────────────────────────────────
def test_list_filters_by_status_and_project():
    reg = AgentRegistry()
    reg.register(AgentRecord(agent_id="a", name="A", version="1.0.0", project="p1", status=AgentLifecycle.PRODUCTION))
    reg.register(AgentRecord(agent_id="b", name="B", version="1.0.0", project="p1", status=AgentLifecycle.DEVELOPMENT))
    reg.register(AgentRecord(agent_id="c", name="C", version="1.0.0", project="p2", status=AgentLifecycle.PRODUCTION))

    assert {r.agent_id for r in reg.list(status=AgentLifecycle.PRODUCTION)} == {"a", "c"}
    assert {r.agent_id for r in reg.list(project="p1")} == {"a", "b"}
    assert {r.agent_id for r in reg.list(status=AgentLifecycle.PRODUCTION, project="p2")} == {"c"}
