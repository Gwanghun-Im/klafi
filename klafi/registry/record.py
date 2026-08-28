"""Agent Registry Record & Lifecycle (요구사항 §15, F10).

Registry는 Agent Source Repository가 아니라 운영·Governance를 위한 Control Plane이다.
AgentRecord는 Agent의 운영 메타데이터를, AgentLifecycle은 상태 전이를 표준화한다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from klafi.core.spec import AgentSpec


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentLifecycle(str, Enum):  # §15 Lifecycle
    DEVELOPMENT = "DEVELOPMENT"
    TEST = "TEST"
    APPROVED = "APPROVED"
    PRODUCTION = "PRODUCTION"
    DEPRECATED = "DEPRECATED"
    RETIRED = "RETIRED"


# 허용 전이 (Governance). 앞으로 진행 + 일부 롤백 허용, RETIRED는 종착.
_TRANSITIONS: dict[AgentLifecycle, set[AgentLifecycle]] = {
    AgentLifecycle.DEVELOPMENT: {AgentLifecycle.TEST},
    AgentLifecycle.TEST: {AgentLifecycle.APPROVED, AgentLifecycle.DEVELOPMENT},
    AgentLifecycle.APPROVED: {AgentLifecycle.PRODUCTION, AgentLifecycle.TEST},
    AgentLifecycle.PRODUCTION: {AgentLifecycle.DEPRECATED},
    AgentLifecycle.DEPRECATED: {AgentLifecycle.RETIRED, AgentLifecycle.PRODUCTION},
    AgentLifecycle.RETIRED: set(),
}


def can_transition(src: AgentLifecycle, dst: AgentLifecycle) -> bool:
    return dst in _TRANSITIONS[src]


class AgentRecord(BaseModel):
    # §15 Registry Metadata
    agent_id: str
    name: str
    version: str
    project: str | None = None
    owner: str | None = None  # REG-04
    description: str | None = None
    agent_type: str | None = None
    framework_version: str | None = None  # REG-06
    status: AgentLifecycle = AgentLifecycle.DEVELOPMENT  # REG-05
    model: str | None = None
    tools: list[str] = Field(default_factory=list)
    security_level: str | None = None
    endpoint: str | None = None  # REG-07
    created_date: str = Field(default_factory=_now)
    approved_date: str | None = None  # REG-08

    @classmethod
    def from_spec(
        cls,
        spec: AgentSpec,
        *,
        owner: str | None = None,
        endpoint: str | None = None,
        security_level: str | None = None,
        framework_version: str | None = None,
    ) -> "AgentRecord":
        return cls(
            agent_id=spec.id,
            name=spec.name,
            version=spec.version,
            project=spec.project,
            owner=owner or spec.owner,
            description=spec.description,
            agent_type=spec.agent_type,
            framework_version=framework_version,
            model=spec.model,
            tools=list(spec.tools),
            security_level=security_level,
            endpoint=endpoint,
        )
