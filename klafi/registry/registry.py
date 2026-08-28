"""AgentRegistry — Control Plane (요구사항 §15, F10).

(agent_id, version)을 키로 AgentRecord를 보관하고 Lifecycle 전이를 통제한다.
저장은 RegistryStore Protocol로 추상화 → 운영에서는 DB Adapter로 교체.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Protocol

from klafi.core.exceptions import AgentNotFoundError, KlafiException

from .record import AgentLifecycle, AgentRecord, can_transition

_audit = logging.getLogger("klafi.registry")


AgentNotRegistered = AgentNotFoundError  # 하위호환 별칭


class InvalidTransition(KlafiException):
    error_code = "INVALID_LIFECYCLE_TRANSITION"


def _ver_key(v: str) -> tuple:
    parts = []
    for p in v.split("."):
        parts.append(int(p) if p.isdigit() else 0)
    return tuple(parts)


class RegistryStore(Protocol):
    def save(self, record: AgentRecord) -> None: ...
    def get(self, agent_id: str, version: str) -> AgentRecord | None: ...
    def all(self) -> list[AgentRecord]: ...


class InMemoryRegistryStore:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], AgentRecord] = {}

    def save(self, record: AgentRecord) -> None:
        self._records[(record.agent_id, record.version)] = record

    def get(self, agent_id: str, version: str) -> AgentRecord | None:
        return self._records.get((agent_id, version))

    def all(self) -> list[AgentRecord]:
        return list(self._records.values())


class AgentRegistry:
    def __init__(self, store: RegistryStore | None = None) -> None:
        self._store = store or InMemoryRegistryStore()

    # ── 등록/조회 (REG-01/02/03) ────────────────────────────────────────
    def register(self, record: AgentRecord) -> AgentRecord:
        self._store.save(record)
        _audit.info("registry.register agent=%s version=%s status=%s", record.agent_id, record.version, record.status.value)
        return record

    def register_agent(self, agent: Any, **meta: Any) -> AgentRecord:
        """BaseGraph의 spec으로부터 Record를 만들어 등록. framework_version 자동 기록(REG-06)."""
        from klafi import __version__

        record = AgentRecord.from_spec(agent.spec, framework_version=meta.pop("framework_version", __version__), **meta)
        return self.register(record)

    def get(self, agent_id: str, version: str) -> AgentRecord:
        rec = self._store.get(agent_id, version)
        if rec is None:
            raise AgentNotFoundError(f"agent '{agent_id}' v{version} 미등록", agent_id=agent_id, version=version)
        return rec

    def versions(self, agent_id: str) -> list[str]:
        vs = [r.version for r in self._store.all() if r.agent_id == agent_id]
        return sorted(vs, key=_ver_key)

    def latest(self, agent_id: str) -> AgentRecord:
        vs = self.versions(agent_id)
        if not vs:
            raise AgentNotFoundError(f"agent '{agent_id}' 미등록", agent_id=agent_id)
        return self.get(agent_id, vs[-1])

    def list(self, *, status: AgentLifecycle | None = None, project: str | None = None) -> list[AgentRecord]:
        out = self._store.all()
        if status is not None:
            out = [r for r in out if r.status == status]
        if project is not None:
            out = [r for r in out if r.project == project]
        return out

    # ── Lifecycle (REG-05/08/10) ────────────────────────────────────────
    def transition(self, agent_id: str, version: str, to: AgentLifecycle) -> AgentRecord:
        rec = self.get(agent_id, version)
        if not can_transition(rec.status, to):
            raise InvalidTransition(
                f"{rec.status.value} → {to.value} 전이 불가", agent_id=agent_id, version=version
            )
        rec.status = to
        if to == AgentLifecycle.APPROVED:
            rec.approved_date = datetime.now(timezone.utc).isoformat()
        self._store.save(rec)
        _audit.info("registry.transition agent=%s version=%s to=%s", agent_id, version, to.value)
        return rec

    def approve(self, agent_id: str, version: str) -> AgentRecord:  # REG-08
        return self.transition(agent_id, version, AgentLifecycle.APPROVED)

    def deprecate(self, agent_id: str, version: str) -> AgentRecord:  # REG-10
        return self.transition(agent_id, version, AgentLifecycle.DEPRECATED)
