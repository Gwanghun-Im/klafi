"""AgentServer — 실행 Runtime 레지스트리 (요구사항 §19, F13).

HTTP Layer와 분리된 순수 런타임. 여러 Agent를 id로 보관하고 조회한다.
(F10 Agent Registry의 Governance/Control Plane과는 별개: 이건 in-process 실행 레지스트리다.)
"""

from __future__ import annotations

from typing import Any

from klafi.core.base_graph import BaseGraph
from klafi.core.exceptions import AgentNotFoundError


AgentNotFound = AgentNotFoundError  # 하위호환 별칭


class AgentServer:
    def __init__(self) -> None:
        self._agents: dict[str, BaseGraph] = {}

    def register(self, agent: BaseGraph, agent_id: str | None = None) -> str:
        aid = agent_id or agent.spec.id
        self._agents[aid] = agent
        return aid

    def get(self, agent_id: str) -> BaseGraph:
        try:
            return self._agents[agent_id]
        except KeyError:
            raise AgentNotFoundError(f"agent '{agent_id}' 미등록", agent_id=agent_id) from None

    def ids(self) -> list[str]:
        return list(self._agents)

    def metadata(self, agent_id: str) -> dict[str, Any]:
        return self.get(agent_id).spec.model_dump()

    def list_metadata(self) -> list[dict[str, Any]]:
        return [a.spec.model_dump() for a in self._agents.values()]
