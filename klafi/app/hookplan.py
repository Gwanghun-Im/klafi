"""HookPlan — config/hooks.yaml 로 **명명 훅**을 에이전트별로 해석.

YAML 스키마 (고정 위치: <config_dir>/hooks.yaml):

    all:                       # 모든 에이전트
      hooks: [event, audit]    # 이름 (prebuilt 또는 프로젝트 등록)
    agents:                    # 개별 에이전트
      refund:
        hooks: [refund_audit]

가드레일은 더 이상 YAML로 배치하지 않는다. 코드에서 @klafi_node/@klafi_graph 데코레이터나
GuardrailHook(공통 훅)으로 직접 적용한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from klafi.core.hook import Hook


def _validate(spec: dict[str, Any], where: str) -> None:
    """스키마 오타를 조용히 무시하지 않도록 fail-fast 검증."""
    from klafi.core.exceptions import ConfigSchemaError

    unknown = set(spec) - {"all", "agents"}
    if unknown:
        raise ConfigSchemaError(f"{where}: 알 수 없는 최상위 항목 {sorted(unknown)} (가능: ['all', 'agents'])")

    def check_block(block: Any, label: str) -> None:
        if block is None:
            return
        if not isinstance(block, dict):
            raise ConfigSchemaError(f"{where}: {label} 은 매핑이어야 합니다")
        bad = set(block) - {"hooks"}
        if bad:
            raise ConfigSchemaError(
                f"{where}: {label} 에 알 수 없는 항목 {sorted(bad)} (가능: ['hooks']). "
                "가드레일은 YAML이 아니라 코드(@klafi_node/@klafi_graph)로 적용합니다"
            )

    check_block(spec.get("all"), "all")
    agents = spec.get("agents") or {}
    if not isinstance(agents, dict):
        raise ConfigSchemaError(f"{where}: agents 는 매핑이어야 합니다")
    for agent_id, block in agents.items():
        check_block(block, f"agents.{agent_id}")


class HookPlan:
    def __init__(self, spec: dict[str, Any], where: str = "hooks.yaml") -> None:
        _validate(spec, where)
        self._all = spec.get("all") or {}
        self._agents = spec.get("agents") or {}

    @classmethod
    def from_file(cls, path: str | Path) -> "HookPlan":
        p = Path(path)
        if not p.exists():
            return cls({})
        return cls(yaml.safe_load(p.read_text(encoding="utf-8")) or {}, where=str(p))

    def validate_names(self) -> None:
        """YAML이 참조하는 훅 이름이 전부 등록돼 있는지 확인 (부트스트랩 시 1회)."""
        for agent_id in ["__all__", *self._agents]:
            self.for_agent(agent_id)  # 미등록 이름이면 여기서 예외

    def for_agent(self, agent_id: str) -> list[Hook]:
        """all + 개별 에이전트의 명명 훅을 해석해 Hook 리스트로 반환."""
        from klafi.hookdefs import resolve_named_hooks

        agent = self._agents.get(agent_id) or {}
        names = list(self._all.get("hooks") or []) + list(agent.get("hooks") or [])
        return resolve_named_hooks(names)
