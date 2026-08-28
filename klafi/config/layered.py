"""Config Framework (요구사항 §22).

5계층 우선순위 (뒤일수록 우선):
    Framework Default → Environment → Project → Agent → Runtime Override

§22 디렉터리 구조:
    config/
      framework.yaml            # 공통
      model.yaml                # → {model: ...}
      policy.yaml               # → {policy: ...}
      observability.yaml        # → {observability: ...}
      environments/<env>.yaml   # Environment 계층
      project.yaml              # Project 계층
      agents/<agent_id>.yaml    # Agent 계층
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

LAYER_ORDER = ("framework", "environment", "project", "agent", "runtime")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """dict는 재귀 병합, 그 외(스칼라/리스트)는 override가 대체."""
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def expand_env(value: Any) -> Any:
    """설정 값의 ${VAR} / ${VAR:default} 를 환경변수로 치환 (SEC-05 Secret 외부화).

    DB Credential·API Key를 config 파일에 평문으로 두지 않기 위한 장치.
    기본값 없이 환경변수도 없으면 기동 시 실패한다(fail-fast).
    """
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if not isinstance(value, str):
        return value

    def sub(m: re.Match) -> str:
        name, default = m.group(1), m.group(2)
        got = os.environ.get(name)
        if got is not None:
            return got
        if default is not None:
            return default
        from klafi.core.exceptions import ConfigNotFoundError

        raise ConfigNotFoundError(f"환경변수 '{name}' 미설정 (config의 ${{{name}}} 치환 실패)")

    return _ENV_RE.sub(sub, value)


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return expand_env(data or {})


class LayeredConfig:
    def __init__(self, **layers: dict[str, Any]) -> None:
        self._layers: dict[str, dict[str, Any]] = {n: {} for n in LAYER_ORDER}
        for name, data in layers.items():
            self.set(name, data or {})

    def set(self, layer: str, data: dict[str, Any]) -> "LayeredConfig":
        if layer not in self._layers:
            raise ValueError(f"알 수 없는 config 계층: {layer} (허용: {LAYER_ORDER})")
        self._layers[layer] = data
        return self

    def override(self, data: dict[str, Any]) -> "LayeredConfig":
        """Runtime Override 계층에 병합 (최우선)."""
        self._layers["runtime"] = deep_merge(self._layers["runtime"], data)
        return self

    def resolve(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in LAYER_ORDER:  # Framework → ... → Runtime 순으로 겹쳐 올림
            out = deep_merge(out, self._layers[name])
        return out

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.resolve()
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    @classmethod
    def from_dir(
        cls,
        path: str | Path,
        *,
        environment: str | None = None,
        project: bool = True,
        agent_id: str | None = None,
    ) -> "LayeredConfig":
        root = Path(path)
        # 경로 오타로 빈 설정이 조용히 로드되는 것을 막는다(fail-fast).
        if not root.is_dir():
            from klafi.core.exceptions import ConfigNotFoundError

            raise ConfigNotFoundError(f"config 디렉터리를 찾을 수 없습니다: {root}")
        if environment and not (root / "environments" / f"{environment}.yaml").exists():
            from klafi.core.exceptions import ConfigNotFoundError

            raise ConfigNotFoundError(f"environment '{environment}' 설정이 없습니다: {root}/environments/{environment}.yaml")

        # Framework 계층: framework.yaml + 도메인별 yaml을 네임스페이스로 병합
        framework = _load(root / "framework.yaml")
        for domain in ("model", "policy", "observability", "security", "context"):
            data = _load(root / f"{domain}.yaml")
            if data:
                framework = deep_merge(framework, {domain: data})

        env_layer = _load(root / "environments" / f"{environment}.yaml") if environment else {}
        project_layer = _load(root / "project.yaml") if project else {}
        agent_layer = _load(root / "agents" / f"{agent_id}.yaml") if agent_id else {}

        return cls(
            framework=framework,
            environment=env_layer,
            project=project_layer,
            agent=agent_layer,
        )
