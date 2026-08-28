"""ExecutionPolicy — 실행 정책 (요구사항 §12, F07).

정책은 Agent 코드에 하드코딩하지 않는다. spec.config["policy"] 또는 명시 인자로 주입.
Override 체계(Enterprise→Project→Agent→Execution)는 "더 구체적인 것이 우선"으로 시작하고,
Config Framework(§22) 도입 시 계층 병합으로 확장한다.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from klafi.core.exceptions import GuardrailException, PolicyException


@dataclass
class ExecutionPolicy:
    timeout: float | None = None  # POL-01 / EXE-07 (초)
    max_retries: int = 0  # POL-02·04 / EXE-08
    concurrency: int | None = None  # 서버 전역 동시 실행 상한(초과 시 429). None=무제한
    backoff_base: float = 0.5  # POL-03 (초)
    backoff_factor: float = 2.0
    backoff_max: float = 30.0
    # 결정적 실패는 재시도해도 같은 결과이므로 기본 제외.
    retry_on: tuple[type[BaseException], ...] = (Exception,)
    no_retry_on: tuple[type[BaseException], ...] = (GuardrailException, PolicyException)

    @classmethod
    def from_config(cls, data: "ExecutionPolicy | dict | None") -> "ExecutionPolicy | None":
        if data is None or isinstance(data, cls):
            return data
        # dict에서는 스칼라 설정만 받는다(retry_on 등 예외타입은 코드 레벨).
        allowed = {f.name for f in fields(cls)} - {"retry_on", "no_retry_on"}
        unknown = set(data) - allowed
        if unknown:  # 오타가 조용히 무시되지 않도록 fail-fast
            from klafi.core.exceptions import ConfigSchemaError

            raise ConfigSchemaError(
                f"policy 설정에 알 수 없는 항목: {sorted(unknown)} (가능: {sorted(allowed)})"
            )
        return cls(**data)

    def backoff_delay(self, attempt: int) -> float:
        return min(self.backoff_base * (self.backoff_factor**attempt), self.backoff_max)

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        if attempt >= self.max_retries:
            return False
        if isinstance(exc, self.no_retry_on):
            return False
        return isinstance(exc, self.retry_on)
