"""ExecutionPolicy — 실행 정책 (요구사항 §12, F07).

정책은 Agent 코드에 하드코딩하지 않는다. spec.config["policy"] 또는 명시 인자로 주입.
Override 체계(Enterprise→Project→Agent→Execution)는 "더 구체적인 것이 우선"으로 시작하고,
Config Framework(§22) 도입 시 계층 병합으로 확장한다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, fields
from typing import Any

from klafi.core.exceptions import GuardrailException, PolicyException

# 결정적 실패(코드 버그·잘못된 입력)는 재시도해도 같은 결과 — LangGraph default_retry_on 과 같은 목록.
# ConnectionError 는 OSError 하위지만 일시 장애라 should_retry 에서 먼저 재시도로 판정한다.
_DETERMINISTIC: tuple[type[BaseException], ...] = (
    ValueError, TypeError, ArithmeticError, ImportError, LookupError, NameError, SyntaxError,
    RuntimeError, ReferenceError, StopIteration, StopAsyncIteration, OSError,
)


@dataclass
class ExecutionPolicy:
    timeout: float | None = None  # POL-01 / EXE-07 (초)
    max_retries: int = 0  # POL-02·04 / EXE-08
    concurrency: int | None = None  # 서버 전역 동시 실행 상한(초과 시 429). None=무제한
    backoff_base: float = 0.5  # POL-03 (초)
    backoff_factor: float = 2.0
    backoff_max: float = 30.0
    jitter: bool = True  # 동시 재시도 폭주(thundering herd) 완화 — 지연에 0~min(지연,1s) 무작위 가산
    # 결정적 실패는 재시도해도 같은 결과이므로 기본 제외.
    retry_on: tuple[type[BaseException], ...] = (Exception,)
    no_retry_on: tuple[type[BaseException], ...] = (GuardrailException, PolicyException, *_DETERMINISTIC)

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

    def merge(self, overrides: "dict | None") -> "ExecutionPolicy":
        """이 policy 위에 overrides(스칼라 dict)를 덮어 새 ExecutionPolicy를 만든다.

        per-agent config.yaml 의 policy 블록을 전역 policy 위에 얹을 때 쓴다 — 명시한 키만
        바뀌고 나머지(backoff 등)는 전역값을 상속한다. self 는 변형되지 않는다.
        """
        if not overrides:
            return self
        from dataclasses import replace

        allowed = {f.name for f in fields(self)} - {"retry_on", "no_retry_on"}
        unknown = set(overrides) - allowed
        if unknown:  # 오타 fail-fast (from_config 와 동일 규칙)
            from klafi.core.exceptions import ConfigSchemaError

            raise ConfigSchemaError(
                f"policy 설정에 알 수 없는 항목: {sorted(unknown)} (가능: {sorted(allowed)})"
            )
        return replace(self, **overrides)

    def backoff_delay(self, attempt: int) -> float:
        delay = min(self.backoff_base * (self.backoff_factor**attempt), self.backoff_max)
        return delay + random.uniform(0, min(delay, 1.0)) if self.jitter else delay

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        if attempt >= self.max_retries:
            return False
        if isinstance(exc, ConnectionError):  # 일시 장애 — OSError 하위지만 재시도 대상
            return True
        if isinstance(exc, self.no_retry_on):
            return False
        return isinstance(exc, self.retry_on)
