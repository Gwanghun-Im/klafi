"""Long-Term Memory (요구사항 §10.2, F05 / MEM-11~17).

Checkpoint(=Workflow 실행상태)와 다른 개념: 세션을 넘어 지속되는 사용자/Agent 지식.

LangGraph Native 원칙: Memory Store를 재구현하지 않는다.
- MEM-11: KLAFI Memory Store 표준 인터페이스 = LangGraph BaseStore.
- MEM-12/13/14: Scope = namespace tuple (user / agent / project).
- MEM-15: TTL — 백엔드가 지원할 때만(Postgres/Redis). InMemoryStore는 ttl=None만 허용.
- MEM-16: 삭제 API = forget().
- MEM-17: 개인정보 정책 = 저장 전 PII 스크럽(pii_filter).

Config 예:  store: memory   |   store: {type: postgres, conn_string: "..."}
"""

from __future__ import annotations

import re
from typing import Any, Callable

from langgraph.store.base import BaseStore

from klafi.core.exceptions import KlafiException

# MEM-11: 별도 인터페이스를 만들지 않고 LangGraph 표준을 채택.
MemoryStoreSPI = BaseStore

Scope = tuple[str, ...]


# ── Scope 빌더 (MEM-12/13/14) ───────────────────────────────────────────
def user_scope(user_id: str) -> Scope:
    return ("user", user_id)


def agent_scope(agent_id: str) -> Scope:
    return ("agent", agent_id)


def project_scope(project_id: str) -> Scope:
    return ("project", project_id)


# ── PII 정책 (MEM-17) ───────────────────────────────────────────────────
_PII_RES = [re.compile(p, re.IGNORECASE) for p in (r"\d{6}-\d{7}", r"\d{16}", r"[\w.]+@[\w.]+\.\w+")]
PIIFilter = Callable[[dict[str, Any]], dict[str, Any]]


def redact_pii(value: dict[str, Any]) -> dict[str, Any]:
    """저장 전 문자열 값의 PII를 [REDACTED]로 치환."""

    def scrub(v: Any) -> Any:
        if isinstance(v, str):
            for rx in _PII_RES:
                v = rx.sub("[REDACTED]", v)
            return v
        if isinstance(v, dict):
            return {k: scrub(x) for k, x in v.items()}
        if isinstance(v, list):
            return [scrub(x) for x in v]
        return v

    return scrub(value)


class MemoryStore:
    """BaseStore 위의 KLAFI 편의 래퍼: Scope·TTL·삭제·PII 정책."""

    def __init__(self, store: BaseStore, pii_filter: PIIFilter | None = None) -> None:
        self.store = store
        self._pii = pii_filter

    def remember(self, scope: Scope, key: str, value: dict[str, Any], ttl: float | None = None) -> None:
        if self._pii:
            value = self._pii(value)  # MEM-17
        if ttl is None:
            self.store.put(scope, key, value)
        else:
            self.store.put(scope, key, value, ttl=ttl)  # MEM-15 (백엔드 지원 필요)

    def recall(self, scope: Scope, key: str) -> dict[str, Any] | None:
        item = self.store.get(scope, key)
        return item.value if item else None

    def search(self, scope: Scope, query: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        return [i.value for i in self.store.search(scope, query=query, limit=limit)]

    def forget(self, scope: Scope, key: str) -> None:  # MEM-16
        self.store.delete(scope, key)


# ── Store Adapter 해석 (FAC-04 Store 자동 주입) ─────────────────────────
StoreFactory = Callable[[dict[str, Any]], BaseStore]
_REGISTRY: dict[str, StoreFactory] = {}


def register_store(name: str, factory: StoreFactory) -> None:
    _REGISTRY[name.lower()] = factory


def _memory_factory(_: dict[str, Any]) -> BaseStore:
    from langgraph.store.memory import InMemoryStore

    return InMemoryStore()


def _postgres_factory(cfg: dict[str, Any]) -> BaseStore:
    """운영 배선: ConnectionPool + 스키마 setup (checkpoint 어댑터와 동일 방식)."""
    try:
        from langgraph.store.postgres import PostgresStore
    except ImportError as exc:
        raise KlafiException("postgres store에는 langgraph-checkpoint-postgres 패키지가 필요합니다") from exc

    from .checkpoint import _build_pool

    store = PostgresStore(_build_pool(cfg, "store"))
    if cfg.get("setup", True):
        store.setup()
    return store


for _n in ("memory", "inmemory"):
    _REGISTRY[_n] = _memory_factory
for _n in ("postgres", "postgresql"):
    _REGISTRY[_n] = _postgres_factory


def resolve_store(config: Any) -> BaseStore | None:
    if config is None or isinstance(config, BaseStore):
        return config
    if isinstance(config, str):
        name, cfg = config, {}
    elif isinstance(config, dict):
        name, cfg = config.get("type", ""), config
    else:
        raise KlafiException(f"지원하지 않는 store config: {config!r}")
    factory = _REGISTRY.get(str(name).lower())
    if factory is None:
        raise KlafiException(f"알 수 없는 store type: {name!r}")
    return factory(cfg)
