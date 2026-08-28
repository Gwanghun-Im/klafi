"""Checkpointer Adapter (요구사항 §10.1, F05 / MEM-01~05).

LangGraph Native 원칙: Checkpoint를 재구현하지 않는다.
KLAFI의 Checkpointer 표준 인터페이스 = LangGraph BaseCheckpointSaver 이며,
KLAFI가 표준화하는 것은 "Config(name/dict) → Saver 인스턴스" 해석과 자동 주입뿐이다.

Config 예:
    checkpoint: memory
    checkpoint: {type: postgres, conn_string: "postgresql://..."}
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.checkpoint.base import BaseCheckpointSaver

from klafi.core.exceptions import CheckpointException

# MEM-01: 별도 인터페이스를 만들지 않고 LangGraph 표준을 채택한다.
CheckpointerSPI = BaseCheckpointSaver

CheckpointerFactory = Callable[[dict[str, Any]], BaseCheckpointSaver]
_REGISTRY: dict[str, CheckpointerFactory] = {}


def register_checkpointer(name: str, factory: CheckpointerFactory) -> None:
    """프로젝트 Custom Adapter 등록 (확장성 NFR)."""
    _REGISTRY[name.lower()] = factory


def _memory_factory(_: dict[str, Any]) -> BaseCheckpointSaver:  # MEM-02
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


def _postgres_factory(cfg: dict[str, Any]) -> BaseCheckpointSaver:  # MEM-03
    """운영 배선: ConnectionPool + 스키마 setup.

    from_conn_string()은 context manager를 돌려주므로 쓰지 않는다.
    config: {type: postgres, conn_string: ..., min_size: 1, max_size: 10, setup: true}
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:
        raise CheckpointException(
            "postgres checkpointer에는 langgraph-checkpoint-postgres 패키지가 필요합니다"
        ) from exc

    saver = PostgresSaver(_build_pool(cfg, "checkpoint"))
    if cfg.get("setup", True):  # 스키마 없으면 생성(멱등). 운영에서 마이그레이션 분리 시 false
        saver.setup()
    return saver


def _build_pool(cfg: dict[str, Any], what: str) -> Any:
    """PostgreSQL ConnectionPool 생성 (Resource 재사용, NFR 성능)."""
    conn = cfg.get("conn_string") or cfg.get("url")
    if not conn:
        raise CheckpointException(f"postgres {what}에는 conn_string(url)이 필요합니다")
    try:
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
    except ImportError as exc:
        raise CheckpointException("psycopg[binary,pool] 패키지가 필요합니다") from exc

    connect_timeout = cfg.get("connect_timeout", 5)  # 기동 실패는 빠르고 명확하게
    pool = ConnectionPool(
        conn,
        min_size=cfg.get("min_size", 1),
        max_size=cfg.get("max_size", 10),
        timeout=cfg.get("pool_timeout", connect_timeout),
        open=False,
        # LangGraph Postgres 어댑터 요구사항
        kwargs={"autocommit": True, "row_factory": dict_row, "connect_timeout": connect_timeout},
    )
    try:
        pool.open(wait=True, timeout=connect_timeout)  # 접속 불가를 부트스트랩에서 검출
    except Exception as exc:  # noqa: BLE001
        pool.close()
        raise CheckpointException(
            f"postgres {what} 접속 실패: {conn.rsplit('@', 1)[-1]} ({exc})"
        ) from exc
    return pool


for _name in ("memory", "inmemory"):
    _REGISTRY[_name] = _memory_factory
for _name in ("postgres", "postgresql"):
    _REGISTRY[_name] = _postgres_factory


def resolve_checkpointer(config: Any) -> BaseCheckpointSaver | None:
    """None → None, Saver 인스턴스 → 그대로, name/dict → Registry로 생성."""
    if config is None or isinstance(config, BaseCheckpointSaver):
        return config
    if isinstance(config, str):
        name, cfg = config, {}
    elif isinstance(config, dict):
        name, cfg = config.get("type", ""), config
    else:
        raise CheckpointException(f"지원하지 않는 checkpoint config: {config!r}")
    factory = _REGISTRY.get(str(name).lower())
    if factory is None:
        raise CheckpointException(f"알 수 없는 checkpointer type: {name!r}")
    return factory(cfg)
