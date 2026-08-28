"""Execution 상태머신 (요구사항 §8).

CREATED → QUEUED → RUNNING → WAITING_APPROVAL → COMPLETED
예외: FAILED / CANCELLED / TIMEOUT

현재 WS2 슬라이스에서는 RUNNING/COMPLETED/FAILED/TIMEOUT만 전이한다.
QUEUED/WAITING_APPROVAL은 Factory·HITL(WS3/WS7) 붙일 때 사용한다.
"""

from __future__ import annotations

from enum import Enum


class ExecutionState(str, Enum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
