"""Execution Engine — Timeout/Retry 표준 적용 (요구사항 §8·§12, EXE-07/08).

retry 루프가 timeout을 감싼다 → 재시도마다 새 timeout이 적용된다.
set_state는 최종 상태(COMPLETED/FAILED/TIMEOUT)만 기록한다.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Awaitable, Callable

from klafi.core.exceptions import TimeoutException
from klafi.core.hook import is_control_flow

from .policy import ExecutionPolicy
from .state import ExecutionState

SetState = Callable[[ExecutionState], None]


def _timeout_sync(fn: Callable[[], Any], timeout: float | None) -> Any:
    if timeout is None:
        return fn()
    # ContextVar는 thread로 자동 전파되지 않으므로 현재 Context를 복사해 넘긴다.
    ctx = contextvars.copy_context()
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(ctx.run, fn)
    try:
        return fut.result(timeout=timeout)
    except FutureTimeout:
        raise TimeoutException(
            f"실행이 timeout {timeout}s를 초과 (동기 경로: 대기만 해제되고 작업 스레드는 끝까지 실행됨 — "
            "취소가 필요하면 ainvoke/astream 경로)",
            timeout=timeout,
        ) from None
    finally:
        # ponytail: 초과한 worker thread는 강제 종료 불가 → 백그라운드로 흘려보냄.
        #           협조적 취소가 필요하면 async(ainvoke) 경로를 쓴다.
        ex.shutdown(wait=False)


def run_sync(fn: Callable[[], Any], policy: ExecutionPolicy, set_state: SetState) -> Any:
    set_state(ExecutionState.RUNNING)
    attempt = 0
    while True:
        try:
            result = _timeout_sync(fn, policy.timeout)
            set_state(ExecutionState.COMPLETED)
            return result
        except Exception as exc:  # noqa: BLE001
            if is_control_flow(exc):  # interrupt(HITL): 재시도 금지, 승인 대기
                set_state(ExecutionState.WAITING_APPROVAL)
                raise
            if isinstance(exc, TimeoutException):
                # 동기 timeout 은 취소가 아니라 대기 해제 — 1회차 스레드가 아직 같은 ctx 위에서 돌고 있으므로
                # 재시도하면 두 attempt 가 동시에 실행된다. 동기 경로에서는 timeout 을 재시도하지 않는다.
                set_state(ExecutionState.TIMEOUT)
                raise
            if policy.should_retry(exc, attempt):
                time.sleep(policy.backoff_delay(attempt))
                attempt += 1
                continue
            set_state(
                ExecutionState.TIMEOUT
                if isinstance(exc, TimeoutException)
                else ExecutionState.FAILED
            )
            raise


async def _timeout_async(coro_fn: Callable[[], Awaitable[Any]], timeout: float | None) -> Any:
    if timeout is None:
        return await coro_fn()
    try:
        return await asyncio.wait_for(coro_fn(), timeout)  # 실제 협조적 취소
    except asyncio.TimeoutError:
        raise TimeoutException(f"실행이 timeout {timeout}s를 초과", timeout=timeout) from None


async def run_async(
    coro_fn: Callable[[], Awaitable[Any]], policy: ExecutionPolicy, set_state: SetState
) -> Any:
    set_state(ExecutionState.RUNNING)
    attempt = 0
    while True:
        try:
            result = await _timeout_async(coro_fn, policy.timeout)
            set_state(ExecutionState.COMPLETED)
            return result
        except Exception as exc:  # noqa: BLE001
            if is_control_flow(exc):  # interrupt(HITL): 재시도 금지, 승인 대기
                set_state(ExecutionState.WAITING_APPROVAL)
                raise
            if policy.should_retry(exc, attempt):
                await asyncio.sleep(policy.backoff_delay(attempt))
                attempt += 1
                continue
            set_state(
                ExecutionState.TIMEOUT
                if isinstance(exc, TimeoutException)
                else ExecutionState.FAILED
            )
            raise
