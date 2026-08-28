"""before/after/on_error 파이프라인 — @klafi_node(노드)와 @klafi_graph(워크플로우)가 공유한다.

한 리스트에 두 종류가 섞여 들어간다. 원소 타입으로 구분한다:

  * `.check` 가 있으면 **문자열 정책 가드레일** → enforce(바인딩이 문자열 리프마다 검사·치환).
    판정(BLOCK/WARN)과 치환(MASK, replacement) 모두 여기서.
  * 그 외 콜러블은 **값 콜러블** → 값 전체를 한 번 받는다. 반환이 None이 아니면 값을 교체.
    리프 유무와 무관하게 **항상 한 번** 돈다 — messages가 비어도 도는 게 필요한 검증
    (require_orders_read 같은 권한 확인)이 여기 속한다. 실패는 예외(fail-close). 관측만 하면
    None을 반환한다(노드 지역 관측 = audit_log).

    before(value)        / before(value, ctx)        -> value | None
    after(value)         / after(value, ctx)         -> value | None
    on_error(exc)        / on_error(exc, value, ctx) -> None

리스트 **순서대로** 적용되므로 "정규화 후 검사" 같은 순서도 표현할 수 있다.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from .hook import is_control_flow


def as_list(x: Any) -> list:
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def arity(fn: Callable) -> int:
    """위치 인자 개수 — ctx를 받는 미들웨어인지 판별한다."""
    try:
        return len([
            p for p in inspect.signature(fn).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ])
    except (ValueError, TypeError):
        return 1


def is_guardrail(x: Any) -> bool:
    """가드레일이면 True — `check` 를 가진 객체(@guardrail 데코레이터 산출물, Guardrail 구현체)."""
    return hasattr(x, "check")


def apply(items: list[Any], value: Any, ctx: Any, stage: str = "node") -> Any:
    """가드레일·값 콜러블을 리스트 순서대로 적용하고 최종 값을 돌려준다."""
    from klafi.guardrail.base import enforce  # lazy: core 독립성

    for it in items:
        if is_guardrail(it):
            value = enforce([it], value, stage, ctx)
        else:
            new = it(value, ctx) if arity(it) >= 2 else it(value)
            if new is not None:
                value = new
    return value


async def aapply(items: list[Any], value: Any, ctx: Any, stage: str = "node") -> Any:
    """apply의 async 버전 — 코루틴 값 콜러블도 허용한다."""
    from klafi.guardrail.base import enforce

    for it in items:
        if is_guardrail(it):
            value = enforce([it], value, stage, ctx)
        else:
            new = it(value, ctx) if arity(it) >= 2 else it(value)
            if inspect.isawaitable(new):
                new = await new
            if new is not None:
                value = new
    return value


def fire_error(mws: list[Callable], exc: BaseException, value: Any, ctx: Any) -> None:
    """on_error 미들웨어 발화. 흐름제어(interrupt/HITL)는 오류가 아니므로 제외."""
    if is_control_flow(exc):
        return
    for mw in mws:
        r = mw(exc, value, ctx) if arity(mw) >= 3 else mw(exc)
        if inspect.isawaitable(r):
            r.close()  # sync 경로에서 코루틴이 오면 실행하지 않고 정리(경고 방지)


async def afire_error(mws: list[Callable], exc: BaseException, value: Any, ctx: Any) -> None:
    if is_control_flow(exc):
        return
    for mw in mws:
        r = mw(exc, value, ctx) if arity(mw) >= 3 else mw(exc)
        if inspect.isawaitable(r):
            await r
