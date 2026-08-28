"""@klafi_node — 노드 전용 미들웨어 + 가드레일을 노드 함수에 직접 붙인다 (요구사항 §11, F06).

KlafiGraph 는 이미 모든 노드에 common(플랫폼)→agent 훅을 자동 발화한다(`wrap_node`).
`@klafi_node` 는 거기에 **이 노드에만** 붙는 것을 더한다:

* before / after — 노드에 들어오는 state / 나가는 result 에 적용할 **가드레일·미들웨어 리스트**.
  한 리스트에 섞어 넣고 **원소 타입으로 구분**한다(klafi.core.middleware):
    - 가드레일(`.check` 보유) : 등급(BLOCK/WARN/MASK)·감사로그·GuardrailViolationError
    - 미들웨어(그냥 콜러블)    : 세션·로그인 검증, state 보강 등. 반환하면 값을 **교체**한다.
      before(state[, ctx]) -> state | None / after(result[, ctx]) -> result | None
      검증 실패는 그냥 예외를 던지면 된다(fail-close).
* on_error(exc, state[, ctx]) — 예외 관측(그 뒤 재발생). 흐름제어/interrupt 는 제외.

리스트 순서대로 적용되므로 "정규화 후 검사" 같은 순서도 표현할 수 있다.

발화 순서(어니언):
  common.before → agent.before → **before 파이프라인** → fn
              → **after 파이프라인** → agent.after → common.after

    class MyAgent(KlafiGraph):
        def define(self):
            def require_login(state, ctx):        # 미들웨어: 검증 + 보강
                if not ctx.security_context.get("user_id"):
                    raise PermissionError("로그인 필요")
                return {**state, "verified": True}  # state 교체

            @klafi_node("plan", before=[require_login, no_secrets], after=[mask_pii])
            def plan(state):
                ...
            self.add_node("plan", plan)
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable

from .context import get_context
from .middleware import aapply, afire_error, apply, as_list, fire_error


def klafi_node(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    before: Any = None,     # 가드레일·미들웨어 (들어온 state)
    after: Any = None,      # 가드레일·미들웨어 (반환값)
    on_error: Any = None,
) -> Any:
    """노드 함수에 가드레일·미들웨어를 붙인다. **name 은 필수**.

        @klafi_node("plan")                                  # 위치 인자로 이름
        @klafi_node("plan", before=[require_login, no_secrets])   # 미들웨어 + 가드레일 혼합
    """
    # @klafi_node("name") — 위치 인자로 이름을 준 경우
    if isinstance(fn, str):
        name, fn = fn, None
    if not name:
        from .exceptions import AgentExecutionException

        raise AgentExecutionException("@klafi_node 에는 name 이 필수입니다 (예: @klafi_node(\"plan\"))")
    pipe_before, pipe_after, mws_error = as_list(before), as_list(after), as_list(on_error)

    def deco(f: Callable) -> Callable:
        node_name = name
        is_async = asyncio.iscoroutinefunction(f)

        def _state_at(args: tuple) -> int:
            return 1 if args and _is_graph(args[0]) else 0

        if is_async:
            @functools.wraps(f)
            async def awrapped(*a: Any, **kw: Any) -> Any:
                a, state, ctx = list(a), None, get_context()
                at = _state_at(tuple(a))
                state = a[at] if len(a) > at else kw.get("state")
                if pipe_before:
                    state = await aapply(pipe_before, state, ctx, "input")
                    if len(a) > at:
                        a[at] = state
                    else:
                        kw["state"] = state
                try:
                    result = await f(*a, **kw)
                except BaseException as exc:
                    await afire_error(mws_error, exc, state, ctx)
                    raise
                return await aapply(pipe_after, result, ctx, "output")

            return _stamp(awrapped, node_name)

        @functools.wraps(f)
        def wrapped(*a: Any, **kw: Any) -> Any:
            a, ctx = list(a), get_context()
            at = _state_at(tuple(a))
            state = a[at] if len(a) > at else kw.get("state")
            if pipe_before:
                state = apply(pipe_before, state, ctx, "input")
                if len(a) > at:
                    a[at] = state
                else:
                    kw["state"] = state
            try:
                result = f(*a, **kw)
            except BaseException as exc:
                fire_error(mws_error, exc, state, ctx)
                raise
            return apply(pipe_after, result, ctx, "output")

        return _stamp(wrapped, node_name)

    return deco(fn) if callable(fn) else deco


def _stamp(wrapper: Callable, node_name: str) -> Callable:
    """KlafiGraph 가 노드 강제 검사에 쓰는 마커."""
    wrapper.__klafi_node__ = True  # type: ignore[attr-defined]
    wrapper.__klafi_node_name__ = node_name  # type: ignore[attr-defined]
    return wrapper


def _is_graph(obj: Any) -> bool:
    from .graph import KlafiGraph

    return isinstance(obj, KlafiGraph)
