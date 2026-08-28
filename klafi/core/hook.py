"""Hook / AOP 엔진 (요구사항 §11, F06).

목표: 개발자가 Node마다 Logging/Guardrail/Trace 코드를 반복해 넣지 않도록,
BaseGraph가 모든 Node를 자동 래핑해 Before/After/Error/Finally Hook을 발화한다.

Fail-Open/Close (§25): Hook이 예외를 던졌을 때 업무를 계속할지(fail_open=True,
로깅/트레이스류)  중단할지(fail_open=False, Guardrail류)를 Hook별로 정한다.
Error/Finally Hook은 이미 예외 처리 중이므로 항상 삼킨다.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable

from .context import ExecutionContext, get_context

_engine_log = logging.getLogger("klafi.hook")


def is_control_flow(exc: BaseException) -> bool:
    """LangGraph 제어흐름 신호(interrupt/command bubbling)는 오류가 아니다.

    이런 예외는 error Hook을 발화하지 않고(그냥 통과), retry 대상도 아니다.
    """
    try:
        from langgraph.errors import GraphBubbleUp
    except ImportError:  # pragma: no cover
        return False
    return isinstance(exc, GraphBubbleUp)


class Hook:
    """필요한 콜백만 override 한다. 나머지는 no-op."""

    priority: int = 100  # 낮을수록 먼저(before). after/finally는 역순.
    enabled: bool = True
    fail_open: bool = True  # False면 이 Hook의 예외가 실행을 중단시킴(Guardrail)

    # Node 단위
    def before_node(self, node: str, state: Any, ctx: ExecutionContext | None) -> None: ...
    def after_node(self, node: str, state: Any, result: Any, ctx: ExecutionContext | None) -> None: ...
    def on_node_error(self, node: str, state: Any, exc: BaseException, ctx: ExecutionContext | None) -> None: ...
    def finally_node(self, node: str, state: Any, ctx: ExecutionContext | None) -> None: ...

    # Agent(=Graph, 전체 실행) 단위
    def before_agent(self, input: Any, ctx: ExecutionContext | None) -> None: ...
    def after_agent(self, input: Any, result: Any, ctx: ExecutionContext | None) -> None: ...
    def on_agent_error(self, input: Any, exc: BaseException, ctx: ExecutionContext | None) -> None: ...
    def finally_agent(self, input: Any, ctx: ExecutionContext | None) -> None: ...

    # Tool 단위 (HOK-04). before/after_tool 은 반환하면 kwargs/result 를 교체한다(_transform).
    def before_tool(self, tool: str, kwargs: dict, ctx: ExecutionContext | None) -> Any: ...
    def after_tool(self, tool: str, kwargs: dict, result: Any, ctx: ExecutionContext | None) -> Any: ...
    def on_tool_error(self, tool: str, kwargs: dict, exc: BaseException, ctx: ExecutionContext | None) -> None: ...

    # Model(LLM) 단위 (HOK-05). before/after_model 은 반환하면 prompt/result 를 교체한다.
    def before_model(self, model: str, prompt: str, ctx: ExecutionContext | None) -> Any: ...
    def after_model(self, model: str, prompt: str, result: Any, ctx: ExecutionContext | None) -> Any: ...


# ── 전역 Hook 등록 (HOK-08) ────────────────────────────────────────────
_GLOBAL_HOOKS: list[Hook] = []


def register_hook(hook: Hook) -> None:
    _GLOBAL_HOOKS.append(hook)


def clear_hooks() -> None:
    _GLOBAL_HOOKS.clear()


def resolve_hooks(agent_hooks: list[Hook]) -> list[Hook]:
    """전역 + Agent Hook을 합쳐 enabled만, priority 오름차순 정렬 (HOK-09/11/12)."""
    merged = [h for h in (_GLOBAL_HOOKS + agent_hooks) if h.enabled]
    return sorted(merged, key=lambda h: h.priority)


# ── 실행 중 활성 Hook (Tool/Model 경계에서 참조) ────────────────────────
from contextlib import contextmanager  # noqa: E402
from contextvars import ContextVar  # noqa: E402
from typing import Iterator  # noqa: E402

_active_hooks: ContextVar[list["Hook"]] = ContextVar("klafi_active_hooks", default=[])


def active_hooks() -> list["Hook"]:
    return _active_hooks.get()


@contextmanager
def bind_hooks(hooks: list["Hook"]) -> Iterator[None]:
    token = _active_hooks.set(hooks)
    try:
        yield
    finally:
        _active_hooks.reset(token)


# ── 발화 (fail_open 처리) ──────────────────────────────────────────────
def _fire(hook: Hook, method: str, *args: Any, swallow: bool) -> None:
    fn = getattr(hook, method)
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001
        if swallow or hook.fail_open:
            _engine_log.warning("hook %s.%s 무시됨(fail-open): %s", type(hook).__name__, method, exc)
        else:
            raise  # Guardrail 등 fail-close: 실행 중단


def _before(hooks: list[Hook], method: str, *args: Any) -> None:
    for h in hooks:  # 오름차순
        _fire(h, method, *args, swallow=False)


def _after(hooks: list[Hook], method: str, *args: Any) -> None:
    for h in reversed(hooks):  # 역순(onion)
        _fire(h, method, *args, swallow=False)


def _error(hooks: list[Hook], method: str, *args: Any) -> None:
    for h in hooks:
        _fire(h, method, *args, swallow=True)  # 오류 처리 중 → 항상 삼킴


def _finally(hooks: list[Hook], method: str, *args: Any) -> None:
    for h in reversed(hooks):
        _fire(h, method, *args, swallow=True)


def _transform(
    hooks: list[Hook], method: str, value: Any, rebuild: Callable[[Any], tuple], *, reverse: bool = False
) -> Any:
    """경계 훅(before/after_model·tool)을 발화하되 **반환값이 None이 아니면 값을 교체**한다.

    데코레이터를 붙일 수 없는 LLM·Tool 경계에서 GuardrailHook 이 프롬프트·응답·인자·결과를
    마스킹할 수 있게 한다. rebuild(value)가 메서드 시그니처에 맞는 인자 튜플을 만든다(값 위치는
    호출부가 안다). fail_open 훅이 예외를 던지면 값은 **그 훅 직전 상태**를 유지한다 — 정상
    반환에서만 대입하므로 부분 변환이 새지 않는다.
    """
    for h in (reversed(hooks) if reverse else hooks):
        try:
            out = getattr(h, method)(*rebuild(value))
        except Exception as exc:  # noqa: BLE001
            if h.fail_open:
                _engine_log.warning("hook %s.%s 무시됨(fail-open): %s", type(h).__name__, method, exc)
                continue
            raise  # Guardrail 등 fail-close: 실행 중단
        if out is not None:
            value = out
    return value


# ── Node 래핑 ──────────────────────────────────────────────────────────
def wrap_node(fn: Callable[..., Any], node: str, get_hooks: Callable[[], list[Hook]], is_async: bool) -> Callable[..., Any]:
    """Node 함수를 Hook 발화로 감싼다. get_hooks는 호출 시점에 최신 목록을 반환."""

    if is_async:
        @functools.wraps(fn)
        async def awrapped(state: Any, *a: Any, **kw: Any) -> Any:
            hooks = get_hooks()
            ctx = get_context()
            _before(hooks, "before_node", node, state, ctx)
            try:
                result = await fn(state, *a, **kw)
                _after(hooks, "after_node", node, state, result, ctx)
                return result
            except BaseException as exc:
                if not is_control_flow(exc):  # interrupt는 실패가 아님 → error Hook 제외
                    _error(hooks, "on_node_error", node, state, exc, ctx)
                raise
            finally:
                _finally(hooks, "finally_node", node, state, ctx)

        return awrapped

    @functools.wraps(fn)
    def wrapped(state: Any, *a: Any, **kw: Any) -> Any:
        hooks = get_hooks()
        ctx = get_context()
        _before(hooks, "before_node", node, state, ctx)
        try:
            result = fn(state, *a, **kw)
            _after(hooks, "after_node", node, state, result, ctx)
            return result
        except BaseException as exc:
            if not is_control_flow(exc):  # interrupt는 실패가 아님 → error Hook 제외
                _error(hooks, "on_node_error", node, state, exc, ctx)
            raise
        finally:
            _finally(hooks, "finally_node", node, state, ctx)

    return wrapped
