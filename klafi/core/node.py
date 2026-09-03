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
* visibility — "external"(기본) | "internal". internal 노드의 LLM 토큰·updates 는 스트림으로 **전달되지 않는다**
  (라우팅·판정 등 호출자에게 보일 필요 없는 내부 호출). 상태(state)까지 숨기지는 않으므로 내부 판단은
  messages 가 아닌 별도 키에 쓴다.
* output — pydantic 스키마(또는 ("state_key", Schema)). 노드가 돌려준 값 중 그 키(또는 유일한 dict/모델 값)를
  스키마로 **검증·강제**하고(툴의 output_schema 와 같은 계약), 스트림에서는 토큰 대신 완료 시
  ("structured", {node, key, data}) 한 청크로 전달한다. 스키마는 /agents/{id} 메타데이터(nodes)에 노출된다.
  주의: 구조화 출력의 원문 AIMessage 를 messages 에 넣지 말 것(Anthropic 은 tool 호출로 구현 → tool_result
  없는 tool_use 가 이력에 남아 400). 파싱된 객체만 전용 키에 저장한다.

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
    visibility: str = "external",  # "external" | "internal" — internal 은 스트림 미전달
    output: Any = None,     # pydantic 스키마 | ("state_key", Schema) — 구조화 출력 계약
) -> Any:
    """노드 함수에 가드레일·미들웨어를 붙인다. **name 은 필수**.

        @klafi_node("plan")                                  # 위치 인자로 이름
        @klafi_node("plan", before=[require_login, no_secrets])   # 미들웨어 + 가드레일 혼합
        @klafi_node("route", visibility="internal")          # 호출자에게 안 보이는 내부 판단
        @klafi_node("extract", output=Report)                # 구조화 출력(검증·스키마 노출·스트림 1회 전달)
    """
    from .exceptions import AgentExecutionException

    # @klafi_node("name") — 위치 인자로 이름을 준 경우
    if isinstance(fn, str):
        name, fn = fn, None
    if not name:
        raise AgentExecutionException("@klafi_node 에는 name 이 필수입니다 (예: @klafi_node(\"plan\"))")
    if visibility not in ("external", "internal"):
        raise AgentExecutionException(f"@klafi_node('{name}') visibility 는 'external' | 'internal' — {visibility!r}")
    contract = _output_contract(name, output)
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
                result = _coerce_output(node_name, result, contract)  # 계약 검증 → 그 다음 after(마스킹 등)
                return await aapply(pipe_after, result, ctx, "output")

            return _stamp(awrapped, node_name, guarded=bool(pipe_after), visibility=visibility, output=contract)

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
            result = _coerce_output(node_name, result, contract)
            return apply(pipe_after, result, ctx, "output")

        return _stamp(wrapped, node_name, guarded=bool(pipe_after), visibility=visibility, output=contract)

    return deco(fn) if callable(fn) else deco


def _stamp(
    wrapper: Callable, node_name: str, *, guarded: bool = False, visibility: str = "external", output: Any = None
) -> Callable:
    """KlafiGraph 가 노드 강제 검사·스트림 게이팅·메타데이터에 쓰는 마커."""
    wrapper.__klafi_node__ = True  # type: ignore[attr-defined]
    wrapper.__klafi_node_name__ = node_name  # type: ignore[attr-defined]
    wrapper.__klafi_after__ = guarded  # type: ignore[attr-defined]  # after 파이프라인 보유 → 스트림 원문 토큰 억제
    wrapper.__klafi_visibility__ = visibility  # type: ignore[attr-defined]
    wrapper.__klafi_output__ = output  # type: ignore[attr-defined]  # (state_key | None, Schema) | None
    return wrapper


def _output_contract(node: str, output: Any) -> tuple[str | None, Any] | None:
    """output 인자 정규화 → (state_key | None, Schema). Schema 는 pydantic 모델 클래스여야 한다."""
    if output is None:
        return None
    key, schema = output if isinstance(output, tuple) else (None, output)
    if not (isinstance(schema, type) and hasattr(schema, "model_validate")):
        from .exceptions import AgentExecutionException

        raise AgentExecutionException(f"@klafi_node('{node}') output 은 pydantic 모델 클래스여야 합니다: {schema!r}")
    return key, schema


def _coerce_output(node: str, result: Any, contract: tuple[str | None, Any] | None) -> Any:
    """output 계약 적용: 반환 dict 의 대상 값을 Schema 인스턴스로 검증·강제한다.

    대상 키는 명시(("key", Schema)) 또는 자동(messages 제외, dict/모델 값이 정확히 하나). 검증 실패는 fail-close.
    dict 가 아닌 반환(Command 등)은 건드리지 않는다.
    """
    if contract is None or not isinstance(result, dict):
        return result
    from .exceptions import AgentExecutionException, ValidationError

    key, schema = contract
    if key is None:
        cands = [k for k, v in result.items() if k != "messages" and (isinstance(v, dict) or hasattr(v, "model_dump"))]
        if len(cands) != 1:
            raise AgentExecutionException(
                f"노드 '{node}' output={schema.__name__}: 스키마 객체를 담은 상태 키를 하나 찾을 수 없습니다"
                f"(후보 {cands}). output=(\"key\", {schema.__name__}) 로 키를 지정하세요"
            )
        key = cands[0]
    if key not in result:
        raise AgentExecutionException(f"노드 '{node}' 반환값에 output 키 '{key}' 가 없습니다")
    value = result[key]
    if isinstance(value, schema):
        return result
    try:
        obj = schema.model_validate(value.model_dump() if hasattr(value, "model_dump") else value)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"노드 '{node}' 출력이 {schema.__name__} 스키마에 맞지 않습니다: {exc}", node=node) from exc
    return {**result, key: obj}


def _is_graph(obj: Any) -> bool:
    from .graph import KlafiGraph

    return isinstance(obj, KlafiGraph)
