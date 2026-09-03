"""Tool Framework (요구사항 §14.2, F09 / TOL-01~10).

평범한 함수에 Enterprise 실행 기능을 얹은 표준 Tool.
- 표준 Interface + Metadata (TOL-01/02)
- Timeout/Retry (TOL-04/05): ExecutionPolicy 재사용
- 실행 Logging + 사용량 Metric (TOL-07/10): span() 재사용
- 권한 (TOL-06): ExecutionContext.security_context 검사 (최소권한, SEC-10)
- Input/Output Validation (TOL-08/09): pydantic 스키마
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from klafi.core.context import get_context
from klafi.core.exceptions import ToolException, ToolPermissionError, ToolValidationError
from klafi.observability.tracing import span

_log = logging.getLogger("klafi.tool")


@dataclass
class ToolMetadata:
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)


class Tool:
    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str = "",
        input_schema: Any = None,  # pydantic BaseModel 서브클래스
        output_schema: Any = None,
        policy: Any = None,  # ExecutionPolicy
        required_permission: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        import inspect

        if inspect.iscoroutinefunction(fn):
            # Tool.run 은 sync 라 코루틴 함수를 그대로 감싸면 '<coroutine object>' 문자열이 결과로 나간다.
            raise ToolException(
                f"tool '{name or fn.__name__}': async 함수는 @tool 로 감쌀 수 없습니다 — sync 함수로 만들거나 "
                "(외부 async 도구는) from_langchain_tool(...) 브리지를 쓰세요",
                tool=name or fn.__name__,
            )
        self._fn = fn
        self.metadata = ToolMetadata(name or fn.__name__, description or (fn.__doc__ or "").strip(), tags or [])
        self._input_schema = input_schema
        self._output_schema = output_schema
        self._policy = policy
        self._permission = required_permission

    @property
    def name(self) -> str:
        return self.metadata.name

    def _check_permission(self) -> None:
        if self._permission is None:
            return
        ctx = get_context()
        granted = (ctx.security_context.get("permissions", []) if ctx else [])
        if self._permission not in granted:  # 최소권한: 없으면 거부
            raise ToolPermissionError(
                f"tool '{self.name}' 권한 없음: {self._permission}",
                tool=self.name,
                required=self._permission,
            )

    def _validate_input(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        from pydantic import BaseModel

        schema = self._input_schema
        # pydantic 모델일 때만 KLAFI 검증. MCP 도구처럼 args_schema 가 JSON-schema dict 면
        # (호출 불가) 검증은 생략하고 도구 자체 검증에 맡긴다 — 단, 스키마는 LLM 바인딩용으로 보존한다.
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            return kwargs
        try:
            model = schema(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ToolValidationError(f"tool '{self.name}' 입력 검증 실패: {exc}", tool=self.name) from exc
        return model.model_dump()

    def _validate_output(self, result: Any) -> Any:
        if self._output_schema is None:
            return result
        try:
            return self._output_schema(**result) if isinstance(result, dict) else self._output_schema(result)
        except Exception as exc:  # noqa: BLE001
            raise ToolValidationError(f"tool '{self.name}' 출력 검증 실패: {exc}", tool=self.name) from exc

    def run(self, **kwargs: Any) -> Any:
        from klafi.core.hook import _error, _transform, active_hooks
        from klafi.events import EventType, emit  # lazy: 순환 방지

        hooks = active_hooks()
        ctx = get_context()
        with span(f"tool.{self.name}") as sp:
            sp.set_attribute("klafi.tool", self.name)  # TOL-07/10
            emit(EventType.ToolStarted, tool=self.name)
            try:
                # 권한·입력검증도 span 안에서 — 권한 거부(가장 감사해야 할 케이스)가
                # trace·이벤트에 흔적을 남기도록 (TOL-06 은 감사 사각지대가 되면 안 된다).
                self._check_permission()  # TOL-06
                kwargs = self._validate_input(kwargs)  # TOL-08
                # HOK-04: 인자 경계 가드레일 — 반환값이 kwargs를 교체(마스킹)한다.
                # 마스킹 후 재검증은 하지 않는다(이미 검증된 값의 문자열 리프만 바뀜).
                kwargs = _transform(hooks, "before_tool", kwargs, lambda k: (self.name, k, ctx))
                result = self._apply_policy(lambda: self._fn(**kwargs))  # TOL-04/05
                result = self._validate_output(result)  # TOL-09
            except Exception as exc:  # noqa: BLE001
                _error(hooks, "on_tool_error", self.name, kwargs, exc, ctx)
                emit(EventType.ToolFailed, tool=self.name, error=str(exc))
                raise
            # 반환 경계 가드레일 — 반환값이 결과를 교체한다(ToolNode 경유 시 ToolMessage까지 반영).
            result = _transform(hooks, "after_tool", result, lambda r: (self.name, kwargs, r, ctx), reverse=True)
            sp.set_attribute("klafi.tool_ok", True)
            emit(EventType.ToolCompleted, tool=self.name)
            _log.info("tool.call tool=%s ok", self.name)
            return result

    __call__ = run

    def as_langchain(self) -> Any:
        """LangChain StructuredTool로 변환 (bind_tools/ToolNode에 사용).

        ToolNode가 실행할 때도 self.run을 거치므로 권한·검증·Timeout·Hook·Metric이 그대로 적용된다.
        인자 스키마는 input_schema가 있으면 그것을, 없으면 원함수 시그니처에서 추론한다.
        """
        from langchain_core.tools import StructuredTool

        name = self.name
        description = self.metadata.description or self.name
        run = self.run

        def _route(**kw: Any) -> Any:  # 실행은 항상 KLAFI run 경유
            return run(**kw)

        if self._input_schema is not None:
            return StructuredTool.from_function(
                func=_route, name=name, description=description, args_schema=self._input_schema
            )
        # input_schema 없으면 원함수에서 스키마 추론 후 실행만 run으로 교체
        st = StructuredTool.from_function(self._fn, name=name, description=description)
        st.func = _route
        return st

    def _apply_policy(self, fn: Callable[[], Any]) -> Any:
        if self._policy is None:
            return fn()
        from klafi.runtime.engine import run_sync  # lazy

        return run_sync(fn, self._policy, lambda _s: None)


def tool(
    name: str | None = None,
    *,
    description: str = "",
    input_schema: Any = None,
    output_schema: Any = None,
    policy: Any = None,
    required_permission: str | None = None,
    tags: list[str] | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """함수를 Tool로 감싸는 데코레이터."""

    def wrap(fn: Callable[..., Any]) -> Tool:
        return Tool(
            fn,
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            policy=policy,
            required_permission=required_permission,
            tags=tags,
        )

    return wrap


def to_langchain_tools(tools: list[Any]) -> list[Any]:
    """KLAFI Tool·Skill 목록을 LangChain tool 목록으로 변환.

    Skill이 섞여 있으면 툴만 꺼낸다(지침은 bind_skills가 처리).
    변환 후에도 실행은 Tool.run을 거치므로 권한·검증·Timeout·Hook·Metric이 유지된다.
    """
    from .skill import Skill

    flat, _ = Skill.flatten(tools)
    return [t.as_langchain() if isinstance(t, Tool) else t for t in flat]
