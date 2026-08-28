"""Observability — OpenTelemetry Tracing (요구사항 §16, F11 / OBS-01~07).

DoD: Execution ID → Agent → Node → Tool → Model → Error를 하나의 Trace로 추적.

구조: TracingHook이 Agent/Node span을 자동 생성하고, Tool/Model 및 커스텀 구간은
span() 헬퍼로 감싼다. span()은 OTel의 현재 context에 자동 중첩되므로, Node 안에서
호출된 Tool/Model span은 그 Node span의 자식이 된다.

원칙:
- LangGraph Native: 실행 자체는 그대로 두고 Hook으로 관측만 부착한다.
- Fail-Open(§25): Provider 미설정 시 OTel no-op tracer → 무해. Hook도 fail_open=True.
- Correlation(§16): execution/trace/agent/session/thread/request id를 span 속성으로.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, set_span_in_context

from klafi.core.context import ExecutionContext
from klafi.core.hook import Hook, is_control_flow

_TRACER_NAME = "klafi"


def setup_tracing(
    exporter: Any = None, service_name: str = "klafi", simple: bool = False
) -> Any:
    """TracerProvider 설치. exporter로 Loki/Tempo/Langfuse(OTLP) 등을 연결 (OBS-12~14).

    OTel은 프로세스당 Provider를 한 번만 설정하므로, 두 번째 호출 시 processor만 추가한다.
    """
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        provider = current  # 이미 설치됨 → 재사용
    else:
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        trace.set_tracer_provider(provider)
    if exporter is not None:
        proc = SimpleSpanProcessor(exporter) if simple else BatchSpanProcessor(exporter)
        provider.add_span_processor(proc)
    return provider


def _corr(ctx: ExecutionContext | None) -> dict[str, Any]:
    """필수 Correlation ID를 span 속성으로 (§16)."""
    if ctx is None:
        return {}
    return {
        "klafi.execution_id": ctx.execution_id,
        "klafi.trace_id": ctx.trace_id,
        "klafi.agent_id": ctx.agent_id,
        "klafi.agent_version": ctx.agent_version,
        "klafi.project_id": ctx.project_id,
        "klafi.session_id": ctx.session_id,
        "klafi.thread_id": ctx.session_id or ctx.execution_id,
        "klafi.request_id": ctx.request_id,
        "klafi.user_id": ctx.user_id,
        "klafi.tenant_id": ctx.tenant_id,
    }


def _set_attrs(sp: Any, attrs: dict[str, Any]) -> None:
    for k, v in attrs.items():
        if v is not None:
            sp.set_attribute(k, v)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Any]:
    """Tool/Model/커스텀 구간 span. 현재 Node span의 자식으로 자동 중첩 (OBS-04/05).

    Model Gateway/Tool Framework(F09)가 이 헬퍼를 호출하면 Token/Cost 속성을 얹어
    Model Usage Metric(OBS-08/11)까지 확장된다.
    """
    tracer = trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(name) as sp:
        _set_attrs(sp, attrs)
        try:
            yield sp
        except BaseException as exc:
            if not is_control_flow(exc):  # interrupt(HITL)는 오류가 아님
                sp.record_exception(exc)
                sp.set_status(Status(StatusCode.ERROR))
            raise


class TracingHook(Hook):
    """Agent/Node span 자동 생성 (OBS-02/03). Business Exception을 span에 연결 (OBS-07)."""

    priority = 5  # LoggingHook(10)보다 바깥에서 감싼다
    fail_open = True  # 관측 실패가 업무를 막지 않는다

    def __init__(self) -> None:
        self._spans: dict[tuple, tuple[Any, object]] = {}

    def _tracer(self) -> Any:
        return trace.get_tracer(_TRACER_NAME)

    def _start(self, key: tuple, name: str, attrs: dict[str, Any]) -> None:
        sp = self._tracer().start_span(name)
        _set_attrs(sp, attrs)
        token = otel_context.attach(set_span_in_context(sp))
        self._spans[key] = (sp, token)

    def _mark_error(self, key: tuple, exc: BaseException) -> None:
        item = self._spans.get(key)
        if item:
            sp, _ = item
            sp.record_exception(exc)
            sp.set_status(Status(StatusCode.ERROR))

    def _end(self, key: tuple) -> None:
        item = self._spans.pop(key, None)
        if not item:
            return
        sp, token = item
        otel_context.detach(token)
        sp.end()

    # Agent span (before_agent~finally_agent은 invoke 프레임 내 연속 호출 → 안전)
    def before_agent(self, input: Any, ctx: ExecutionContext | None) -> None:
        eid = ctx.execution_id if ctx else "-"
        self._start(("agent", eid), f"agent.{ctx.agent_id if ctx else '?'}", _corr(ctx))

    def on_agent_error(self, input: Any, exc: BaseException, ctx: ExecutionContext | None) -> None:
        self._mark_error(("agent", ctx.execution_id if ctx else "-"), exc)

    def finally_agent(self, input: Any, ctx: ExecutionContext | None) -> None:
        self._end(("agent", ctx.execution_id if ctx else "-"))

    # Node span (before_node~finally_node은 wrap_node 프레임 내 연속 호출 → 안전)
    def _nkey(self, node: str, state: Any, ctx: ExecutionContext | None) -> tuple:
        return ("node", ctx.execution_id if ctx else "-", node, id(state))

    def before_node(self, node: str, state: Any, ctx: ExecutionContext | None) -> None:
        attrs = {"klafi.node": node, **_corr(ctx)}
        self._start(self._nkey(node, state, ctx), f"node.{node}", attrs)

    def on_node_error(self, node: str, state: Any, exc: BaseException, ctx: ExecutionContext | None) -> None:
        self._mark_error(self._nkey(node, state, ctx), exc)

    def finally_node(self, node: str, state: Any, ctx: ExecutionContext | None) -> None:
        self._end(self._nkey(node, state, ctx))
