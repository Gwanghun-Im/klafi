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

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, set_span_in_context

from klafi.core.context import ExecutionContext
from klafi.core.hook import Hook, is_control_flow

_TRACER_NAME = "klafi"
_log = logging.getLogger("klafi.observability")


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


# ── OTLP exporter 결정 (Intelligence 원격 설정 > 로컬 config > 표준 env > 없음) ──
# 어떤 실패도 부팅을 막지 않는다(§25 fail-open — 관측은 fail-fast 대상이 아니다).
# 시크릿(헤더 값, URL 의 userinfo·query 토큰)은 어떤 로그에도 남기지 않는다.

_otlp_attached = False  # from_config 재호출(테스트·멀티앱) 시 processor 중복 부착 방지


def _safe_url(url: str) -> str:
    """로그용 URL — userinfo(`pk:sk@`)·query(`?api-key=`) 를 벗겨 시크릿 누출을 막는다."""
    try:
        from urllib.parse import urlsplit

        u = urlsplit(url)
        host = u.netloc.rsplit("@", 1)[-1]  # userinfo 제거, 포트 유지
        return f"{u.scheme}://{host}{u.path}"
    except Exception:  # noqa: BLE001
        return "<url>"


def _intelligence_otlp() -> "dict | None":
    """INTELLIGENCE_MODE=ON 이면 사내 Intelligence 서비스에서 OTLP 설정을 조회. 실패 시 None.

    계약(사내 스펙 확정 시 이 함수만 고치면 된다):
      GET {INTELLIGENCE_ENDPOINT}{INTELLIGENCE_CONFIG_PATH:-/v1/observability/otlp}
      → 200 {"endpoint": "<full traces URL(…/v1/traces)>", "headers": {...}?}
    인증: INTELLIGENCE_TOKEN → Authorization: Bearer. 타임아웃: INTELLIGENCE_TIMEOUT(기본 3s),
    재시도 없음 — 부팅 경로이고 로컬 폴백이 재시도를 대신한다.
    """
    import os

    mode = os.environ.get("INTELLIGENCE_MODE", "").strip().upper()
    if mode != "ON":
        if mode and mode != "OFF":  # 'true'/'1' 오타가 조용히 꺼지지 않게
            _log.warning("INTELLIGENCE_MODE=%r 은 무시됨 — 'ON' 만 활성입니다", mode)
        return None
    try:  # URL 조립·Request 생성까지 전부 안에서 — 어떤 실패든 이 함수는 None(로컬 폴백 보장)
        import json
        import urllib.request

        base = os.environ.get("INTELLIGENCE_ENDPOINT", "").strip().rstrip("/")
        if not base:
            _log.warning("INTELLIGENCE_MODE=ON 이지만 INTELLIGENCE_ENDPOINT 미설정 — 로컬 OTLP 로 폴백")
            return None
        url = base + os.environ.get("INTELLIGENCE_CONFIG_PATH", "/v1/observability/otlp")
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        if tok := os.environ.get("INTELLIGENCE_TOKEN"):
            req.add_header("Authorization", f"Bearer {tok}")
        try:
            timeout = float(os.environ.get("INTELLIGENCE_TIMEOUT", "3"))
        except ValueError:
            timeout = 3.0
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 — 운영자 설정 URL
            data = json.loads(r.read())
        return {"endpoint": data["endpoint"], "headers": dict(data.get("headers") or {}), "source": "intelligence"}
    except Exception as exc:  # noqa: BLE001 — fail-open: 사유는 남기되 값은 안 남긴다
        _log.warning("Intelligence OTLP 설정 조회 실패(%s) — 로컬 OTLP 로 폴백", type(exc).__name__)
        return None


def _local_otlp(cfg: Any) -> "dict | None":
    """로컬 설정: config observability.otlp.endpoint > OTel 표준 env(SDK 가 직접 해석)."""
    import os

    otlp = cfg.get("observability.otlp") if cfg is not None else None
    if isinstance(otlp, dict) and otlp.get("endpoint"):
        headers = {k: v for k, v in (otlp.get("headers") or {}).items() if v}  # ${VAR:} 빈 값 drop
        return {"endpoint": otlp["endpoint"], "headers": headers, "source": "local-config"}
    if os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return {"endpoint": None, "headers": None, "source": "otel-env"}  # 무인자 생성 → SDK 표준 처리
    return None


def _make_exporter(conf: dict) -> Any:
    """conf → OTLPSpanExporter. 미설치·생성 실패는 None(호출부가 다음 소스로 폴백)."""
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError:
        _log.warning(
            "OTLP 설정(source=%s)이 있으나 exporter 미설치 — pip install 'klafi[otlp]' 후 송출됩니다",
            conf["source"],
        )
        return None
    try:
        if conf["endpoint"] is None:  # otel-env: OTEL_EXPORTER_OTLP_* 를 exporter 가 직접 읽는다
            return OTLPSpanExporter()
        return OTLPSpanExporter(endpoint=conf["endpoint"], headers=conf["headers"] or None)
    except Exception as exc:  # noqa: BLE001 — 기형 endpoint 등: 이 소스만 버리고 다음 소스로
        _log.warning("OTLP exporter 생성 실패(source=%s, %s) — 다음 소스로", conf["source"], type(exc).__name__)
        return None


def resolve_otlp_exporter(cfg: Any = None) -> Any:
    """trace 송출 대상 결정 — Intelligence(ON) > 로컬 config > 표준 env > 없음(계측만).

    from_config 가 setup_tracing(exporter=...) 에 꽂는다. 프로세스당 1회만 부착(재호출 시 None —
    BatchSpanProcessor 중복으로 span 이 이중 송출되는 사고 방지). 어떤 실패도 부팅을 막지 않는다.
    """
    global _otlp_attached
    try:
        if _otlp_attached:
            _log.info("trace export: OTLP 이미 부착됨 — 건너뜀")
            return None
        for conf in (_intelligence_otlp(), _local_otlp(cfg)):
            if conf is None:
                continue
            exporter = _make_exporter(conf)
            if exporter is not None:
                _otlp_attached = True
                _log.info(
                    "trace export 활성: source=%s endpoint=%s headers=%d개",
                    conf["source"],
                    _safe_url(conf["endpoint"]) if conf["endpoint"] else "OTEL_* env",
                    len(conf["headers"] or {}),
                )
                return exporter
        _log.info("trace export 미설정 — 계측만 수행(span 송출 없음)")
        return None
    except Exception as exc:  # noqa: BLE001 — 최종 방어선: 어떤 경우에도 부팅 계속
        _log.warning("OTLP exporter 구성 실패(%s) — 계측만 수행", type(exc).__name__)
        return None


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
        "klafi.thread_id": ctx.thread_id or ctx.session_id or ctx.execution_id,
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
        key = ("agent", eid)
        self._start(key, f"agent.{ctx.agent_id if ctx else '?'}", _corr(ctx))
        otel_tid = self._spans[key][0].get_span_context().trace_id
        if ctx is not None and otel_tid:
            # 실제 OTel trace id 로 통일 — 로그·span·응답이 서로 다른 'trace id' 를 보이지 않게 한다.
            ctx.trace_id = format(otel_tid, "032x")
            self._spans[key][0].set_attribute("klafi.trace_id", ctx.trace_id)

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
