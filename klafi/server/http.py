"""HTTP Layer — FastAPI 어댑터 (요구사항 §19, F13).

이 파일이 유일한 FastAPI 의존 지점이다. AgentServer(런타임)는 FastAPI를 모른다.
→ 다른 Runtime으로 교체해도 Agent 코드/런타임은 영향 없음.

Endpoints:
  GET  /health                     (API-07)
  GET  /agents                     목록
  GET  /agents/{id}                Metadata (API-08)
  POST /agents/{id}/invoke         (API-01)
  POST /agents/{id}/stream         (API-02, NDJSON)
  GET  /openapi.json, /docs        OpenAPI 자동생성 (API-09, FastAPI 기본 제공)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from klafi.core.context import ExecutionContext
from .concurrency import install_concurrency_limit
from .registry import AgentNotFound, AgentServer

# Authentication Adapter (API-10): Request → security_context dict | None. 기본 미적용.
AuthAdapter = Callable[[Request], "dict[str, Any] | None"]

_log = logging.getLogger("klafi.server")


class InvokeRequest(BaseModel):
    input: Any = None  # None이면 Resume(Checkpoint 이후 재개)
    thread_id: str | None = None

    model_config = {
        "json_schema_extra": {
            # Swagger "Try it out" 기본 body. input 형태는 에이전트 state 스키마에 따라 다르다.
            "examples": [
                {
                    "summary": "대화형(messages) — support/triage/schedule",
                    "value": {
                        "input": {"messages": [{"role": "user", "content": "A-100 주문 언제 도착해?"}]},
                        "thread_id": "t1",
                    },
                },
                {
                    "summary": "구조화 입력 — stock(주식 매수, HITL)",
                    "value": {"input": {"symbol": "AAPL", "quantity": 10}, "thread_id": "buy1"},
                },
                {
                    "summary": "Resume(체크포인트 이후 재개) — input 생략",
                    "value": {"input": None, "thread_id": "t1"},
                },
            ]
        }
    }


class ResumeRequest(BaseModel):
    thread_id: str
    decision: Any = None  # HITL 승인 결정 {approved, comment, ...} 또는 임의 resume 값

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "승인",
                    "value": {"thread_id": "buy1", "decision": {"approved": True, "decided_by": "manager"}},
                },
                {
                    "summary": "반려(사유 포함)",
                    "value": {"thread_id": "buy1", "decision": {"approved": False, "comment": "리스크 한도"}},
                },
            ]
        }
    }


def _result_body(ctx: ExecutionContext, result: Any) -> dict[str, Any]:
    state = ctx.state
    if state == "CREATED":  # 정책 미설정 시 상태 미전이 → 성공은 COMPLETED로 표기
        state = "COMPLETED"
    return {"execution_id": ctx.execution_id, "state": state, "result": _encode(result)}


def _context(req: Request, agent_spec: Any, thread_id: str | None, auth: AuthAdapter | None) -> ExecutionContext:
    sec = auth(req) if auth else None
    return ExecutionContext.new(
        agent_id=agent_spec.id,
        agent_version=agent_spec.version,
        project_id=agent_spec.project,
        session_id=thread_id,  # thread_id = session (Resume 키)
        request_id=req.headers.get("x-request-id"),
        user_id=(sec or {}).get("user_id"),
        security_context=sec or {},
    )


def _encode(obj: Any) -> Any:
    # LangGraph 결과에 비직렬화 타입이 섞여도 안전하게 (message 객체 등).
    # HITL interrupt는 value(action/payload/approver...)를 구조 그대로 노출 → 프론트가 양식을 범용 렌더.
    def _default(o: Any) -> Any:
        if type(o).__name__ == "Interrupt":
            return {"value": getattr(o, "value", None)}
        return str(o)

    return json.loads(json.dumps(obj, default=_default, ensure_ascii=False))


def _error_body(ctx: ExecutionContext, exc: Exception) -> "tuple[int, dict[str, Any]]":
    """실패를 원인별 (status, body)로 분류. 가드레일·권한 차단은 서버 장애(5xx)가 아니라
    클라이언트측 정책 거부(4xx)다 — 500으로 두면 에러 대시보드·알람이 오탐한다.
    invoke/resume는 status로 HTTP 응답을, stream은 body로 에러 청크를 만든다."""
    from klafi.core.exceptions import PermissionDeniedError, ViolationError

    if isinstance(exc, ViolationError):  # GuardrailViolationError 등 — 정책 위반으로 차단(fail-close)
        status, state = 403, "BLOCKED"
    elif isinstance(exc, (PermissionDeniedError, PermissionError)):  # 권한 없음
        status, state = 403, "DENIED"
    else:
        status, state = 500, "FAILED"  # 예기치 못한 서버 오류
    if status >= 500:  # 예상된 4xx 차단은 조용히(가드레일은 이미 WARNING). 진짜 오류만 ERROR+스택.
        _log.error("agent.execution_error execution_id=%s", ctx.execution_id, exc_info=exc)
    body: dict[str, Any] = {"execution_id": ctx.execution_id, "state": state, "error": str(exc)}
    code = getattr(exc, "error_code", None)
    if code:
        body["error_code"] = code
    detail = getattr(exc, "context", None)
    if detail:  # stage/guard 등 (KlafiException.context)
        body["detail"] = {k: str(v) for k, v in detail.items()}
    return status, body


def _fail_response(ctx: ExecutionContext, exc: Exception) -> JSONResponse:
    status, body = _error_body(ctx, exc)
    return JSONResponse(status_code=status, content=body)


def _msg_text(msg: Any) -> str:
    """AIMessageChunk의 토큰 델타 텍스트. content가 str/블록리스트 모두 지원."""
    c = getattr(msg, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):  # anthropic content blocks
        return "".join(b.get("text", "") for b in c if isinstance(b, dict))
    return ""


def _stream_line(ctx: ExecutionContext, item: Any) -> "str | None":
    """stream_mode=["updates","messages"] 항목 → NDJSON 한 줄.
    messages=LLM 토큰(token), updates=노드 결과(chunk, interrupt 감지·최종 상태용)."""
    if isinstance(item, tuple) and len(item) == 2 and item[0] in ("updates", "messages"):
        mode, payload = item
    else:  # 단일 모드 방어
        mode, payload = "updates", item
    if mode == "messages":
        msg = payload[0] if isinstance(payload, tuple) else payload
        if "AIMessage" not in type(msg).__name__:  # 사람/툴 메시지는 토큰으로 흘리지 않는다
            return None
        text = _msg_text(msg)
        if not text:  # tool_call만 있는 청크 등
            return None
        return json.dumps({"execution_id": ctx.execution_id, "token": text}, ensure_ascii=False) + "\n"
    return json.dumps({"execution_id": ctx.execution_id, "chunk": _encode(payload)}, ensure_ascii=False) + "\n"


def create_app(
    server: AgentServer,
    auth: AuthAdapter | None = None,
    title: str = "KLAFI Agent Server",
    max_concurrency: int | None = None,
) -> FastAPI:
    """max_concurrency: 서버 전역 동시 실행 상한. 초과 요청은 429로 즉시 거절(백프레셔)."""
    app = FastAPI(title=title)

    # 동시 실행 상한은 ASGI 미들웨어 한 곳에서 관리한다(엔드포인트마다 복붙 금지).
    install_concurrency_limit(app, max_concurrency)

    @app.exception_handler(AgentNotFound)
    async def _not_found(_req: Request, exc: AgentNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error_code": exc.error_code, "message": exc.message})

    @app.get("/health")
    def health() -> dict[str, str]:  # API-07
        return {"status": "ok", "agents": str(len(server.ids()))}

    @app.get("/agents")
    def list_agents() -> list[dict[str, Any]]:
        return server.list_metadata()

    @app.get("/agents/{agent_id}")
    def agent_meta(agent_id: str) -> dict[str, Any]:  # API-08
        return server.metadata(agent_id)

    @app.post("/agents/{agent_id}/invoke")
    def invoke(agent_id: str, body: InvokeRequest, request: Request) -> JSONResponse:  # API-01
        agent = server.get(agent_id)
        ctx = _context(request, agent.spec, body.thread_id, auth)
        try:
            result = agent.invoke(body.input, context=ctx, thread_id=body.thread_id)
        except Exception as exc:  # noqa: BLE001 — 실패도 execution_id로 Trace 상관관계 확보
            return _fail_response(ctx, exc)  # 가드레일·권한 차단은 4xx, 그 외 500
        return JSONResponse(content=_result_body(ctx, result))

    @app.post("/agents/{agent_id}/resume")
    def resume(agent_id: str, body: ResumeRequest, request: Request) -> JSONResponse:  # API-06
        from langgraph.types import Command

        agent = server.get(agent_id)
        ctx = _context(request, agent.spec, body.thread_id, auth)
        try:
            result = agent.invoke(Command(resume=body.decision), context=ctx, thread_id=body.thread_id)
        except Exception as exc:  # noqa: BLE001 — invoke와 동일하게 원인별 status
            return _fail_response(ctx, exc)
        return JSONResponse(content=_result_body(ctx, result))

    def _make_stream(agent_id: str, request: Request, thread_id: str | None, stream_input: Any) -> Any:
        """스트리밍 응답 조립 — /stream(입력)과 /resume/stream(Command resume)이 공유.

        동시성 슬롯은 install_concurrency_limit 미들웨어가 StreamingResponse 전송 완료까지 잡는다.
        """
        agent = server.get(agent_id)
        ctx = _context(request, agent.spec, thread_id, auth)

        # async 제너레이터: 이벤트 루프에서 직접 iteration → ContextVar 안정
        # (sync 제너레이터는 starlette가 iteration마다 context를 복사해 token reset이 깨진다)
        async def gen() -> Any:
            # updates=노드/interrupt, messages=LLM 토큰 → 진짜 토큰 단위 스트리밍
            try:
                async for item in agent.astream(
                    stream_input, context=ctx, thread_id=thread_id, stream_mode=["updates", "messages"]
                ):
                    line = _stream_line(ctx, item)
                    if line:
                        yield line
            except Exception as exc:  # noqa: BLE001 — 헤더는 이미 전송됨(status 변경 불가) → 에러 청크로 통지
                _, err = _error_body(ctx, exc)  # invoke와 동일 분류(+5xx는 ERROR 로깅)
                yield json.dumps({"execution_id": ctx.execution_id, "error": err}, ensure_ascii=False) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.post("/agents/{agent_id}/stream")
    async def stream(agent_id: str, body: InvokeRequest, request: Request) -> Any:  # API-02
        return _make_stream(agent_id, request, body.thread_id, body.input)

    @app.post("/agents/{agent_id}/resume/stream")
    async def resume_stream(agent_id: str, body: ResumeRequest, request: Request) -> Any:  # HITL 재개도 스트리밍
        from langgraph.types import Command

        return _make_stream(agent_id, request, body.thread_id, Command(resume=body.decision))

    return app
