"""ExecutionRecorder — 실행 타임라인 in-memory 기록 (Playground 트레이스 뷰어, GET /agents/{id}/executions/{eid}).

"이 답이 어떻게 나왔나"를 LangSmith 없이 보여주기 위한 최소 저장소다. 이미 있는 신호만 모은다:
  · 노드/모델/툴 소요시간 — Hook(before/after/finally)
  · 토큰·비용 — ModelCalled 이벤트
  · 가드레일 판정 — GuardrailViolation 이벤트(stage·guard·severity·reason)
  · 승인 — ApprovalRequested/Completed 이벤트
프로세스 전역 싱글턴(RECORDER)이 전역 훅 + EventBus 구독으로 붙는다. 용량 초과 시 오래된 실행부터 버린다.
ponytail: in-memory 만 — 여러 워커·재시작 간 보존이 필요하면 get() 결과를 그대로 외부 저장소에 쓰면 된다.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

from klafi.core.context import ExecutionContext
from klafi.core.hook import Hook, _GLOBAL_HOOKS, register_hook
from klafi.events.bus import Event, EventType, subscribe


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExecutionRecorder(Hook):
    priority = 20  # 관측 훅 — 가드레일(1)·트레이싱(5)·로깅(10) 안쪽
    fail_open = True

    def __init__(self, capacity: int = 500) -> None:
        self._capacity = capacity
        self._runs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._open: dict[tuple, float] = {}  # (eid, kind, name[, id]) → 시작 perf_counter
        self._errors: dict[tuple, str] = {}
        self._lock = threading.Lock()

    # ── 조회 ──
    def get(self, execution_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(execution_id)
            if run is None:
                return None
            events = list(run["events"])
        totals = {
            "nodes": sum(1 for e in events if e["kind"] == "node"),
            "tools": sum(1 for e in events if e["kind"] == "tool"),
            "models": sum(1 for e in events if e["kind"] == "model"),
            "tokens": sum(e.get("tokens") or 0 for e in events if e["kind"] == "model"),
            "cost_usd": round(sum(e.get("cost_usd") or 0.0 for e in events if e["kind"] == "model"), 6),
            "violations": sum(1 for e in events if e["kind"] == "guardrail"),
        }
        return {**{k: v for k, v in run.items() if k not in ("events", "_t0")}, "events": events, "totals": totals}

    # ── 내부 ──
    def _run(self, ctx: ExecutionContext | None) -> dict[str, Any] | None:
        if ctx is None:
            return None
        run = self._runs.get(ctx.execution_id)
        if run is None:
            run = {
                "execution_id": ctx.execution_id,
                "agent_id": ctx.agent_id,
                "thread_id": ctx.thread_id or ctx.session_id,
                "user_id": ctx.user_id,
                "started_at": _now(),
                "ended_at": None,
                "duration_ms": None,
                "state": ctx.state,
                "error": None,
                "events": [],
                "_t0": time.perf_counter(),
            }
            self._runs[ctx.execution_id] = run
            while len(self._runs) > self._capacity:
                self._runs.popitem(last=False)
        return run

    def _add(self, ctx: ExecutionContext | None, kind: str, name: str, **fields: Any) -> None:
        with self._lock:
            run = self._run(ctx)
            if run is None:
                return
            row = {"seq": len(run["events"]), "t_ms": round((time.perf_counter() - run["_t0"]) * 1000, 1),
                   "kind": kind, "name": name, **fields}
            run["events"].append(row)

    def _start(self, key: tuple) -> None:
        with self._lock:
            self._open[key] = time.perf_counter()

    def _finish(self, key: tuple) -> tuple[float | None, str | None]:
        with self._lock:
            t0 = self._open.pop(key, None)
            err = self._errors.pop(key, None)
        return (round((time.perf_counter() - t0) * 1000, 1) if t0 is not None else None), err

    @staticmethod
    def _eid(ctx: ExecutionContext | None) -> str:
        return ctx.execution_id if ctx else "-"

    # ── Agent ──
    def before_agent(self, input: Any, ctx: ExecutionContext | None) -> None:
        with self._lock:
            self._run(ctx)

    def on_agent_error(self, input: Any, exc: BaseException, ctx: ExecutionContext | None) -> None:
        with self._lock:
            run = self._run(ctx)
            if run is not None:
                run["error"] = f"{type(exc).__name__}: {exc}"

    def finally_agent(self, input: Any, ctx: ExecutionContext | None) -> None:
        with self._lock:
            run = self._run(ctx)
            if run is None:
                return
            run["ended_at"] = _now()
            run["duration_ms"] = round((time.perf_counter() - run["_t0"]) * 1000, 1)
            state = ctx.state if ctx else run["state"]
            if state == "CREATED":  # 정책 미설정이면 상태 전이가 없다 — HTTP 응답과 같은 규칙으로 표기
                state = "FAILED" if run["error"] else "COMPLETED"
            run["state"] = state

    # ── Node ──
    def _nkey(self, node: str, state: Any, ctx: ExecutionContext | None) -> tuple:
        return (self._eid(ctx), "node", node, id(state))

    def before_node(self, node: str, state: Any, ctx: ExecutionContext | None) -> None:
        self._start(self._nkey(node, state, ctx))

    def on_node_error(self, node: str, state: Any, exc: BaseException, ctx: ExecutionContext | None) -> None:
        with self._lock:
            self._errors[self._nkey(node, state, ctx)] = f"{type(exc).__name__}: {exc}"

    def finally_node(self, node: str, state: Any, ctx: ExecutionContext | None) -> None:
        dur, err = self._finish(self._nkey(node, state, ctx))
        self._add(ctx, "node", node, duration_ms=dur, status="error" if err else "ok", error=err)

    # ── Model ──
    def before_model(self, model: str, prompt: str, ctx: ExecutionContext | None) -> None:
        self._start((self._eid(ctx), "model", model))

    def after_model(self, model: str, prompt: str, result: Any, ctx: ExecutionContext | None) -> None:
        dur, _ = self._finish((self._eid(ctx), "model", model))
        self._add(ctx, "model", model, duration_ms=dur, status="ok", tokens=None, cost_usd=None)

    # ── Tool ──
    def before_tool(self, tool: str, kwargs: dict, ctx: ExecutionContext | None) -> None:
        self._start((self._eid(ctx), "tool", tool))

    def after_tool(self, tool: str, kwargs: dict, result: Any, ctx: ExecutionContext | None) -> None:
        dur, _ = self._finish((self._eid(ctx), "tool", tool))
        self._add(ctx, "tool", tool, duration_ms=dur, status="ok")

    def on_tool_error(self, tool: str, kwargs: dict, exc: BaseException, ctx: ExecutionContext | None) -> None:
        dur, _ = self._finish((self._eid(ctx), "tool", tool))
        self._add(ctx, "tool", tool, duration_ms=dur, status="error", error=f"{type(exc).__name__}: {exc}")

    # ── Events(토큰·비용·가드레일·승인) ──
    def on_event(self, ev: Event) -> None:
        from klafi.core.context import get_context

        ctx = get_context()
        if ev.type == EventType.ModelCalled:
            with self._lock:
                run = self._run(ctx)
                if run is None:
                    return
                # 직전 model 행(같은 alias, 토큰 미기록)에 토큰·비용을 붙인다 — after_model 훅 뒤에 발행된다
                for row in reversed(run["events"]):
                    if row["kind"] == "model" and row["name"] == ev.data.get("model") and row.get("tokens") is None:
                        row["tokens"] = ev.data.get("tokens", 0)
                        row["cost_usd"] = ev.data.get("cost_usd")
                        return
            self._add(ctx, "model", ev.data.get("model", "?"), duration_ms=None, status="ok",
                      tokens=ev.data.get("tokens", 0), cost_usd=ev.data.get("cost_usd"))
        elif ev.type == EventType.GuardrailViolation:
            d = ev.data
            self._add(ctx, "guardrail", d.get("guard", "?"), stage=d.get("stage"), severity=d.get("severity"),
                      reason=d.get("reason"))
        elif ev.type == EventType.ApprovalRequested:
            self._add(ctx, "approval", ev.data.get("action", "?"), status="requested",
                      approval_id=ev.data.get("approval_id"), approver=ev.data.get("approver"))
        elif ev.type == EventType.ApprovalCompleted:
            self._add(ctx, "approval", ev.data.get("action", "?"), status="approved" if ev.data.get("approved") else "rejected",
                      approval_id=ev.data.get("approval_id"))


RECORDER = ExecutionRecorder()


def install() -> ExecutionRecorder:
    """전역 훅 + EventBus 구독으로 RECORDER 를 붙인다(멱등). clear_hooks() 뒤에도 다시 부르면 복구된다."""
    from klafi.events.bus import EVENTS

    if RECORDER not in _GLOBAL_HOOKS:  # clear_hooks() 뒤 복구
        register_hook(RECORDER)
    if not any(h == RECORDER.on_event for _, h in EVENTS._subs):  # EVENTS.clear() 뒤 복구
        subscribe(RECORDER.on_event, [EventType.ModelCalled, EventType.GuardrailViolation,
                                      EventType.ApprovalRequested, EventType.ApprovalCompleted])
    return RECORDER
