"""Human-in-the-Loop — Enterprise 승인 (요구사항 §13, F08).

LangGraph interrupt를 기반으로 승인 프로세스를 표준화한다.
- request_approval(): Node 안에서 호출 → 실행을 중단(interrupt)하고 승인 요청을 남긴다 (HIT-01/02).
- resume_approval(): 승인/반려 결정으로 재개한다 (HIT-03/10).
- Approval Adapter(HIT-11): 요청을 전자결재/Portal/Mobile로 밀어내는 확장점.
- 승인 요청/결정은 audit log로 남긴다 (HIT-09).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from langgraph.types import Command, interrupt

from klafi.core.context import get_context

_audit_log = logging.getLogger("klafi.approval")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ApprovalRequest:
    action: str
    payload: Any = None
    approver: str | None = None  # HIT-04
    approver_group: str | None = None  # HIT-05
    approval_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    requested_at: str = field(default_factory=_now)


@dataclass
class ApprovalDecision:
    approved: bool
    comment: str | None = None  # HIT-08
    decided_by: str | None = None
    decided_at: str = field(default_factory=_now)


# Approval Adapter (HIT-11): 요청을 외부 승인 시스템으로 전달. 기본 미설정.
ApprovalAdapter = Callable[[ApprovalRequest], None]
_adapter: ApprovalAdapter | None = None


def register_approval_adapter(adapter: ApprovalAdapter | None) -> None:
    global _adapter
    _adapter = adapter


def _audit(event: str, data: dict[str, Any]) -> None:
    ctx = get_context()
    eid = ctx.execution_id if ctx else "-"
    _audit_log.info("%s execution_id=%s %s", event, eid, data)


def request_approval(
    action: str,
    payload: Any = None,
    *,
    approver: str | None = None,
    approver_group: str | None = None,
) -> ApprovalDecision:
    """Node 안에서 호출. 실행을 중단하고 승인 요청을 남긴 뒤, 재개 시 결정을 반환한다.

    첫 실행에서는 interrupt로 중단되어 caller의 결과에 __interrupt__가 담긴다.
    resume_approval()로 재개하면 이 함수가 ApprovalDecision을 반환하며 Node가 이어진다.
    """
    req = ApprovalRequest(
        action=action, payload=payload, approver=approver, approver_group=approver_group
    )
    from klafi.events import EventType, emit  # lazy

    _audit("approval.requested", asdict(req))
    emit(EventType.ApprovalRequested, approval_id=req.approval_id, action=action, approver=approver)
    if _adapter is not None:
        _adapter(req)  # 전자결재/Portal 등으로 전달

    raw = interrupt(asdict(req))  # 여기서 중단. 재개 시 Command(resume=...) 값이 raw로 들어온다.

    decision = _normalize(raw)
    _audit("approval.decided", {"approval_id": req.approval_id, **asdict(decision)})
    emit(EventType.ApprovalCompleted, approval_id=req.approval_id, approved=decision.approved)
    return decision


def _normalize(raw: Any) -> ApprovalDecision:
    if isinstance(raw, ApprovalDecision):
        return raw
    if isinstance(raw, bool):
        return ApprovalDecision(approved=raw)
    if isinstance(raw, dict):
        return ApprovalDecision(
            approved=bool(raw.get("approved", False)),
            comment=raw.get("comment"),
            decided_by=raw.get("decided_by"),
        )
    return ApprovalDecision(approved=bool(raw))


def resume_approval(
    agent: Any,
    thread_id: str,
    approved: bool,
    *,
    comment: str | None = None,
    decided_by: str | None = None,
    context: Any = None,
) -> Any:
    """승인/반려 결정으로 중단된 Agent를 재개한다 (HIT-03/10).

    재개는 새 실행이므로 권한이 필요한 Node(예: 매수)가 있으면 승인자의 security_context를
    context로 넘긴다. (HTTP /resume는 auth 어댑터가 자동 주입)
    """
    decision = {"approved": approved, "comment": comment, "decided_by": decided_by}
    return agent.invoke(Command(resume=decision), context=context, thread_id=thread_id)


def pending_approvals(result: Any) -> list[Any]:
    """invoke 결과에서 대기 중인 interrupt(승인요청) 목록을 꺼낸다."""
    if isinstance(result, dict):
        return list(result.get("__interrupt__", []))
    return []
