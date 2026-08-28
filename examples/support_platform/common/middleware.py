"""공통 노드 미들웨어 — 값 콜러블. @klafi_node(before=/after=) 리스트에 넣는다.

가드레일(문자열 정책, .check 보유)과 달리 값 콜러블은 값 전체를 한 번 받는다. messages가
비어도 항상 실행되므로, 권한처럼 '내용과 무관하게 반드시 도는' 검증은 이쪽이어야 한다
(문자열 리프가 없으면 가드레일은 호출조차 안 됨 — 조용한 권한 우회 방지). None을 반환하면
값을 바꾸지 않고 관측만 한다.
"""

import logging


def require_orders_read(state, ctx):
    """세션/권한 확인 — orders:read 없으면 fail-close."""
    perms = ctx.security_context.get("permissions", []) if ctx else []
    if "orders:read" not in perms:
        raise PermissionError("orders:read 권한이 필요합니다")
    return None  # state 변경 없음(관측·검증만)


def audit_log(state, ctx):
    """감사 로깅 — 어느 노드가 어떤 사용자로 실행됐는지 기록."""
    logging.getLogger("klafi.audit").info(
        "audit agent=%s user=%s", ctx.agent_id if ctx else "-", ctx.user_id if ctx else "-"
    )
