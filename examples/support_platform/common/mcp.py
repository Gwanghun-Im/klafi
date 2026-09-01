"""MCP 외부 도구 — config/mcp.yaml 을 connect_mcp 로 조립해 에이전트가 import 할 수 있게 노출한다.

에이전트는 `from common.mcp import fs` 후 `init_chat_model(...).bind_tools([*fs])` 로 쓴다.
MCP 도구도 KLAFI Tool 로 감싸져 권한(서버 permission)·검증·audit·가드레일을 그대로 탄다.

graceful degrade: klafi[mcp] 미설치 / 서버 미기동 / mcp.yaml 없음 → 빈 리스트. 예제·demo 는 MCP 없이도
그대로 돌아간다(MCP 는 opt-in).
"""

import logging
from pathlib import Path

from klafi.tool import McpTools, connect_mcp

_HERE = Path(__file__).resolve().parent


def _load() -> McpTools:
    try:
        return connect_mcp(_HERE / "config" / "mcp.yaml")
    except Exception as exc:  # noqa: BLE001 — lib 미설치·서버 미기동 등 → MCP 없이 진행
        logging.getLogger("klafi").info("MCP 연결 건너뜀 (MCP 없이 실행): %s", exc)
        return McpTools({})


_mcp = _load()

search = _mcp.tools("tavily")  # Tavily 웹 검색 도구 (없으면 []). 에이전트가 bind_tools 에 넣어 쓴다.
