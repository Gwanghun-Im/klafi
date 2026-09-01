"""MCP(및 임의 LangChain) 외부 도구를 KLAFI Tool 로 감싸 governance 안으로 넣는다.

passthrough(to_langchain_tools)로 그냥 bind 하면 실행이 Tool.run 을 안 거쳐 권한·입력검증·audit·
tool 가드레일·policy 가 전부 우회된다("연결은 되나 통제 밖"). from_langchain_tool 로 감싸면 실행이
Tool.run 을 경유해 그 governance 가 그대로 적용된다.

MCP 도구는 async(ainvoke)라 sync 인 Tool.run 에서 돌도록 _sync 브리지를 쓴다.
connect_mcp 는 mcp.yaml(서버 목록)로 MultiServerMCPClient 를 조립해 서버별 permission 을 부착한
KLAFI Tool 묶음을 돌려준다. langchain-mcp-adapters 는 선택 의존(klafi[mcp]).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .tool import Tool


def _expand_env(v: Any) -> Any:
    """문자열 리프의 ${VAR} 를 os.environ 으로 치환(재귀). 비밀(API 키 등)을 mcp.yaml 평문 대신
    .env 로 주입하기 위함(SEC-05). 미정의 변수는 빈 문자열."""
    import os
    import re

    if isinstance(v, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), v)
    if isinstance(v, list):
        return [_expand_env(x) for x in v]
    if isinstance(v, dict):
        return {k: _expand_env(x) for k, x in v.items()}
    return v


def _sync(coro: Any) -> Any:
    """coroutine 을 sync 로 실행한다.

    KLAFI Tool.run 은 sync 이고(ToolNode 도 sync 툴을 executor 스레드에서 부른다), MCP 도구는 async 다.
    실행 중 이벤트 루프가 없으면 asyncio.run, 있으면 별도 스레드에서 asyncio.run(현 루프를 막지 않음).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # 실행 중 루프 없음 → 그대로
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def from_langchain_tool(
    lc: Any, *, required_permission: str | None = None, policy: Any = None, name: str | None = None
) -> Tool:
    """LangChain 도구(=MCP 도구)를 KLAFI Tool 로 감싼다 → 실행이 Tool.run 경유(governance 적용).

    lc.args_schema(pydantic)를 input_schema 로 재사용해 입력 검증도 붙는다. async(ainvoke)는
    _sync 브리지로 실행한다. required_permission 을 주면 최소권한 검사가 걸린다.
    """

    def _call(**kwargs: Any) -> Any:
        return _sync(lc.ainvoke(kwargs))

    # args_schema(원본)를 그대로 보존한다 — LLM 바인딩(as_langchain)이 이걸 LLM 에 노출해야
    # 모델이 올바른 인자(예: query)를 만든다. MCP 는 이게 JSON-schema dict 인 경우가 많은데,
    # KLAFI 입력검증(_validate_input)은 pydantic 일 때만 돌고 dict 는 건너뛴다(도구 자체 검증에 위임).
    # 권한·audit·가드레일·policy 는 그대로 적용.
    return Tool(
        _call,
        name=name or getattr(lc, "name", None) or "mcp_tool",
        description=getattr(lc, "description", "") or "",
        input_schema=getattr(lc, "args_schema", None),
        required_permission=required_permission,
        policy=policy,
    )


class McpTools:
    """connect_mcp 결과 — 서버별 KLAFI Tool 묶음. 클라이언트 참조를 잡아 세션 수명을 앱과 함께 유지."""

    def __init__(self, by_server: dict[str, list[Tool]], client: Any = None) -> None:
        self._by_server = by_server
        self._client = client  # 세션 수명 유지용(GC 방지)

    def tools(self, server: str) -> list[Tool]:
        return list(self._by_server.get(server, []))

    def all(self) -> list[Tool]:
        return [t for ts in self._by_server.values() for t in ts]

    def servers(self) -> list[str]:
        return list(self._by_server)


def connect_mcp(config: Any) -> McpTools:
    """mcp.yaml(경로/dict)로 MCP 서버를 조립해 서버별 KLAFI Tool 묶음을 돌려준다.

    형식:
        servers:
          fs: {command: npx, args: [...], transport: stdio, permission: fs:read, timeout: 20}
    각 서버의 `permission` → 그 서버 도구 전부의 required_permission(governance).
    `timeout`(초, 선택) → 그 서버 도구 전부에 툴별 실행 타임아웃(느린 외부 호출이 에이전트 전체
    예산을 잡아먹지 않게 빨리 실패). permission·timeout 은 KLAFI governance 키라 연결 설정에서
    분리하고, 나머지 키만 그대로 MultiServerMCPClient 로 넘긴다. langchain-mcp-adapters 미설치면
    친절한 에러. 서버가 없으면 빈 McpTools(예제 graceful degrade).
    """
    if isinstance(config, (str, Path)):
        p = Path(config)
        if not p.exists():
            return McpTools({})
        import yaml

        config = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    servers = (config or {}).get("servers") or {}
    if not servers:
        return McpTools({})

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:  # 선택 의존
        raise ImportError(
            "MCP 연결에는 langchain-mcp-adapters 가 필요합니다 — pip install 'klafi[mcp]'"
        ) from exc

    from klafi.runtime.policy import ExecutionPolicy

    _klafi_keys = ("permission", "timeout")  # governance 키 — 연결설정에서 분리
    connections = {
        name: _expand_env({k: v for k, v in (spec or {}).items() if k not in _klafi_keys})
        for name, spec in servers.items()
    }
    client = MultiServerMCPClient(connections)
    by_server: dict[str, list[Tool]] = {}
    for name, spec in servers.items():
        perm = (spec or {}).get("permission")
        timeout = (spec or {}).get("timeout")
        policy = ExecutionPolicy(timeout=timeout) if timeout is not None else None
        lc_tools = _sync(client.get_tools(server_name=name))  # 서버별로 받아 permission·timeout 부착
        by_server[name] = [
            from_langchain_tool(t, required_permission=perm, policy=policy) for t in lc_tools
        ]
    return McpTools(by_server, client)
