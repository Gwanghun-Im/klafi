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
    """문자열 리프의 ${VAR}/${VAR:default} 치환 — LayeredConfig 와 **같은 구현**(klafi.config.layered.expand_env).

    비밀(API 키 등)을 mcp.yaml 평문 대신 .env 로 주입하기 위함(SEC-05). 미정의 변수에 기본값이 없으면
    fail-fast(ConfigNotFoundError) — 예전 구현은 빈 문자열로 조용히 통과시켜 키 없는 서버가 떴다.
    """
    from klafi.config.layered import expand_env

    return expand_env(v)


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


class _LoopThread:
    """전용 이벤트루프 스레드 — 영속 MCP 세션이 사는 곳.

    MCP 세션은 자기가 태어난 루프에서만 안전하다. 일회용 루프(_sync)로 돌리면 호출마다 세션을
    새로 만들어 서버 프로세스 재기동(npx ~1s)을 물게 되므로, 세션의 생성·도구 호출·종료를
    이 한 루프에 몰아넣고 어느 스레드에서든 run_coroutine_threadsafe 로 넘긴다.
    """

    def __init__(self) -> None:
        import threading

        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="klafi-mcp", daemon=True).start()

    def run(self, coro: Any, timeout: float | None = None) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def open_session(self, cm: Any, timeout: float | None = None) -> tuple[Any, Any]:
        """async CM(client.session)을 **한 태스크 안에서** 열고 닫는다 → (session, handle).

        __aenter__/__aexit__ 를 run() 으로 따로 던지면 태스크가 달라져 anyio cancel scope 가
        "다른 태스크에서 exit" 오류를 낸다. 태스크 하나가 세션을 잡고 stop 신호까지 기다린다.
        """
        import concurrent.futures

        ready: concurrent.futures.Future = concurrent.futures.Future()

        async def hold() -> None:
            stop = asyncio.Event()
            try:
                async with cm as session:
                    ready.set_result((session, stop))
                    await stop.wait()
            except BaseException as exc:
                if not ready.done():
                    ready.set_exception(exc)
                    return
                raise

        fut = asyncio.run_coroutine_threadsafe(hold(), self.loop)
        session, stop = ready.result(timeout)
        return session, (fut, stop)

    def close_session(self, handle: Any, timeout: float | None = 10) -> None:
        fut, stop = handle
        self.loop.call_soon_threadsafe(stop.set)
        fut.result(timeout)  # 같은 태스크에서 __aexit__ 완료

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)


def from_langchain_tool(
    lc: Any,
    *,
    required_permission: str | None = None,
    policy: Any = None,
    name: str | None = None,
    runner: Any = None,
) -> Tool:
    """LangChain 도구(=MCP 도구)를 KLAFI Tool 로 감싼다 → 실행이 Tool.run 경유(governance 적용).

    lc.args_schema(pydantic)를 input_schema 로 재사용해 입력 검증도 붙는다. async(ainvoke)는
    runner(기본 _sync 브리지)로 실행한다 — 영속 세션 도구는 세션이 사는 루프의 runner 를 받는다.
    required_permission 을 주면 최소권한 검사가 걸린다.
    """
    run = runner or _sync
    tool_name = name or getattr(lc, "name", None) or "mcp_tool"

    def _call(**kwargs: Any) -> Any:
        import uuid

        from langchain_core.tools import BaseTool

        from klafi.core.exceptions import ToolException

        if not isinstance(lc, BaseTool):  # 덕타이핑 러너블(테스트 페이크 등) — kwargs 그대로
            return run(lc.ainvoke(kwargs))
        # kwargs 로 부르면 BaseTool 이 status·artifact 를 버린 bare content 만 돌려준다(tool_call_id 없음)
        # → MCP isError 가 성공으로 둔갑. ToolCall 로 불러 ToolMessage 를 받고 status 를 KLAFI 예외로 올린다.
        call = {"type": "tool_call", "name": lc.name, "args": kwargs, "id": f"klafi-{uuid.uuid4().hex[:12]}"}
        msg = run(lc.ainvoke(call))
        if getattr(msg, "status", None) == "error":
            raise ToolException(f"tool '{tool_name}' 실패: {_text(getattr(msg, 'content', msg))}", tool=tool_name)
        return getattr(msg, "content", msg)

    # args_schema(원본)를 그대로 보존한다 — LLM 바인딩(as_langchain)이 이걸 LLM 에 노출해야
    # 모델이 올바른 인자(예: query)를 만든다. MCP 는 이게 JSON-schema dict 인 경우가 많은데,
    # KLAFI 입력검증(_validate_input)은 pydantic 일 때만 돌고 dict 는 건너뛴다(도구 자체 검증에 위임).
    # 권한·audit·가드레일·policy 는 그대로 적용.
    return Tool(
        _call,
        name=tool_name,
        description=getattr(lc, "description", "") or "",
        input_schema=getattr(lc, "args_schema", None),
        required_permission=required_permission,
        policy=policy,
    )


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


class McpTools:
    """connect_mcp 결과 — 서버별 KLAFI Tool 묶음. 클라이언트·영속 세션을 잡아 앱 수명과 함께 유지."""

    def __init__(
        self,
        by_server: dict[str, list[Tool]],
        client: Any = None,
        sessions: dict[str, Any] | None = None,
        loop: "_LoopThread | None" = None,
    ) -> None:
        self._by_server = by_server
        self._client = client  # 세션 수명 유지용(GC 방지)
        self._sessions = sessions or {}  # {server: async CM} — 영속 세션(열린 채 유지)
        self._loop = loop  # 세션이 사는 전용 루프

    def tools(self, server: str) -> list[Tool]:
        return list(self._by_server.get(server, []))

    def all(self) -> list[Tool]:
        return [t for ts in self._by_server.values() for t in ts]

    def servers(self) -> list[str]:
        return list(self._by_server)

    def close(self) -> None:
        """영속 세션·루프 정리(테스트·명시적 종료용). 데몬 스레드라 프로세스 종료 시엔 불필요."""
        if self._loop is None:
            return
        import logging

        for name, handle in self._sessions.items():
            try:
                self._loop.close_session(handle, timeout=10)
            except Exception as exc:  # noqa: BLE001 — 종료 실패가 종료를 막지 않지만 삼키지도 않는다
                logging.getLogger("klafi.tool").warning("mcp.close server=%s 실패: %s", name, exc)
        self._sessions.clear()
        self._loop.stop()


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

    def _gov(spec: dict) -> tuple:
        perm = (spec or {}).get("permission")
        timeout = (spec or {}).get("timeout")
        return perm, ExecutionPolicy(timeout=timeout) if timeout is not None else None

    by_server: dict[str, list[Tool]] = {}

    # ── 영속 세션 경로(기본): 서버 프로세스를 앱 수명 동안 유지 ─────────────
    # get_tools 는 "툴 호출마다 새 세션"이라 stdio 서버(npx 등)가 호출마다 재기동된다(~1s).
    # session()+load_mcp_tools(session) 조합이면 세션 바인딩 도구가 되어 재기동이 사라진다.
    try:
        from langchain_mcp_adapters.tools import load_mcp_tools
    except ImportError:
        load_mcp_tools = None
    if load_mcp_tools is not None and hasattr(client, "session"):
        loop = _LoopThread()
        sessions: dict[str, Any] = {}
        for name, spec in servers.items():
            perm, policy = _gov(spec)
            # 서버 기동+핸드셰이크(부팅 1회). enter/exit 를 같은 태스크가 담당한다(anyio cancel scope).
            session, sessions[name] = loop.open_session(client.session(name), timeout=60)
            lc_tools = loop.run(load_mcp_tools(session, server_name=name), timeout=60)
            by_server[name] = [
                from_langchain_tool(t, required_permission=perm, policy=policy, runner=loop.run)
                for t in lc_tools
            ]
        return McpTools(by_server, client, sessions=sessions, loop=loop)

    # ── 폴백: 구버전 어댑터(session/load_mcp_tools 미지원) → 호출마다 새 세션 ──
    for name, spec in servers.items():
        perm, policy = _gov(spec)
        lc_tools = _sync(client.get_tools(server_name=name))  # 서버별로 받아 permission·timeout 부착
        by_server[name] = [
            from_langchain_tool(t, required_permission=perm, policy=policy) for t in lc_tools
        ]
    return McpTools(by_server, client)
