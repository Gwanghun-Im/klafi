"""MCP/LangChain 외부 도구 → KLAFI Tool 어댑터·커넥터.

실제 MCP 서버 없이 검증한다(가짜 async LangChain 도구). 핵심: from_langchain_tool 로 감싸면
async MCP 도구가 sync Tool.run 을 경유해 권한·검증·audit·가드레일을 그대로 탄다.
"""

import sys
import types

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from klafi.core.context import ExecutionContext, bind_context
from klafi.core.hook import bind_hooks
from klafi.events import EventType, subscribe
from klafi.events.bus import EVENTS
from klafi.tool import connect_mcp, from_langchain_tool


class _In(BaseModel):
    x: int


async def _afn(x: int) -> str:  # async 전용 LangChain 도구 (MCP 도구가 이 모양)
    return f"got {x}"


def _fake_lc(name: str = "echo"):
    return StructuredTool.from_function(coroutine=_afn, name=name, description="echo", args_schema=_In)


# ── 어댑터: async→sync 브리지 + governance ────────────────────────────────
def test_async_tool_runs_through_sync_tool_run():
    t = from_langchain_tool(_fake_lc())  # 권한 없음
    assert t.run(x=5) == "got 5"  # _sync 브리지로 async 실행


def test_input_schema_is_enforced():
    from klafi.core.exceptions import ToolValidationError

    t = from_langchain_tool(_fake_lc())
    with pytest.raises(ToolValidationError):
        t.run(x="not-an-int")  # args_schema(_In) 위반


def test_required_permission_is_enforced_and_audited():
    from klafi.core.exceptions import ToolPermissionError

    seen = []
    subscribe(lambda e: seen.append(e.type))
    try:
        t = from_langchain_tool(_fake_lc(), required_permission="fs:read")
        ctx = ExecutionContext.new(agent_id="a", security_context={"permissions": []})
        with bind_context(ctx):
            with pytest.raises(ToolPermissionError):
                t.run(x=1)
        assert EventType.ToolStarted in seen and EventType.ToolFailed in seen  # 감사 흔적
    finally:
        EVENTS.clear()


def test_tool_boundary_guardrail_applies_to_mcp_tool():
    from klafi.guardrail import GuardrailHook, GuardrailResult, guardrail

    @guardrail(raw=True)  # 결과(str)를 마스킹
    def mask(v):
        return GuardrailResult(False, "mask", replacement="MASKED")

    t = from_langchain_tool(_fake_lc())
    with bind_context(ExecutionContext.new()), bind_hooks([GuardrailHook(tool_output=[mask])]):
        assert t.run(x=5) == "MASKED"  # after_tool 가드레일이 MCP 도구 결과도 교체


# ── 커넥터 connect_mcp ────────────────────────────────────────────────────
def _inject_fake_client(monkeypatch, tools):
    class FakeClient:
        def __init__(self, connections):
            self.connections = connections

        async def get_tools(self, server_name=None):
            return tools

    parent = types.ModuleType("langchain_mcp_adapters")
    client_mod = types.ModuleType("langchain_mcp_adapters.client")
    client_mod.MultiServerMCPClient = FakeClient
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters", parent)
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.client", client_mod)


def test_connect_mcp_wraps_with_server_permission(monkeypatch):
    _inject_fake_client(monkeypatch, [_fake_lc("read_file")])
    mt = connect_mcp(
        {"servers": {"fs": {"command": "x", "transport": "stdio", "permission": "fs:read", "timeout": 20}}}
    )
    tools = mt.tools("fs")
    assert len(tools) == 1 and tools[0].name == "read_file"
    assert tools[0]._permission == "fs:read"  # 서버 permission 이 도구에 부착됨
    assert tools[0]._policy is not None and tools[0]._policy.timeout == 20  # 서버 timeout → 툴별 정책
    assert [t.name for t in mt.all()] == ["read_file"]


def test_connect_mcp_empty_is_noop():
    assert connect_mcp({"servers": {}}).all() == []  # 서버 없음 → 빈 묶음(라이브러리도 안 부름)


def test_connect_mcp_requires_extra_when_lib_missing(monkeypatch):
    # 라이브러리 미설치 상황을 강제 (import 실패)
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters", None)
    with pytest.raises(ImportError, match="klafi\\[mcp\\]"):
        connect_mcp({"servers": {"fs": {"command": "x", "permission": "fs:read"}}})


def test_connect_mcp_expands_env_vars(monkeypatch):
    """mcp.yaml 의 ${VAR} 는 .env(os.environ)로 치환된다(비밀 평문 금지, SEC-05)."""
    captured = {}

    class FakeClient:
        def __init__(self, connections):
            captured.update(connections)

        async def get_tools(self, server_name=None):
            return []

    parent = types.ModuleType("langchain_mcp_adapters")
    client_mod = types.ModuleType("langchain_mcp_adapters.client")
    client_mod.MultiServerMCPClient = FakeClient
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters", parent)
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.client", client_mod)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret")

    connect_mcp(
        {"servers": {"tavily": {"command": "npx", "env": {"TAVILY_API_KEY": "${TAVILY_API_KEY}"},
                                "permission": "web:search"}}}
    )
    assert captured["tavily"]["env"]["TAVILY_API_KEY"] == "tvly-secret"  # 치환됨
    assert "permission" not in captured["tavily"]  # permission 은 연결설정에서 분리


def test_connect_mcp_persistent_session_opens_once(monkeypatch):
    """session 지원 어댑터면: 서버 세션을 부팅 시 한 번만 열고(호출마다 재기동 금지),
    도구는 그 세션이 사는 전용 루프에서 실행된다. close() 로 세션·루프가 정리된다."""
    enters, exits = [], []

    class FakeSession: ...

    class _CM:
        async def __aenter__(self):
            enters.append(1)
            return FakeSession()

        async def __aexit__(self, *a):
            exits.append(1)
            return False

    class FakeClient:
        def __init__(self, connections):
            self.connections = connections

        def session(self, name):
            return _CM()

    async def fake_load(session, **kw):
        assert isinstance(session, FakeSession)  # 세션 바인딩 도구로 로드됨
        return [_fake_lc("echo")]

    parent = types.ModuleType("langchain_mcp_adapters")
    client_mod = types.ModuleType("langchain_mcp_adapters.client")
    client_mod.MultiServerMCPClient = FakeClient
    tools_mod = types.ModuleType("langchain_mcp_adapters.tools")
    tools_mod.load_mcp_tools = fake_load
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters", parent)
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.client", client_mod)
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.tools", tools_mod)

    mt = connect_mcp({"servers": {"echo": {"command": "x", "transport": "stdio"}}})
    try:
        t = mt.tools("echo")[0]
        assert t.run(x=1) == "got 1" and t.run(x=2) == "got 2"  # 호출 2번
        assert enters == [1]  # 세션은 부팅 때 딱 한 번 (호출마다 재기동 아님)
    finally:
        mt.close()
    assert exits == [1]  # close 가 세션을 닫는다


def test_dict_args_schema_is_preserved_for_llm_and_skips_klafi_validation():
    """MCP 도구의 args_schema 가 JSON-schema dict 면: KLAFI 검증은 생략하되(호출 불가),
    스키마는 보존해 LLM 바인딩(as_langchain)에 그대로 노출한다(→ 모델이 query 를 만든다)."""

    class _DictSchemaTool:  # StructuredTool 아님 — args_schema 가 dict 인 MCP 도구 모사
        name = "tavily_search"
        description = "web search"
        args_schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

        async def ainvoke(self, kwargs):
            return f"results:{kwargs['query']}"

    t = from_langchain_tool(_DictSchemaTool(), required_permission="web:search")
    assert t._input_schema is not None  # 스키마 보존(LLM 노출용)
    assert t.as_langchain().args == {"query": {"type": "string"}}  # LLM 이 query 를 본다(kwargs 감싸기 아님)
    with bind_context(ExecutionContext.new(security_context={"permissions": ["web:search"]})):
        assert t.run(query="seoul") == "results:seoul"  # KLAFI 검증 생략, 그대로 실행(권한 적용)
