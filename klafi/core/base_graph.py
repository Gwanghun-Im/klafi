"""BaseGraph — KLAFI 표준 Agent 실행 진입점 (요구사항 F01).

개발자는 build()에서 순수 LangGraph StateGraph를 그리기만 한다.
BaseGraph는 실행 시 ExecutionContext(execution_id/trace_id 포함)를 만들어
실행 Scope에 바인딩하고, LangGraph config에 thread_id·metadata를 얹는다.

Open Framework 원칙(§3.3): 컴파일된 LangGraph는 .compiled로 그대로 노출한다.
개발자가 원하면 StateGraph native API에 언제든 직접 접근할 수 있다.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import Any, AsyncIterator, Iterator

from langgraph.graph.state import CompiledStateGraph, StateGraph

from .context import ExecutionContext, bind_context, get_context
from .exceptions import AgentExecutionException
from .hook import (
    Hook,
    _error,
    _finally,
    _transform,
    bind_hooks,
    is_control_flow,
    resolve_hooks,
    wrap_node,
)
from .spec import AgentSpec

_log = logging.getLogger("klafi.graph")

# LangGraph Pregel.invoke/stream 의 실행 키워드 — RunnableConfig 가 아니라 호출 인자로 넘겨야 동작한다.
# (config 에 접어 넣으면 무음으로 무시된다: interrupt_before/durability/stream_mode 유실 결함의 원인)
_PREGEL_KWARGS = frozenset(
    {"stream_mode", "print_mode", "output_keys", "interrupt_before", "interrupt_after",
     "durability", "subgraphs", "debug", "control"}
)


class BaseGraph:
    def __init__(
        self,
        spec: AgentSpec,
        checkpointer: Any = None,
        hooks: list[Hook] | None = None,
        policy: Any = None,
        store: Any = None,
        cache: Any = None,
    ) -> None:
        self.spec = spec
        self.hooks = hooks or []
        # 정책이 없으면 runtime을 import하지 않아 core 독립성 유지.
        self.policy = self._init_policy(policy)
        # checkpointer 자동 주입 (FAC-02 / MEM-05): 명시 인자 > spec.config["checkpoint"].
        # 인자·config 모두 saver 인스턴스 / "memory" 같은 name / dict 를 허용.
        src = checkpointer if checkpointer is not None else (spec.config or {}).get("checkpoint")
        self.checkpointer = self._resolve_checkpointer(src)
        # Long-Term Memory Store 자동 주입 (FAC-04): 명시 인자 > spec.config["store"].
        store_src = store if store is not None else (spec.config or {}).get("store")
        self.store = self._resolve_store(store_src)
        # 노드 캐시 백엔드: add_node(cache_policy=...) 는 compile(cache=) 가 있어야 동작한다.
        cache_src = cache if cache is not None else (spec.config or {}).get("cache")
        self.cache = self._resolve_cache(cache_src)
        graph = self.build()
        if not isinstance(graph, StateGraph):
            raise AgentExecutionException(
                "build()는 langgraph StateGraph를 반환해야 합니다",
                agent_id=spec.id,
            )
        self._guarded_nodes: set[str] = set()  # after 파이프라인·구조화 출력 노드 — 스트림 원문 토큰 억제
        self._internal_nodes: set[str] = set()  # visibility="internal" — 스트림 미전달
        self._structured_nodes: dict[str, str | None] = {}  # output 계약 노드 → state key(자동이면 None)
        self._node_contracts: dict[str, dict[str, Any]] = {}
        self._install_node_hooks(graph)
        self.compiled: CompiledStateGraph = graph.compile(
            checkpointer=self.checkpointer, store=self.store, cache=self.cache
        )

    @staticmethod
    def _resolve_cache(src: Any) -> Any:
        if src is None:
            return None
        if src == "memory":
            from langgraph.cache.memory import InMemoryCache

            return InMemoryCache()
        return src  # BaseCache 인스턴스

    def _resolve_checkpointer(self, src: Any) -> Any:
        if src is None:
            return None  # 미사용 시 context 패키지를 import하지 않아 core 독립
        from klafi.context.checkpoint import resolve_checkpointer  # lazy

        return resolve_checkpointer(src)

    def _resolve_store(self, src: Any) -> Any:
        if src is None:
            return None
        from klafi.context.memory import resolve_store  # lazy

        return resolve_store(src)

    def memory(self, pii_filter: Any = None) -> Any:
        """주입된 Store를 KLAFI MemoryStore 래퍼로 감싸 반환 (없으면 None)."""
        if self.store is None:
            return None
        from klafi.context.memory import MemoryStore

        return MemoryStore(self.store, pii_filter=pii_filter)

    @staticmethod
    def _thread_id(ctx: ExecutionContext, thread_id: str | None) -> str:
        # Thread ID 표준 (MEM-05): 명시 thread_id > session_id > execution_id.
        return thread_id or ctx.session_id or ctx.execution_id

    def get_state(
        self, thread_id: str | None = None, context: ExecutionContext | None = None
    ) -> Any:
        """Checkpoint 조회 (MEM-07). Resume 대상 Thread의 저장 상태를 반환."""
        ctx = context or self._make_context(None)
        tid = self._thread_id(ctx, thread_id)
        return self.compiled.get_state({"configurable": {"thread_id": tid}})

    # 전역 + Agent Hook을 호출 시점에 최신으로 해석 (enable/disable, 지연 등록 반영)
    def _resolved_hooks(self) -> list[Hook]:
        return resolve_hooks(self.hooks)

    # ── 실행 정책 (Timeout/Retry) ────────────────────────────────────────
    def _init_policy(self, policy: Any) -> Any:
        src = policy if policy is not None else (self.spec.config or {}).get("policy")
        if src is None:
            return None
        from klafi.runtime.policy import ExecutionPolicy  # lazy: core→runtime 경계

        return ExecutionPolicy.from_config(src)

    def _resolve_policy(self, override: Any) -> Any:
        if override is None:
            return self.policy
        from klafi.runtime.policy import ExecutionPolicy

        return ExecutionPolicy.from_config(override)

    def _set_state(self, ctx: ExecutionContext, state: Any) -> None:
        ctx.state = state.value

    def _retry_payload(self, input: Any) -> Any:
        """정책 Retry 시 넘길 입력을 결정한다.

        Checkpointer가 있으면 2회차부터 None(=Resume)을 넘겨 **완료된 Node를 재실행하지 않는다**.
        (원본 input을 다시 넘기면 그래프가 처음부터 재실행되어 결제 등 부수효과가 중복된다.)
        Checkpointer가 없으면 재개할 상태가 없으므로 원본 input으로 재시도한다.
        """
        state = {"attempt": 0}

        def payload() -> Any:
            first = state["attempt"] == 0
            state["attempt"] += 1
            if not first and self.checkpointer is None:
                _log.warning(
                    "policy.retry: checkpointer 없이 재시도 — 완료된 노드의 부수효과가 재실행됩니다 "
                    "(agent=%s). 노드 단위 재시도는 add_node(retry_policy=RetryPolicy(...)) 를 쓰세요.",
                    self.spec.id,
                )
            return input if (first or self.checkpointer is None) else None

        return payload

    @staticmethod
    def _mark_interrupt_state(ctx: ExecutionContext, result: Any) -> None:
        # 최신 LangGraph는 interrupt 시 예외 대신 결과에 __interrupt__를 담아 반환한다.
        if isinstance(result, dict) and result.get("__interrupt__"):
            ctx.state = "WAITING_APPROVAL"  # §8 승인 대기 (HIT)

    def _run_sync(self, fn: Any, ctx: ExecutionContext, policy: Any, hooks: list[Hook]) -> Any:
        def wrapped() -> Any:
            with bind_hooks(hooks):  # Tool/Model 경계에서 참조할 활성 Hook
                return fn()

        if policy is None:
            return wrapped()
        from klafi.runtime.engine import run_sync

        return run_sync(wrapped, policy, lambda s: self._set_state(ctx, s))

    async def _run_async(self, coro_fn: Any, ctx: ExecutionContext, policy: Any, hooks: list[Hook]) -> Any:
        async def wrapped() -> Any:
            with bind_hooks(hooks):
                return await coro_fn()

        if policy is None:
            return await wrapped()
        from klafi.runtime.engine import run_async

        return await run_async(wrapped, policy, lambda s: self._set_state(ctx, s))

    def _install_node_hooks(self, graph: StateGraph) -> None:
        """모든 Node의 runnable을 in-place로 Hook 래핑 (개발자 코드 0줄, §11 DoD)."""
        for name, node_spec in graph.nodes.items():
            runnable = node_spec.runnable
            func, afunc = getattr(runnable, "func", None), getattr(runnable, "afunc", None)
            fn = func or afunc
            visibility = getattr(fn, "__klafi_visibility__", "external")
            contract = getattr(fn, "__klafi_output__", None)
            if getattr(fn, "__klafi_after__", False) or contract:
                self._guarded_nodes.add(name)
            if visibility == "internal":
                self._internal_nodes.add(name)
            if contract:
                self._structured_nodes[name] = contract[0]
            self._node_contracts[name] = {
                "visibility": visibility,
                "output_key": contract[0] if contract else None,
                "output_schema": contract[1].model_json_schema() if contract else None,
            }
            if func is None and afunc is None:
                # 컴파일된 서브그래프 등 순수 Runnable — invoke/ainvoke 를 훅으로 감싼다.
                from langgraph._internal._runnable import RunnableCallable

                sync_fn, async_fn = _hooked_runnable(runnable, name, self._resolved_hooks)
                graph.nodes[name] = dataclasses.replace(
                    node_spec, runnable=RunnableCallable(sync_fn, afunc=async_fn, name=name)
                )
                continue
            if func is not None:
                runnable.func = wrap_node(func, name, self._resolved_hooks, is_async=False)
            if afunc is not None:
                runnable.afunc = wrap_node(afunc, name, self._resolved_hooks, is_async=True)

    def node_contracts(self) -> dict[str, dict[str, Any]]:
        """노드별 전달 계약 — visibility(internal/external)·output_schema(JSON Schema)·output_key. 서버 메타데이터에 노출."""
        return {k: dict(v) for k, v in self._node_contracts.items()}

    # 개발자가 구현: 순수 LangGraph StateGraph 반환
    def build(self) -> StateGraph:  # pragma: no cover - 추상 메서드
        raise NotImplementedError

    # ── @klafi_graph 파이프라인 (워크플로우 경계 가드레일·미들웨어) ──────
    def _graph_spec(self) -> Any:
        return getattr(type(self), "__klafi_graph__", None)

    def _enter(self, input: Any, ctx: ExecutionContext | None) -> Any:
        """before 파이프라인(가드레일·미들웨어) — 반환값이 그래프로 들어간다."""
        spec = self._graph_spec()
        if spec is None or not spec.before:
            return input
        from .middleware import apply

        return apply(spec.before, input, ctx, "input")

    async def _aenter(self, input: Any, ctx: ExecutionContext | None) -> Any:
        spec = self._graph_spec()
        if spec is None or not spec.before:
            return input
        from .middleware import aapply

        return await aapply(spec.before, input, ctx, "input")

    def _exit(self, result: Any, ctx: ExecutionContext | None) -> Any:
        """after 파이프라인(가드레일·미들웨어) — 반환값이 최종 결과가 된다."""
        spec = self._graph_spec()
        if spec is None or not spec.after:
            return result
        from .middleware import apply

        return apply(spec.after, result, ctx, "output")

    async def _aexit(self, result: Any, ctx: ExecutionContext | None) -> Any:
        spec = self._graph_spec()
        if spec is None or not spec.after:
            return result
        from .middleware import aapply

        return await aapply(spec.after, result, ctx, "output")

    def _graph_error(self, exc: BaseException, input: Any, ctx: ExecutionContext | None) -> None:
        spec = self._graph_spec()
        if spec is None or not spec.on_error:
            return
        from .middleware import fire_error

        fire_error(spec.on_error, exc, input, ctx)

    # ── 실행 API (EXE-01/02/03: sync / async / stream) ──────────────────
    def invoke(
        self,
        input: Any,
        context: ExecutionContext | None = None,
        policy: Any = None,
        thread_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        ctx = self._make_context(context)
        pol = self._resolve_policy(policy)
        cfg_kw, run_kw = self._split_kwargs(kwargs)
        with bind_context(ctx):
            hooks = self._resolved_hooks()
            input = _transform(hooks, "before_agent", input, lambda v: (v, ctx))
            try:
                input = self._enter(input, ctx)  # @klafi_graph before 파이프라인
                payload = self._retry_payload(input)
                result = self._run_sync(
                    lambda: self.compiled.invoke(
                        payload(), config=self._config(ctx, cfg_kw, thread_id), **run_kw
                    ),
                    ctx,
                    pol,
                    hooks,
                )
                self._mark_interrupt_state(ctx, result)
                result = self._exit(result, ctx)  # @klafi_graph after 파이프라인
                result = _transform(hooks, "after_agent", result, lambda v: (input, v, ctx), reverse=True)
                return result
            except BaseException as exc:
                self._on_failure(exc, input, ctx, hooks)
                raise
            finally:
                _finally(hooks, "finally_agent", input, ctx)

    def _on_failure(self, exc: BaseException, input: Any, ctx: ExecutionContext, hooks: list[Hook]) -> None:
        """예외 분류: 취소 → CANCELLED(실패 아님), interrupt → WAITING_APPROVAL, 그 외 → error 훅."""
        if isinstance(exc, asyncio.CancelledError):  # 클라이언트 끊김·타임아웃 취소 — 오류로 집계하지 않는다
            ctx.state = "CANCELLED"
        elif is_control_flow(exc):  # interrupt(HITL) → 실패 아님, WAITING_APPROVAL
            ctx.state = "WAITING_APPROVAL"
        else:
            self._graph_error(exc, input, ctx)
            _error(hooks, "on_agent_error", input, exc, ctx)

    @staticmethod
    def _split_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """invoke/stream kwargs 를 (RunnableConfig 용, Pregel 실행 인자용) 으로 나눈다.

        runtime_context= 는 LangGraph Runtime.context(StateGraph(context_schema=...)) 로 전달된다 —
        KLAFI 의 context= 는 ExecutionContext 라 이름이 겹쳐 별도 키워드로 받는다.
        """
        run = {k: v for k, v in kwargs.items() if k in _PREGEL_KWARGS}
        if "runtime_context" in kwargs:
            run["context"] = kwargs["runtime_context"]
        cfg = {k: v for k, v in kwargs.items() if k not in _PREGEL_KWARGS and k != "runtime_context"}
        return cfg, run

    async def ainvoke(
        self,
        input: Any,
        context: ExecutionContext | None = None,
        policy: Any = None,
        thread_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        ctx = self._make_context(context)
        pol = self._resolve_policy(policy)
        cfg_kw, run_kw = self._split_kwargs(kwargs)
        with bind_context(ctx):
            hooks = self._resolved_hooks()
            input = _transform(hooks, "before_agent", input, lambda v: (v, ctx))
            try:
                input = await self._aenter(input, ctx)  # @klafi_graph before 파이프라인
                payload = self._retry_payload(input)
                result = await self._run_async(
                    lambda: self.compiled.ainvoke(
                        payload(), config=self._config(ctx, cfg_kw, thread_id), **run_kw
                    ),
                    ctx,
                    pol,
                    hooks,
                )
                self._mark_interrupt_state(ctx, result)
                result = await self._aexit(result, ctx)  # @klafi_graph after 파이프라인
                result = _transform(hooks, "after_agent", result, lambda v: (input, v, ctx), reverse=True)
                return result
            except BaseException as exc:
                self._on_failure(exc, input, ctx, hooks)
                raise
            finally:
                _finally(hooks, "finally_agent", input, ctx)

    # ── 스트리밍 ─────────────────────────────────────────────────────────
    #   스트림에도 출력 검사를 건다:
    #   · after 파이프라인이 있는 노드(@klafi_node(after=...))의 LLM 토큰은 내보내지 않고, 노드가 끝나
    #     after 가 적용된 최종 메시지를 토큰 청크 하나로 내보낸다(마스킹 전 원문 유출 방지).
    #   · 스트림이 끝나면 최종 상태에 @klafi_graph after + after_agent 훅을 **판정용**으로 돌린다 —
    #     차단(block)은 예외로 전달되고, 치환은 이미 전송된 스트림에 반영할 수 없어 경고만 남긴다.
    def _stream_setup(self, stream_mode: Any, run_kw: dict[str, Any]) -> "_StreamGate":
        requested = stream_mode if stream_mode is not None else run_kw.pop("stream_mode", None)
        gate = _StreamGate(requested, self.compiled.stream_mode, self._guarded_nodes,
                           passthrough=bool(run_kw.get("subgraphs")),
                           internal=self._internal_nodes, structured=self._structured_nodes)
        run_kw["stream_mode"] = gate.internal_modes
        return gate

    def _stream_finish(self, gate: "_StreamGate", input: Any, ctx: ExecutionContext, hooks: list[Hook]) -> None:
        if gate.values is None:
            return
        final = self._exit(gate.values, ctx)  # @klafi_graph after — 차단은 raise
        final = _transform(hooks, "after_agent", final, lambda v: (input, v, ctx), reverse=True)
        if final is not gate.values:
            _log.warning(
                "stream: after 파이프라인/출력 가드레일의 치환은 이미 전송된 스트림에 반영되지 않습니다"
                " (차단만 유효). 마스킹이 필요한 노드는 @klafi_node(after=...) 로 선언하세요."
            )

    def stream(
        self,
        input: Any,
        context: ExecutionContext | None = None,
        thread_id: str | None = None,
        *,
        stream_mode: Any = None,  # LangGraph stream_mode. "messages"면 LLM 토큰 단위 스트리밍.
        **kwargs: Any,
    ) -> Iterator[Any]:
        ctx = self._make_context(context)
        cfg_kw, run_kw = self._split_kwargs(kwargs)
        with bind_context(ctx):
            # 제너레이터가 소진될 때까지 Context를 유지해야 하므로 with 안에서 yield
            hooks = self._resolved_hooks()
            input = _transform(hooks, "before_agent", input, lambda v: (v, ctx))
            try:
                input = self._enter(input, ctx)  # @klafi_graph before 파이프라인
                gate = self._stream_setup(stream_mode, run_kw)
                with bind_hooks(hooks):
                    for item in self.compiled.stream(
                        input, config=self._config(ctx, cfg_kw, thread_id), **run_kw
                    ):
                        yield from gate.feed(item, ctx)
                self._stream_finish(gate, input, ctx, hooks)
            except BaseException as exc:
                self._on_failure(exc, input, ctx, hooks)
                raise
            finally:
                _finally(hooks, "finally_agent", input, ctx)

    async def astream(
        self,
        input: Any,
        context: ExecutionContext | None = None,
        thread_id: str | None = None,
        *,
        stream_mode: Any = None,  # LangGraph stream_mode. "messages"면 LLM 토큰 단위 스트리밍.
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        ctx = self._make_context(context)
        cfg_kw, run_kw = self._split_kwargs(kwargs)
        with bind_context(ctx):
            hooks = self._resolved_hooks()
            input = _transform(hooks, "before_agent", input, lambda v: (v, ctx))
            try:
                input = await self._aenter(input, ctx)  # @klafi_graph before 파이프라인
                gate = self._stream_setup(stream_mode, run_kw)
                with bind_hooks(hooks):
                    async for item in self.compiled.astream(
                        input, config=self._config(ctx, cfg_kw, thread_id), **run_kw
                    ):
                        for out in gate.feed(item, ctx):
                            yield out
                final = await self._aexit(gate.values, ctx) if gate.values is not None else None
                if final is not None:
                    final = _transform(hooks, "after_agent", final, lambda v: (input, v, ctx), reverse=True)
                    if final is not gate.values:
                        _log.warning(
                            "stream: after 파이프라인/출력 가드레일의 치환은 이미 전송된 스트림에 반영되지"
                            " 않습니다 (차단만 유효). 마스킹이 필요한 노드는 @klafi_node(after=...) 로 선언하세요."
                        )
            except BaseException as exc:
                self._on_failure(exc, input, ctx, hooks)
                raise
            finally:
                _finally(hooks, "finally_agent", input, ctx)

    # ── 내부 ────────────────────────────────────────────────────────────
    def _make_context(self, context: ExecutionContext | None) -> ExecutionContext:
        if context is None:
            return ExecutionContext.new(
                agent_id=self.spec.id,
                agent_version=self.spec.version,
                project_id=self.spec.project,
            )
        # caller가 준 context에 Agent 신원이 없으면 spec에서 채운다(Trace에 항상 agent_id 확보).
        if context.agent_id is None:
            context.agent_id = self.spec.id
        if context.agent_version is None:
            context.agent_version = self.spec.version
        if context.project_id is None:
            context.project_id = self.spec.project
        return context

    def _config(
        self, ctx: ExecutionContext, extra: dict[str, Any], thread_id: str | None = None
    ) -> dict[str, Any]:
        # LangGraph thread_id는 checkpoint/resume의 키 (MEM-05).
        ctx.thread_id = self._thread_id(ctx, thread_id)  # 실제 사용된 thread — span/로그 상관 ID 와 일치
        configurable = {
            "thread_id": ctx.thread_id,
            "execution_id": ctx.execution_id,
            "trace_id": ctx.trace_id,
        }
        # extra 를 파괴하지 않는다(pop 금지) — invoke 는 재시도마다 같은 kwargs 로 _config 를
        # 다시 부르므로, pop 하면 2회차부터 사용자 config(recursion_limit 등)가 사라진다.
        user_config = dict(extra.get("config") or {})
        merged = {k: v for k, v in extra.items() if k != "config"}
        user_config.setdefault("configurable", {})
        user_config["configurable"] = {**configurable, **user_config["configurable"]}
        user_config.update(merged)
        return user_config


def _hooked_runnable(runnable: Any, name: str, get_hooks: Any) -> tuple[Any, Any]:
    """순수 Runnable(컴파일된 서브그래프 등)을 노드 훅으로 감싼 (sync, async) 함수 쌍.

    functools.wraps 를 쓰지 않고 bound method `inv`/`ainv` 를 본문에서 직접 참조한다 — LangGraph 는
    노드 함수의 소스와 클로저(nonlocals)를 읽어 서브그래프를 찾으므로(find_subgraph_pregel),
    이렇게 해야 get_subgraphs()/stream(subgraphs=True) 가 그대로 동작한다. config 파라미터를 받아
    checkpoint_ns·callbacks 가 서브그래프까지 전파되게 한다.
    """
    from langchain_core.runnables import RunnableConfig

    from .hook import arun_hooked, run_hooked

    inv, ainv = runnable.invoke, runnable.ainvoke

    def sync_node(state: Any, config: RunnableConfig | None = None) -> Any:
        return run_hooked(name, get_hooks, state, lambda: inv(state, config=config))

    async def async_node(state: Any, config: RunnableConfig | None = None) -> Any:
        return await arun_hooked(name, get_hooks, state, lambda: ainv(state, config=config))

    from typing import Optional

    for f in (sync_node, async_node):  # `from __future__ import annotations` 라 문자열 — LangGraph 검사는 실제 타입을 본다
        f.__annotations__["config"] = Optional[RunnableConfig]
    return sync_node, async_node


class _StreamGate:
    """스트림 항목 필터. 요청 모드만 원래 모양으로 내보내고, 내부적으로 values(최종 상태)·updates(노드
    완료 신호)를 추가 구독한다.
      · after 파이프라인·구조화 출력 노드의 messages 토큰은 보류했다가 노드 완료 시 최종 메시지 하나로 대체.
      · visibility="internal" 노드의 messages·updates 는 내보내지 않는다.
      · output 계약 노드는 완료 시 ("structured", {node, key, data}) 를 추가로 낸다(리스트 모드 요청일 때만 —
        단일 모드는 항목 모양이 payload 뿐이라 새 모드를 섞을 수 없다).
    subgraphs=True 는 항목 모양이 달라 그대로 통과시킨다.
    """

    def __init__(
        self,
        requested: Any,
        default_mode: str,
        guarded: set[str],
        *,
        passthrough: bool = False,
        internal: set[str] | None = None,
        structured: dict[str, str | None] | None = None,
    ) -> None:
        self.single = not isinstance(requested, (list, tuple))
        modes = [requested if requested is not None else default_mode] if self.single else list(requested)
        self.requested = set(modes)
        self.internal_modes = [*modes, *(m for m in ("values", "updates") if m not in modes)]
        self.guarded = guarded if not passthrough else set()
        self.internal = set(internal or ()) if not passthrough else set()
        self.structured = dict(structured or {}) if not passthrough else {}
        self.passthrough = passthrough
        self.values: Any = None
        self._hold: dict[str, Any] = {}  # node → 마지막 토큰 메타데이터(보류 중 표시)
        if passthrough:
            self.internal_modes = modes

    def feed(self, item: Any, ctx: ExecutionContext) -> list[Any]:
        if self.passthrough:
            return [item]
        mode, payload = item
        if mode == "values":
            self.values = payload
        if mode == "updates" and isinstance(payload, dict) and payload.get("__interrupt__"):
            ctx.state = "WAITING_APPROVAL"
        out: list[Any] = []
        if mode == "messages" and (self.guarded or self.internal):
            msg, meta = payload if isinstance(payload, tuple) else (payload, {})
            node = (meta or {}).get("langgraph_node")
            if node in self.internal:
                return out  # 내부 노드의 토큰은 호출자에게 가지 않는다
            if node in self.guarded:
                self._hold[node] = meta
                return out  # 원문 토큰은 내보내지 않는다
        if mode == "updates" and isinstance(payload, dict):
            if self.internal:
                payload = {n: u for n, u in payload.items() if n not in self.internal}
                if not payload:
                    return out
            for node, upd in payload.items():
                if node in self._hold and "messages" in self.requested:
                    meta = self._hold.pop(node)
                    final = _final_ai_chunk(upd)
                    if final is not None:
                        out.append(self._emit("messages", (final, meta)))
                self._hold.pop(node, None)
                if node in self.structured and not self.single and isinstance(upd, dict):
                    key = self.structured[node] or next(
                        (k for k, v in upd.items() if k != "messages" and hasattr(v, "model_dump")), None
                    )
                    if key is not None:
                        out.append(("structured", {"node": node, "key": key, "data": upd.get(key)}))
        if mode in self.requested:
            out.append(self._emit(mode, payload))
        return out

    def _emit(self, mode: str, payload: Any) -> Any:
        return payload if self.single else (mode, payload)


def _final_ai_chunk(update: Any) -> Any:
    """노드 업데이트에서 마지막 AI 메시지를 토큰 청크 하나로 (after 파이프라인이 적용된 최종 본문)."""
    msgs = update.get("messages") if isinstance(update, dict) else None
    if not msgs:
        return None
    m = msgs[-1] if isinstance(msgs, list) else msgs
    if getattr(m, "type", None) != "ai":
        return None
    from langchain_core.messages import AIMessageChunk

    return AIMessageChunk(content=m.content, id=getattr(m, "id", None))
