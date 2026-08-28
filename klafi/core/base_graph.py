"""BaseGraph — KLAFI 표준 Agent 실행 진입점 (요구사항 F01).

개발자는 build()에서 순수 LangGraph StateGraph를 그리기만 한다.
BaseGraph는 실행 시 ExecutionContext(execution_id/trace_id 포함)를 만들어
실행 Scope에 바인딩하고, LangGraph config에 thread_id·metadata를 얹는다.

Open Framework 원칙(§3.3): 컴파일된 LangGraph는 .compiled로 그대로 노출한다.
개발자가 원하면 StateGraph native API에 언제든 직접 접근할 수 있다.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator

from langgraph.graph.state import CompiledStateGraph, StateGraph

from .context import ExecutionContext, bind_context, get_context
from .exceptions import AgentExecutionException
from .hook import (
    Hook,
    _after,
    _before,
    _error,
    _finally,
    bind_hooks,
    is_control_flow,
    resolve_hooks,
    wrap_node,
)
from .spec import AgentSpec


class BaseGraph:
    def __init__(
        self,
        spec: AgentSpec,
        checkpointer: Any = None,
        hooks: list[Hook] | None = None,
        policy: Any = None,
        store: Any = None,
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
        graph = self.build()
        if not isinstance(graph, StateGraph):
            raise AgentExecutionException(
                "build()는 langgraph StateGraph를 반환해야 합니다",
                agent_id=spec.id,
            )
        self._install_node_hooks(graph)
        self.compiled: CompiledStateGraph = graph.compile(
            checkpointer=self.checkpointer, store=self.store
        )

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
            if getattr(runnable, "func", None) is not None:
                runnable.func = wrap_node(runnable.func, name, self._resolved_hooks, is_async=False)
            if getattr(runnable, "afunc", None) is not None:
                runnable.afunc = wrap_node(runnable.afunc, name, self._resolved_hooks, is_async=True)

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
        with bind_context(ctx):
            hooks = self._resolved_hooks()
            _before(hooks, "before_agent", input, ctx)
            try:
                input = self._enter(input, ctx)  # @klafi_graph before 파이프라인
                payload = self._retry_payload(input)
                result = self._run_sync(
                    lambda: self.compiled.invoke(
                        payload(), config=self._config(ctx, kwargs, thread_id)
                    ),
                    ctx,
                    pol,
                    hooks,
                )
                self._mark_interrupt_state(ctx, result)
                result = self._exit(result, ctx)  # @klafi_graph after 파이프라인
                _after(hooks, "after_agent", input, result, ctx)
                return result
            except BaseException as exc:
                if is_control_flow(exc):  # interrupt(HITL) → 실패 아님, WAITING_APPROVAL
                    ctx.state = "WAITING_APPROVAL"
                else:
                    self._graph_error(exc, input, ctx)
                    _error(hooks, "on_agent_error", input, exc, ctx)
                raise
            finally:
                _finally(hooks, "finally_agent", input, ctx)

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
        with bind_context(ctx):
            hooks = self._resolved_hooks()
            _before(hooks, "before_agent", input, ctx)
            try:
                input = await self._aenter(input, ctx)  # @klafi_graph before 파이프라인
                payload = self._retry_payload(input)
                result = await self._run_async(
                    lambda: self.compiled.ainvoke(
                        payload(), config=self._config(ctx, kwargs, thread_id)
                    ),
                    ctx,
                    pol,
                    hooks,
                )
                self._mark_interrupt_state(ctx, result)
                result = await self._aexit(result, ctx)  # @klafi_graph after 파이프라인
                _after(hooks, "after_agent", input, result, ctx)
                return result
            except BaseException as exc:
                if is_control_flow(exc):  # interrupt(HITL) → 실패 아님, WAITING_APPROVAL
                    ctx.state = "WAITING_APPROVAL"
                else:
                    self._graph_error(exc, input, ctx)
                    _error(hooks, "on_agent_error", input, exc, ctx)
                raise
            finally:
                _finally(hooks, "finally_agent", input, ctx)

    def stream(
        self,
        input: Any,
        context: ExecutionContext | None = None,
        thread_id: str | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        ctx = self._make_context(context)
        with bind_context(ctx):
            # 제너레이터가 소진될 때까지 Context를 유지해야 하므로 with 안에서 yield
            hooks = self._resolved_hooks()
            _before(hooks, "before_agent", input, ctx)
            try:
                input = self._enter(input, ctx)  # @klafi_graph before 파이프라인
                # TODO(stream): after 파이프라인(가드레일·미들웨어)은 스트리밍에 미적용.
                #   스트림은 단일 결과가 없어 "최종 출력"을 검사할 지점이 없다(after_agent 훅도
                #   같은 이유로 stream에서는 발화하지 않는다). 노드 단위 @klafi_node(output=)는
                #   스트리밍에서도 그대로 동작하므로 검사 공백은 아니다.
                #   방향: LLM 노드를 invoke/stream/structured/internal 로 구분해 노드 종류별로
                #   출력 검사 시점을 정의한다(structured는 invoke 전용, internal은 미전달).
                with bind_hooks(hooks):
                    yield from self.compiled.stream(
                        input, config=self._config(ctx, kwargs, thread_id)
                    )
            except BaseException as exc:
                if is_control_flow(exc):  # interrupt(HITL) → 실패 아님, WAITING_APPROVAL
                    ctx.state = "WAITING_APPROVAL"
                else:
                    _error(hooks, "on_agent_error", input, exc, ctx)
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
        with bind_context(ctx):
            hooks = self._resolved_hooks()
            _before(hooks, "before_agent", input, ctx)
            try:
                input = await self._aenter(input, ctx)  # @klafi_graph before 파이프라인
                # TODO(stream): after 파이프라인 미적용 — stream() 의 TODO 참조.
                with bind_hooks(hooks):
                    opts = {"stream_mode": stream_mode} if stream_mode is not None else {}
                    async for chunk in self.compiled.astream(
                        input, config=self._config(ctx, kwargs, thread_id), **opts
                    ):
                        yield chunk
            except BaseException as exc:
                if is_control_flow(exc):  # interrupt(HITL) → 실패 아님, WAITING_APPROVAL
                    ctx.state = "WAITING_APPROVAL"
                else:
                    _error(hooks, "on_agent_error", input, exc, ctx)
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
        configurable = {
            "thread_id": self._thread_id(ctx, thread_id),
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
