"""KlafiGraph — 개발자가 상속해 그래프를 조립하는 KLAFI 표준 그래프 클래스.

BaseGraph(실행·Hook·정책·Context·Checkpoint)를 상속하고, LangGraph StateGraph의
빌더 API(add_node/add_edge/...)를 그대로 노출한다. 개발자는:

    class MyAgent(KlafiGraph):
        state_schema = State
        def define(self):
            self.add_node("plan", plan)
            self.add_edge(START, "plan")
            self.add_edge("plan", END)

    agent = MyAgent(spec, checkpointer="memory")
    agent.invoke({...})

- state_schema: LangGraph State 타입 (필수).
- define(): add_node/add_edge 로 그래프를 조립 (필수 구현).
- 모델 선언: define() 안에서 init_chat_model("<alias>") — config/model.yaml의 alias.
- 기본적으로 Logging/Tracing Hook이 탑재된다(observability=False로 끌 수 있음).
- 컴파일된 LangGraph는 .compiled, 원본 빌더는 .builder 로 그대로 접근 가능.
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph.state import StateGraph

from .base_graph import BaseGraph
from .exceptions import AgentExecutionException
from .hook import Hook
from .logging_hook import LoggingHook
from .spec import AgentSpec


class KlafiGraph(BaseGraph):
    state_schema: type | None = None  # 하위 클래스가 지정
    spec: AgentSpec | None = None  # 클래스 속성으로 지정하거나 생성자 인자로 전달
    observability: bool = True  # 기본 Logging/Tracing Hook 탑재 여부

    def __init__(
        self,
        spec: AgentSpec | None = None,
        *,
        model: Callable[[str], str] | None = None,
        checkpointer: Any = None,
        policy: Any = None,
        store: Any = None,
        hooks: list[Hook] | None = None,
    ) -> None:
        if self.state_schema is None:
            raise AgentExecutionException("KlafiGraph 하위 클래스는 state_schema를 지정해야 합니다")
        resolved_spec = spec or type(self).spec
        if resolved_spec is None:
            raise AgentExecutionException("spec을 생성자 인자 또는 클래스 속성으로 지정하세요")

        self._sg = StateGraph(self.state_schema)
        self.model = model  # (prompt)->str — Template용. chat model은 init_chat_model(alias).

        base: list[Hook] = []
        if self.observability:
            from klafi.observability.tracing import TracingHook

            base = [LoggingHook(), TracingHook()]
        # 워크플로우 경계 미들웨어·가드레일은 @klafi_graph 가 클래스에 붙이고
        # BaseGraph 실행 파이프라인이 직접 적용한다(훅이 아님 — 훅은 값을 교체할 수 없다).
        super().__init__(
            resolved_spec,
            checkpointer=checkpointer,
            policy=policy,
            store=store,
            hooks=[*base, *(hooks or [])],
        )

    # BaseGraph.__init__가 호출하는 내부 훅
    def build(self) -> StateGraph:
        self.define()
        self._require_klafi_nodes()
        return self._sg

    def _require_klafi_nodes(self) -> None:
        """모든 노드 함수는 @klafi_node 로 선언해야 한다(강제). ToolNode 등 Runnable은 예외."""
        from langgraph.prebuilt import ToolNode

        for node_name, node_spec in self._sg.nodes.items():
            runnable = getattr(node_spec, "runnable", None)
            if isinstance(runnable, ToolNode):
                continue  # 프레임워크 제공 ToolNode → 예외
            fn = getattr(runnable, "func", None) or getattr(runnable, "afunc", None)
            if fn is None:
                continue  # 함수가 아닌 노드(다른 Runnable) → 예외
            if not getattr(fn, "__klafi_node__", False):
                raise AgentExecutionException(
                    f"노드 '{node_name}' 는 @klafi_node 로 선언해야 합니다 "
                    f"(예: @klafi_node(\"{node_name}\")). ToolNode 는 예외입니다.",
                    agent_id=getattr(self.spec, "id", None),
                )

    def define(self) -> None:  # pragma: no cover - 추상
        raise NotImplementedError("define()에서 add_node/add_edge로 그래프를 조립하세요")

    @property
    def builder(self) -> StateGraph:
        """원본 LangGraph StateGraph (고급 사용, Open Framework)."""
        return self._sg

    # ── Tool 바인딩 (LangGraph 네이티브 방식) ────────────────────────────
    #   llm = init_chat_model("main").bind_tools([toolA])
    #   llm = bind_skills(init_chat_model("main"), [skillA])     # 툴 + 지침
    #   self.add_node("tools", self.make_tool_node([toolA]))
    def make_tool_node(self, tools: list[Any]) -> Any:
        """주어진 tools로 LangGraph ToolNode 생성 (노드별로 서로 다른 툴셋 가능).

        KLAFI Tool·Skill은 LangChain tool로 자동 변환된다. ToolNode가 실행해도
        권한·검증·Timeout·Hook은 KLAFI Tool.run에서 그대로 적용된다.
        """
        from langgraph.prebuilt import ToolNode

        from klafi.tool.tool import to_langchain_tools

        return ToolNode(to_langchain_tools(tools))

    # ── StateGraph 빌더 위임 (LangGraph API 그대로) ─────────────────────
    def add_node(self, *args: Any, **kwargs: Any) -> "KlafiGraph":
        self._sg.add_node(*args, **kwargs)
        return self

    def add_edge(self, *args: Any, **kwargs: Any) -> "KlafiGraph":
        self._sg.add_edge(*args, **kwargs)
        return self

    def add_conditional_edges(self, *args: Any, **kwargs: Any) -> "KlafiGraph":
        self._sg.add_conditional_edges(*args, **kwargs)
        return self

    def add_sequence(self, *args: Any, **kwargs: Any) -> "KlafiGraph":
        self._sg.add_sequence(*args, **kwargs)
        return self

    def set_entry_point(self, *args: Any, **kwargs: Any) -> "KlafiGraph":
        self._sg.set_entry_point(*args, **kwargs)
        return self

    def set_finish_point(self, *args: Any, **kwargs: Any) -> "KlafiGraph":
        self._sg.set_finish_point(*args, **kwargs)
        return self

    def set_conditional_entry_point(self, *args: Any, **kwargs: Any) -> "KlafiGraph":
        self._sg.set_conditional_entry_point(*args, **kwargs)
        return self


# ── @klafi_graph — 워크플로우 경계 미들웨어·가드레일 ─────────────────────
GRAPH_ATTR = "__klafi_graph__"


class GraphSpec:
    """@klafi_graph 가 클래스에 남기는 선언. BaseGraph 실행 파이프라인이 읽어 적용한다."""

    __slots__ = ("before", "after", "on_error")

    def __init__(self, before: Any, after: Any, on_error: Any) -> None:
        from .middleware import as_list

        self.before = as_list(before)
        self.after = as_list(after)
        self.on_error = as_list(on_error)


def klafi_graph(
    cls: type | None = None,
    *,
    before: Any = None,
    after: Any = None,
    on_error: Any = None,
) -> Any:
    """워크플로우(그래프 전체) 경계에 가드레일·미들웨어를 붙인다 — @klafi_node 의 그래프판.

    before/after 는 @klafi_node 와 **완전히 같은 계약**이다. 한 리스트에 가드레일과 미들웨어를
    섞어 넣고 원소 타입(`.check` 보유 여부)으로 구분한다.
      - before : 그래프에 들어가는 input
      - after  : 그래프가 돌려준 result
      - on_error(exc, input[, ctx]) : 예외 관측(그 뒤 재발생)

    발화 순서:
      공통/에이전트 훅 → **before 파이프라인** → 그래프 실행 → **after 파이프라인** → 훅

        @klafi_graph(before=[require_login, no_secrets], after=[mask_pii])
        class MyAgent(KlafiGraph): ...

    노드 단위 적용은 @klafi_node 를 쓴다.
    스트리밍에서는 after 파이프라인이 적용되지 않는다(BaseGraph.stream 의 TODO 참조).
    """

    def deco(target: Any) -> Any:
        if not isinstance(target, type):
            raise AgentExecutionException(
                "@klafi_graph 는 KlafiGraph 하위 클래스에만 붙입니다 (노드 함수에는 @klafi_node)"
            )
        setattr(target, GRAPH_ATTR, GraphSpec(before, after, on_error))
        return target

    return deco(cls) if cls is not None else deco
