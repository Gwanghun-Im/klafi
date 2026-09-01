"""Execution Factory (요구사항 §7, F02 / FAC-01~09).

KlafiGraph 하위 클래스로 실행환경을 조립해 Runnable Agent를 만든다.
개발자가 Model/Checkpointer/Store/Policy/Hook을 직접 연결하지 않고, Factory가 주입한다.

    factory = ExecutionFactory(gateway=gw, checkpointer="memory",
                               policy=ExecutionPolicy(timeout=30), base_hooks=[...])
    agent = factory.create(QAAgent)      # QAAgent.spec.model alias → gateway로 주입
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from klafi.core.exceptions import AgentExecutionException
from klafi.core.graph import KlafiGraph
from klafi.core.hook import Hook


@dataclass
class ExecutionFactory:
    gateway: Any = None  # ModelGateway (FAC-03 Model 주입)
    checkpointer: Any = None  # FAC-02
    store: Any = None  # FAC-04
    policy: Any = None  # FAC-06 (ExecutionPolicy)
    base_hooks: list[Hook] = field(default_factory=list)  # FAC-07

    def __post_init__(self) -> None:
        # checkpointer/store를 한 번만 인스턴스로 해석해 생성되는 모든 Agent가 공유한다.
        from klafi.context.checkpoint import resolve_checkpointer
        from klafi.context.memory import resolve_store

        self.checkpointer = resolve_checkpointer(self.checkpointer)
        self.store = resolve_store(self.store)
        # gateway는 여기서 전역에 박지 않는다 — create() 조립 구간에만 바인딩(멀티팩토리 격리).

    def model(self, alias: str | None) -> Any:
        if not alias or self.gateway is None:
            return None
        return self.gateway.model(alias)

    def create(
        self, agent_cls: type[KlafiGraph], *, extra_hooks: list[Hook] | None = None, policy: Any = None
    ) -> KlafiGraph:
        """KlafiGraph 하위 클래스로 실행환경이 조립된 Agent 인스턴스 생성 (FAC-01).

        policy: 에이전트별 유효 policy 오버라이드. None 이면 factory 공통 policy 를 쓴다
        (per-agent config.yaml → app.register 가 전역 위에 머지해 넘긴다).

        조립 계약: agent_cls 는 (1) 클래스 레벨 `spec` 을 갖고 (2) factory 가 주입하는 키워드
        (spec/model/checkpointer/store/policy/hooks) 만으로 생성 가능해야 한다.
        생성자에 다른 필수 인자를 요구하는 템플릿(RAGAgent=retriever, SupervisorAgent=router/
        workers 등)은 이 계약을 만족하지 않으므로, 직접 인스턴스화해 server.register(agent) 로
        넘기거나 그 의존성을 바인딩한 서브클래스를 만들어 등록한다(친절한 에러로 안내됨).
        """
        from klafi.model.gateway import using_gateway

        spec = getattr(agent_cls, "spec", None)
        if spec is None:
            raise AgentExecutionException(
                f"{agent_cls.__name__}.spec (클래스 속성)을 지정하세요 — "
                "factory.create/app.register 는 클래스 레벨 spec 으로 조립합니다"
            )
        # define()에서 init_chat_model(alias)이 이 factory의 gateway를 보도록, 조립 동안만 바인딩.
        # 조립이 끝나면 해제되므로 다른 factory의 gateway와 섞이지 않는다.
        with using_gateway(self.gateway):
            try:
                return agent_cls(  # chat model은 define()에서 init_chat_model(alias)로 선언한다
                    spec=spec,
                    model=self.model(spec.model),
                    checkpointer=self.checkpointer,
                    store=self.store,
                    policy=policy if policy is not None else self.policy,
                    hooks=[*self.base_hooks, *(extra_hooks or [])],
                )
            except TypeError as exc:
                # 생성자에 factory가 못 주는 필수 인자가 있는 클래스(RAGAgent=retriever,
                # SupervisorAgent=router/workers 등). 이런 템플릿은 인스턴스로 만들어 서빙에 넘긴다.
                if "argument" in str(exc):
                    raise AgentExecutionException(
                        f"{agent_cls.__name__} 은 factory.create/app.register 로 조립할 수 없습니다 "
                        f"({exc}). 생성자에 필수 의존성이 있는 템플릿(RAGAgent·SupervisorAgent 등)은 "
                        "직접 인스턴스화해 server.register(agent) 로 넘기거나, 필수 의존성을 바인딩한 "
                        "서브클래스를 만들어 등록하세요"
                    ) from exc
                raise
