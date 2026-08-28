"""KlafiApp — 애플리케이션 부트스트랩 (공통개발자 영역).

Spring Boot 비유:
    config/*.yaml   ≈ application.yml        (공통개발자: 모델·정책·보안·포트)
    KlafiApp        ≈ @Configuration/Bean    (공통개발자: DataSource·인프라 빈)
    agents/*.py     ≈ @Controller/@Service   (업무개발자: 업무 로직만)

공통개발자는 config로 모델 연결·정책·Guardrail·Checkpoint를 한곳에서 관리하고,
업무개발자는 각 Agent 파일에서 KlafiGraph 하위 클래스(State/Node/Edge)만 작성한다.
KlafiApp이 공유 리소스를 주입해 실행 가능한 Agent로 조립·등록·서비스한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from klafi.config.layered import LayeredConfig
from klafi.core.exceptions import ConfigSchemaError
from klafi.core.graph import KlafiGraph
from klafi.core.hook import Hook
from klafi.model import ModelGateway
from klafi.observability.logging import setup_logging
from klafi.observability.tracing import setup_tracing
from klafi.registry import AgentRegistry
from klafi.runtime.factory import ExecutionFactory
from klafi.runtime.policy import ExecutionPolicy


_PROVIDER_KEYS = {"type", "model", "cost", "fallback", "policy"}


def _build_gateway(model_cfg: dict[str, Any]) -> ModelGateway:
    from klafi.model.providers import resolve_provider

    gw = ModelGateway()
    for alias, spec in (model_cfg.get("providers") or {}).items():
        unknown = set(spec) - _PROVIDER_KEYS  # 오타를 조용히 무시하지 않는다(다른 config와 일관)
        if unknown:
            raise ConfigSchemaError(
                f"provider '{alias}' 설정에 알 수 없는 항목: {sorted(unknown)} "
                f"(가능: {sorted(_PROVIDER_KEYS)})"
            )
        cost = tuple(spec["cost"]) if spec.get("cost") else None
        policy = ExecutionPolicy.from_config(spec.get("policy"))  # MOD-04/05
        gw.register(
            alias,
            resolve_provider(spec),  # provider type → registry (register_provider 로 확장 가능)
            cost=cost,
            fallback=spec.get("fallback"),  # MOD-08 — 이제 config 경로에서도 도달 가능
            policy=policy,
        )
    return gw


_CONTEXT_KEYS = {"max_tokens", "keep_recent", "model", "key"}


def _register_context_hook(cfg: dict[str, Any], gateway: ModelGateway) -> None:
    """config/context.yaml → 이름 'context' 훅으로 등록 (hooks.yaml에서 참조)."""
    from klafi.context.hook import ContextHook
    from klafi.hookdefs import register_named_hook

    unknown = set(cfg) - _CONTEXT_KEYS
    if unknown:
        raise ConfigSchemaError(
            f"context 설정에 알 수 없는 항목: {sorted(unknown)} (가능: {sorted(_CONTEXT_KEYS)})"
        )
    alias = cfg.get("model")
    summarizer = gateway.model(alias) if alias else None  # 없으면 요약 없이 오래된 건 제거
    params = {k: v for k, v in cfg.items() if k in _CONTEXT_KEYS - {"model"}}
    register_named_hook("context", lambda: ContextHook(summarizer=summarizer, **params))


@dataclass
class KlafiApp:
    config: LayeredConfig
    gateway: ModelGateway
    policy: ExecutionPolicy | None
    base_hooks: list[Hook]
    checkpoint: Any
    store: Any
    factory: ExecutionFactory  # F02 — Spec→실행환경 조립
    hook_plan: Any = None  # HookPlan — config/hooks.yaml 로 훅·가드레일 에이전트별 적용
    server: Any = None  # AgentServer (lazy, server extra 필요)
    registry: AgentRegistry = field(default_factory=AgentRegistry)

    @classmethod
    def from_config(
        cls,
        config_dir: str,
        *,
        environment: str | None = None,
        platform_hooks: list[Hook] | None = None,
    ) -> "KlafiApp":
        cfg = LayeredConfig.from_dir(config_dir, environment=environment)

        # 공통 관측 부트스트랩 — 로깅·트레이싱은 프레임워크가 소유한다(프로젝트 복붙 제거).
        setup_logging()
        setup_tracing(service_name=cfg.get("service", "klafi"))

        # 코드 경로(이중관리): 공통개발자가 플랫폼 전역 Hook을 코드로 추가(EventHook, Metrics 등).
        # 선언 경로: config/hooks.yaml (훅·가드레일). Logging/Tracing은 KlafiGraph 기본 탑재.
        base_hooks: list[Hook] = list(platform_hooks or [])
        from pathlib import Path

        from .hookplan import HookPlan

        gateway = _build_gateway(cfg.get("model", {}) or {})

        # context.yaml이 있으면 'context' 훅 등록 (hooks.yaml 검증 전에 해야 함)
        context_cfg = cfg.get("context")
        if context_cfg:
            _register_context_hook(context_cfg, gateway)

        hook_plan = HookPlan.from_file(Path(config_dir) / "hooks.yaml")  # 고정 위치
        hook_plan.validate_names()  # 미등록 훅·가드레일 이름을 기동 시점에 검출
        policy = ExecutionPolicy.from_config(cfg.get("policy"))
        checkpoint = cfg.get("checkpoint")
        store = cfg.get("store")
        factory = ExecutionFactory(
            gateway=gateway, checkpointer=checkpoint, store=store, policy=policy, base_hooks=base_hooks
        )
        return cls(
            config=cfg,
            gateway=gateway,
            policy=policy,
            base_hooks=base_hooks,
            checkpoint=checkpoint,
            store=store,
            factory=factory,
            hook_plan=hook_plan,
        )

    # ── Agent 조립 (공유 리소스 주입) ────────────────────────────────────
    def create(self, agent_cls: type[KlafiGraph], *, extra_hooks: list[Hook] | None = None) -> KlafiGraph:
        # config/hooks.yaml 로 해석한 에이전트별 훅·가드레일 + 코드 extra_hooks
        planned = self.hook_plan.for_agent(agent_cls.spec.id) if self.hook_plan else []
        return self.factory.create(agent_cls, extra_hooks=[*planned, *(extra_hooks or [])])

    def register(
        self,
        agent_cls: type[KlafiGraph],
        *,
        extra_hooks: list[Hook] | None = None,
        owner: str | None = None,
    ) -> KlafiGraph:
        agent = self.create(agent_cls, extra_hooks=extra_hooks)
        self.registry.register_agent(agent, owner=owner)  # Governance 기록
        self._server().register(agent, agent_id=agent.spec.id)  # 런타임 서비스 등록
        return agent

    def memory(self, pii_filter: Any = None) -> Any:
        """플랫폼 공통 Long-Term Memory 래퍼 (Factory가 공유하는 Store)."""
        if self.factory.store is None:
            return None
        from klafi.context.memory import MemoryStore

        return MemoryStore(self.factory.store, pii_filter=pii_filter)

    def _server(self) -> Any:
        if self.server is None:
            from klafi.server import AgentServer

            self.server = AgentServer()
        return self.server

    def http_app(self, auth: Any = None) -> Any:
        """FastAPI 앱 반환 (server extra 필요). 등록된 모든 Agent를 서비스.

        auth: Request→security_context 어댑터(API-10). Tool 권한 등을 요청에서 주입.
        """
        from klafi.server import create_app

        # policy.yaml 의 concurrency → 서버 전역 동시 실행 상한(초과 시 429)
        max_conc = getattr(self.policy, "concurrency", None) if self.policy else None
        return create_app(self._server(), auth=auth, max_concurrency=max_conc)
