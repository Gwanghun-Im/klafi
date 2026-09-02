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


def _resolve_platform_hook(h: Any, gateway: ModelGateway) -> Hook:
    """platform_hooks 항목 해석 — Hook 인스턴스는 그대로, 콜러블은 (gateway)->Hook 팩토리로 호출."""
    if isinstance(h, Hook):
        return h
    if callable(h):
        made = h(gateway)
        if isinstance(made, Hook):
            return made
        raise ConfigSchemaError(f"platform_hooks 팩토리 {getattr(h, '__name__', h)!r} 가 Hook 이 아닌 {type(made).__name__} 을 반환했습니다")
    raise ConfigSchemaError(f"platform_hooks 항목은 Hook 인스턴스 또는 (gateway)->Hook 팩토리여야 합니다: {h!r}")


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
    # per-agent config.yaml 이 명시한 동시성 상한 {agent_id: n} — http_app 이 미들웨어에 넘긴다.
    _agent_concurrency: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config_dir: str,
        *,
        environment: str | None = None,
        platform_hooks: "list[Hook | Any] | None" = None,
    ) -> "KlafiApp":
        cfg = LayeredConfig.from_dir(config_dir, environment=environment)

        # 공통 관측 부트스트랩 — 로깅·트레이싱은 프레임워크가 소유한다(프로젝트 복붙 제거).
        setup_logging()
        setup_tracing(service_name=cfg.get("service", "klafi"))

        from pathlib import Path

        from .hookplan import HookPlan

        gateway = _build_gateway(cfg.get("model", {}) or {})

        # 플랫폼 공통 훅 — 통일 계약: 항목은 Hook 인스턴스 **또는** (gateway)->Hook 팩토리.
        # gateway 가 필요한 훅(ContextHook·LLM 가드레일 등)도 같은 리스트에 팩토리로 넣으면
        # 여기서 해석된다 — 공통개발자 배선 지점이 hooks.py 리스트 하나로 통일된다.
        base_hooks: list[Hook] = [_resolve_platform_hook(h, gateway) for h in (platform_hooks or [])]

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
    @staticmethod
    def _agent_config(agent_cls: type[KlafiGraph]) -> dict:
        """에이전트 클래스와 같은 폴더의 config.yaml 을 읽는다 (co-located, 없으면 {}).

        클래스의 모듈 파일 위치에서 찾으므로 자동탐색(register_package)·수동등록 모두 동작한다.
        """
        import sys
        from pathlib import Path

        mod = sys.modules.get(agent_cls.__module__)
        path = getattr(mod, "__file__", None)
        if not path:
            return {}
        cfg = Path(path).parent / "config.yaml"
        if not cfg.exists():
            return {}
        import yaml

        return yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}

    @staticmethod
    def _draw_graph(agent: KlafiGraph) -> None:
        """spec.print=True 면 컴파일된 그래프를 터미널(stdout)에 그린다 (부팅 시 1회).

        grandalf 가 있으면 ASCII, 없으면 mermaid 텍스트로 폴백한다(추가 의존 없이 항상 동작).
        """
        try:
            g = agent.compiled.get_graph()
        except Exception as exc:  # noqa: BLE001 — 그래프 접근 실패는 기동을 막지 않는다
            print(f"[klafi] {agent.spec.id}: 그래프를 그릴 수 없습니다 ({exc})")
            return
        print(f"\n=== agent graph: {agent.spec.id} ({agent.spec.name}) ===")
        try:
            print(g.draw_ascii())  # grandalf 설치 시 ASCII 트리
        except Exception:  # noqa: BLE001 — grandalf 미설치 등 → mermaid 소스로
            print(g.draw_mermaid())

    def create(
        self, agent_cls: type[KlafiGraph], *, extra_hooks: list[Hook] | None = None, policy: Any = None
    ) -> KlafiGraph:
        # config/hooks.yaml 로 해석한 에이전트별 훅·가드레일 + 코드 extra_hooks
        planned = self.hook_plan.for_agent(agent_cls.spec.id) if self.hook_plan else []
        return self.factory.create(agent_cls, extra_hooks=[*planned, *(extra_hooks or [])], policy=policy)

    def register(
        self,
        agent_cls: type[KlafiGraph],
        *,
        extra_hooks: list[Hook] | None = None,
        owner: str | None = None,
    ) -> KlafiGraph:
        # per-agent config.yaml 의 policy 를 전역 위에 머지 → 이 에이전트에만 주입 (없으면 전역 그대로).
        pol = self._agent_config(agent_cls).get("policy")
        effective = (self.policy.merge(pol) if self.policy else ExecutionPolicy.from_config(pol)) if pol else None
        agent = self.create(agent_cls, extra_hooks=extra_hooks, policy=effective)
        if getattr(agent.spec, "print", False):  # 부팅 시 그래프를 터미널에 그린다
            self._draw_graph(agent)
        # 미들웨어용: '명시된' per-agent 동시성만 기록(전역 상속값은 전역 캡이 이미 담당).
        if pol and pol.get("concurrency") is not None:
            self._agent_concurrency[agent.spec.id] = pol["concurrency"]
        self.registry.register_agent(agent, owner=owner)  # Governance 기록
        self._server().register(agent, agent_id=agent.spec.id)  # 런타임 서비스 등록
        return agent

    def register_package(self, package: Any, *, owner: str | None = None) -> "list[KlafiGraph]":
        """package 하위 서브패키지를 훑어 KlafiGraph 하위 클래스를 자동 등록한다 (convention).

        업무개발자는 app/agents/<name>/ 폴더만 떨구면 서비스된다 — bootstrap(공통개발자 영역)을
        건드리지 않는다. `_` 로 시작하는 서브패키지는 건너뛴다(WIP·비활성). owner 는 생략 시
        각 spec.owner 로 폴백한다(레지스트리가 `owner or spec.owner`).

            app.register_package("app.agents")
        """
        import importlib
        import pkgutil

        mod = importlib.import_module(package) if isinstance(package, str) else package
        registered: list[KlafiGraph] = []
        seen: set[type] = set()
        for info in pkgutil.iter_modules(mod.__path__):
            if info.name.startswith("_"):
                continue
            sub = importlib.import_module(f"{mod.__name__}.{info.name}")
            for obj in vars(sub).values():  # 서브패키지 __init__ 이 re-export 한 클래스만 본다
                if (
                    isinstance(obj, type)
                    and issubclass(obj, KlafiGraph)
                    and obj is not KlafiGraph
                    and getattr(obj, "spec", None) is not None  # 구체 에이전트만(템플릿·베이스 제외)
                    and obj not in seen
                ):
                    seen.add(obj)
                    registered.append(self.register(obj, owner=owner))
        return registered

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

        # 2단계 동시성: policy.yaml 의 concurrency = 전역 총량 캡, per-agent config.yaml = 에이전트별 캡.
        max_conc = getattr(self.policy, "concurrency", None) if self.policy else None
        return create_app(
            self._server(),
            auth=auth,
            max_concurrency=max_conc,
            per_agent_concurrency=dict(self._agent_concurrency),
        )
