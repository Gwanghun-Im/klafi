"""Model Gateway (요구사항 §14.1, F09).

Agent가 특정 Model SDK에 강하게 종속되지 않도록 Alias 기반으로 모델을 주입한다.
- Model Alias(MOD-02) / Config 외부화(MOD-03): 코드에는 실제 모델명 대신 alias만 노출.
- Timeout/Retry(MOD-04/05): ExecutionPolicy 재사용.
- Token Usage(MOD-06) / Cost(MOD-07): 호출마다 model span 속성으로 기록 → Observability와 연결.
- Fallback(MOD-08): alias 실패 시 대체 alias 순차 시도.

gateway.model("quality-high")는 (prompt)->str 콜러블을 돌려주므로
SimpleAgent(model=gateway.model("quality-high"))처럼 Template에 바로 꽂힌다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from klafi.core.exceptions import ModelException, ModelNotConfiguredError, ModelNotFoundError
from klafi.observability.tracing import span


@dataclass
class ModelResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ModelProvider(Protocol):
    """실제 모델 어댑터. ChatOpenAI 등을 감싼 구현을 등록한다 (MOD-01).

    두 계층을 갖는다:
    - __call__(prompt)->ModelResult : (prompt)->str 콜러블 경로 (Template·gateway.model).
    - chat_model(callbacks)          : bind_tools/structured output 가능한 LangChain chat model.
      init_chat_model(alias) 표준 경로가 이걸 쓴다. 지원하지 않으면 None 을 반환한다.
    """

    def __call__(self, prompt: str) -> ModelResult: ...

    def chat_model(self, callbacks: Any = None) -> Any: ...  # 미지원이면 None


def _naive_tokens(s: str) -> int:
    return len(s.split())


class FunctionProvider:
    """평범한 (prompt)->str 함수를 ModelProvider로 감싼다. 토큰 카운터는 주입 가능."""

    def __init__(self, fn: Callable[[str], str], count_tokens: Callable[[str], int] | None = None) -> None:
        self._fn = fn
        self._count = count_tokens or _naive_tokens

    def __call__(self, prompt: str) -> ModelResult:
        text = self._fn(prompt)
        return ModelResult(text, self._count(prompt), self._count(text))

    def chat_model(self, callbacks: Any = None) -> Any:
        """init_chat_model(alias) 표준 경로도 지원 — 함수를 LangChain chat model 로 감싼다.

        키 없이(echo provider 등) 표준 스타일 에이전트(init_chat_model.bind_tools/bind_skills)를
        그대로 테스트·데모할 수 있게 한다. tool-calling 은 흉내내지 않으므로 마지막 메시지에
        함수를 적용한 텍스트를 AIMessage 로 돌려준다.
        """
        from ._fake import FunctionChatModel

        return FunctionChatModel(fn=self._fn, callbacks=callbacks)


@dataclass
class _Entry:
    provider: ModelProvider
    policy: Any = None  # ExecutionPolicy (MOD-04/05)
    cost: tuple[float, float] | None = None  # (USD/1k prompt, USD/1k completion) (MOD-07)
    fallback: str | None = None  # 실패 시 대체 alias (MOD-08)


class ModelGateway:
    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def register(
        self,
        alias: str,
        provider: ModelProvider,
        *,
        policy: Any = None,
        cost: tuple[float, float] | None = None,
        fallback: str | None = None,
    ) -> None:
        self._entries[alias] = _Entry(provider, policy, cost, fallback)

    def _entry(self, alias: str) -> _Entry:
        try:
            return self._entries[alias]
        except KeyError:
            raise ModelNotFoundError(f"model alias '{alias}' 미등록", model=alias) from None

    def model(self, alias: str) -> Callable[[str], str]:
        """Template/Node에 꽂을 (prompt)->str 콜러블. 호출 시 span+token+cost 기록."""

        def call(prompt: str) -> str:
            return self._invoke(alias, prompt)

        return call

    def chat_model(self, alias: str) -> Any:
        """bind_tools/with_structured_output 가능한 LangChain chat model.

        KLAFI 계측 핸들러를 주입하므로 파생 Runnable(bind_tools·structured output)을 통한
        호출도 span/Token/Cost·Hook·Event 대상이 된다. provider가 미지원이면 None.
        """
        entry = self._entry(alias)
        fn = getattr(entry.provider, "chat_model", None)
        if fn is None:
            return None
        from .callback import KlafiCallbackHandler

        return fn(callbacks=[KlafiCallbackHandler(alias, entry.cost)])

    def _invoke(self, alias: str, prompt: str) -> str:
        from klafi.core.context import get_context
        from klafi.core.hook import _transform, active_hooks

        entry = self._entry(alias)
        hooks = active_hooks()
        ctx = get_context()
        with span(f"model.{alias}") as sp:
            sp.set_attribute("klafi.model", alias)
            # HOK-05: 프롬프트 경계 가드레일 — 반환값이 프롬프트를 교체(마스킹)한다.
            prompt = _transform(hooks, "before_model", prompt, lambda p: (alias, p, ctx))
            try:
                result = self._apply_policy(lambda: entry.provider(prompt), entry.policy)
            except Exception:
                if entry.fallback:  # MOD-08: 대체 모델로 폴백
                    sp.set_attribute("klafi.model_fallback", entry.fallback)
                    return self._invoke(entry.fallback, prompt)
                raise
            sp.set_attribute("klafi.prompt_tokens", result.prompt_tokens)
            sp.set_attribute("klafi.completion_tokens", result.completion_tokens)
            sp.set_attribute("klafi.tokens", result.total_tokens)  # OBS-08
            if entry.cost:
                usd = (result.prompt_tokens / 1000) * entry.cost[0] + (result.completion_tokens / 1000) * entry.cost[1]
                sp.set_attribute("klafi.cost_usd", round(usd, 6))  # OBS-11
            # 응답 경계 가드레일 — 반환값이 응답을 교체한다(어니언: 역순).
            text = _transform(hooks, "after_model", result.text, lambda t: (alias, prompt, t, ctx), reverse=True)
            from klafi.events import EventType, emit  # lazy

            emit(EventType.ModelCalled, model=alias, tokens=result.total_tokens)
            return text

    @staticmethod
    def _apply_policy(fn: Callable[[], ModelResult], policy: Any) -> ModelResult:
        if policy is None:
            return fn()
        from klafi.runtime.engine import run_sync  # lazy

        return run_sync(fn, policy, lambda _s: None)  # model-level은 실행상태 전이 없음


# ── 모델 선언 표준 (LangChain init_chat_model 스타일) ────────────────────
# 활성 Gateway 는 ContextVar 다 — 프로세스 전역이 아니다.
# init_chat_model 은 define() 안에서만 호출되고 define()은 factory.create() 실행 중에만 돌므로,
# "지금 조립 중인 factory 의 gateway"만 바인딩하면 된다. 전역이면 한 프로세스에 앱/팩토리가
# 둘일 때 마지막이 조용히 이겨(멀티테넌트·테스트 격리에서 오배선), ContextVar 는 그 충돌을 없앤다.
from contextlib import contextmanager
from contextvars import ContextVar

_active_gateway: ContextVar["ModelGateway | None"] = ContextVar("klafi_active_gateway", default=None)


def set_active_gateway(gateway: "ModelGateway | None") -> None:
    """활성 Gateway 를 현재 컨텍스트에 지정 (하위호환용 — 리셋 토큰을 돌려주지 않는다).

    새 코드는 `using_gateway(gateway)` 컨텍스트 매니저를 쓰는 게 안전하다.
    """
    _active_gateway.set(gateway)


@contextmanager
def using_gateway(gateway: "ModelGateway | None") -> Any:
    """with 블록 동안만 활성 Gateway 를 바인딩 (factory.create 조립 구간용)."""
    token = _active_gateway.set(gateway)
    try:
        yield gateway
    finally:
        _active_gateway.reset(token)


class ChatModel:
    """LangChain chat model에 bind_skills를 더한 얇은 래퍼.

    나머지 속성(invoke·bind_tools·with_structured_output·stream ...)은 전부 원본에 위임하므로
    LangChain 모델과 똑같이 쓰면 된다.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> Any:
        """툴 바인딩 — KLAFI Tool은 LangChain tool로 자동 변환한다.

            llm = init_chat_model("main").bind_tools([lookup_order])
        """
        from klafi.tool.skill import Skill
        from klafi.tool.tool import to_langchain_tools

        if any(isinstance(t, Skill) for t in tools):  # 지침이 조용히 버려지는 것을 막는다
            raise ModelException("Skill은 bind_skills()로 바인딩하세요 (bind_tools는 툴 전용)")
        return self._model.bind_tools(to_langchain_tools(tools), **kwargs)

    def bind_skills(self, skills: list[Any]) -> Any:
        """Skill(툴 + 지침)을 바인딩 — 툴은 bind_tools로, prompt는 SystemMessage로.

            llm = init_chat_model("main").bind_skills([clock_kst])
        """
        from klafi.tool.skill import bind_skills

        return bind_skills(self, skills)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __or__(self, other: Any) -> Any:
        return self._model | other

    def __ror__(self, other: Any) -> Any:
        return other | self._model


def init_chat_model(alias: str) -> ChatModel:
    """모델 선언 표준 — config/model.yaml의 alias로 chat model을 얻는다.

        class MyAgent(KlafiGraph):
            def define(self):
                llm = init_chat_model("main").bind_tools(...)    # 툴
                llm = init_chat_model("main").bind_skills([...])  # 툴 + 지침

    업무코드에는 alias만 노출된다(실제 provider/모델명은 공통개발자의 config).
    Agent 조립(define) 시점에 호출한다 — import 시점에는 Gateway가 아직 없다.
    """
    active = _active_gateway.get()
    if active is None:
        raise ModelNotConfiguredError(
            f"ModelGateway 미구성 — alias '{alias}'를 해석할 수 없습니다. "
            "KlafiApp/ExecutionFactory로 Agent를 생성하고 define() 안에서 호출하세요",
            model=alias,
        )
    model = active.chat_model(alias)
    if model is None:
        raise ModelNotConfiguredError(f"alias '{alias}' provider는 chat model을 지원하지 않습니다", model=alias)
    return ChatModel(model)
