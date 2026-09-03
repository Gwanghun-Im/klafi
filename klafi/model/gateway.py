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

import logging

from klafi.core.exceptions import ModelException, ModelNotConfiguredError, ModelNotFoundError
from klafi.observability.tracing import span

_gw_log = logging.getLogger("klafi.model")


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

    def chat_model(self, callbacks: Any = None, **_: Any) -> Any:
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

    def has(self, alias: str) -> bool:
        """alias 등록 여부 — 조립 시점 fail-fast 용 (model() 은 lazy 라 첫 호출까지 오타를 숨긴다)."""
        return alias in self._entries

    def model(self, alias: str) -> Callable[[str], str]:
        """Template/Node에 꽂을 (prompt)->str 콜러블. 호출 시 span+token+cost 기록."""

        def call(prompt: str) -> str:
            return self._invoke(alias, prompt)

        return call

    @staticmethod
    def _chat_overrides(entry: _Entry) -> dict[str, Any]:
        """alias policy → LangChain chat model 생성 인자. model.yaml 의 policy 가 init_chat_model 경로에서도
        효력을 갖게 한다(이전엔 (prompt)->str 경로에만 적용되어 무음으로 무시됐다)."""
        pol = entry.policy
        if pol is None:
            return {}
        out: dict[str, Any] = {"max_retries": pol.max_retries}
        if pol.timeout is not None:
            out["timeout"] = pol.timeout
        return out

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

        callbacks = [KlafiCallbackHandler(alias, entry.cost)]
        overrides = self._chat_overrides(entry)
        try:
            return fn(callbacks=callbacks, **overrides)
        except TypeError:
            if not overrides:
                raise
            _gw_log.warning("provider '%s' 의 chat_model 이 policy 인자(%s)를 받지 않아 정책 없이 생성", alias, sorted(overrides))
            return fn(callbacks=callbacks)

    def chat_fallbacks(self, alias: str) -> list[Any]:
        """alias 의 fallback 체인(raw chat model 목록). 순환(a↔b)은 끊는다 — init_chat_model 경로 MOD-08."""
        out: list[Any] = []
        seen = {alias}
        cur = self._entry(alias).fallback
        while cur and cur not in seen:
            seen.add(cur)
            m = self.chat_model(cur)
            if m is not None:
                out.append(m)
            cur = self._entry(cur).fallback
        return out

    def _invoke(self, alias: str, prompt: str, seen: frozenset = frozenset()) -> str:
        from klafi.core.context import get_context
        from klafi.core.exceptions import GuardrailException, PolicyException
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
            except (GuardrailException, PolicyException):
                raise  # 정책·가드레일 차단은 폴백 대상이 아니다(우회 금지)
            except Exception:
                nxt = entry.fallback
                if nxt and nxt not in seen and nxt != alias:  # MOD-08: 대체 모델로 폴백 — 순환(a↔b)은 원 예외
                    sp.set_attribute("klafi.model_fallback", nxt)
                    return self._invoke(nxt, prompt, seen | {alias})
                raise
            sp.set_attribute("klafi.prompt_tokens", result.prompt_tokens)
            sp.set_attribute("klafi.completion_tokens", result.completion_tokens)
            sp.set_attribute("klafi.tokens", result.total_tokens)  # OBS-08
            usd: float | None = None
            if entry.cost:
                usd = round((result.prompt_tokens / 1000) * entry.cost[0] + (result.completion_tokens / 1000) * entry.cost[1], 6)
                sp.set_attribute("klafi.cost_usd", usd)  # OBS-11
            # 응답 경계 가드레일 — 반환값이 응답을 교체한다(어니언: 역순).
            text = _transform(hooks, "after_model", result.text, lambda t: (alias, prompt, t, ctx), reverse=True)
            from klafi.events import EventType, emit  # lazy

            emit(EventType.ModelCalled, model=alias, tokens=result.total_tokens, cost_usd=usd)
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
    """LangChain chat model 래퍼 — bind_tools / bind_skills 를 **누적 체이닝**한다.

        init_chat_model("main").bind_skills([clock_kst]).bind_tools(search)   # 툴 합집합 + 스킬 지침
        init_chat_model("main").bind_skills([clock_kst, *search])             # 한 번에 — 동일 결과

    LangChain 원본의 bind_tools 는 덮어쓰기지만 KLAFI 에선 **누적**이다: 체이닝 순서대로 툴은
    합쳐지고 스킬 지침(prompt)은 이어 붙어 SystemMessage 로 선행 주입된다. 각 bind 는 새 ChatModel 을
    돌려주며(불변) 즉시 materialize 한다. 그 외 속성(invoke·stream·with_structured_output ...)은
    바인딩된 runnable 에 위임하므로 LangChain 모델과 똑같이 쓰면 된다.
    """

    # BaseChatModel/Runnable 파생 메서드 — 프롬프트 주입(RunnableSequence) '안쪽' 모델에 적용해야 한다.
    # 시퀀스에 위임하면 with_structured_output 은 없고 bind(stop=) 은 앞단 람다로 가서 TypeError 였다.
    _DERIVE = frozenset(
        {"with_structured_output", "with_retry", "with_fallbacks", "with_config", "bind", "with_listeners", "with_types"}
    )

    def __init__(
        self,
        model: Any,
        *,
        items: list[Any] | None = None,
        kwargs: dict | None = None,
        fallbacks: list[Any] | None = None,
        derive: list[tuple] | None = None,
    ) -> None:
        self._model = model  # 원본 LangChain chat model
        self._items = list(items or [])  # 누적된 Tool | Skill | LangChain tool (바인딩 순서 유지)
        self._kw = dict(kwargs or {})  # bind_tools 추가 인자(tool_choice 등) 누적
        self._fallbacks = list(fallbacks or [])  # alias fallback 체인의 raw chat model (MOD-08)
        self._derive = list(derive or [])  # [(메서드, args, kwargs)] 적용 순서
        self._bound = self._materialize()

    def _build(self, base: Any) -> Any:
        from klafi.tool.skill import Skill, _with_system
        from klafi.tool.tool import to_langchain_tools

        tools, prompts = Skill.flatten(self._items) if self._items else ([], [])
        m = base.bind_tools(to_langchain_tools(tools), **self._kw) if tools else base
        for name, a, k in self._derive:
            m = getattr(m, name)(*a, **k)
        return _with_system(m, "\n\n".join(prompts)) if prompts else m

    def _materialize(self) -> Any:
        primary = self._build(self._model)
        if self._fallbacks:
            # 폴백 모델에도 같은 툴·지침·파생을 입힌다. 가드레일 차단은 각 모델의 콜백에서 똑같이 걸리므로 우회되지 않는다.
            primary = primary.with_fallbacks([self._build(f) for f in self._fallbacks])
        return primary

    def _clone(self, **changes: Any) -> "ChatModel":
        base: dict[str, Any] = dict(
            items=self._items, kwargs=self._kw, fallbacks=self._fallbacks, derive=self._derive
        )
        base.update(changes)
        return ChatModel(self._model, **base)

    def _extend(self, items: list[Any], **kwargs: Any) -> "ChatModel":
        return self._clone(items=[*self._items, *items], kwargs={**self._kw, **kwargs})

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ChatModel":
        """툴 바인딩(누적) — KLAFI Tool 은 LangChain tool 로 자동 변환한다.

            llm = init_chat_model("main").bind_tools([lookup_order])
        """
        from klafi.tool.skill import Skill

        if any(isinstance(t, Skill) for t in tools):  # 지침이 조용히 버려지는 것을 막는다
            raise ModelException("Skill은 bind_skills()로 바인딩하세요 (bind_tools는 툴 전용)")
        return self._extend(tools, **kwargs)

    def bind_skills(self, skills: list[Any]) -> "ChatModel":
        """Skill(툴 + 지침) 바인딩(누적) — Tool 이 섞인 리스트도 받는다.

            llm = init_chat_model("main").bind_skills([clock_kst])
        """
        return self._extend(skills)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):  # 내부 속성 미설정 시 재귀 방지(copy/pickle 등)
            raise AttributeError(name)
        if name in self._DERIVE:  # 파생도 누적·불변 — 새 ChatModel 을 돌려준다
            return lambda *a, **k: self._clone(derive=[*self._derive, (name, a, k)])
        return getattr(self._bound, name)

    def __or__(self, other: Any) -> Any:
        return self._bound | other

    def __ror__(self, other: Any) -> Any:
        return other | self._bound


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
    return ChatModel(model, fallbacks=active.chat_fallbacks(alias))
