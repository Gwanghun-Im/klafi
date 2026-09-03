"""실제 LLM Provider 어댑터 (MOD-01).

KLAFI ModelProvider 계약: (prompt: str) -> ModelResult(text, prompt_tokens, completion_tokens).
API 키는 환경변수에서 읽는다(코드/Config에 저장 금지 — SEC-05). SDK는 lazy import.

    export OPENAI_API_KEY=...      # OpenAIProvider
    export ANTHROPIC_API_KEY=...   # AnthropicProvider
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import SecretStr

from klafi.core.exceptions import ModelException, ModelNotConfiguredError

from .gateway import FunctionProvider, ModelResult


def _secret(value: str | None, env: str) -> SecretStr | None:
    """API 키는 SecretStr 로만 보관 — vars()/repr/로그/피클에 평문이 새지 않게 (SEC-05)."""
    raw = value or os.environ.get(env)
    return SecretStr(raw) if raw else None


class OpenAIProvider:
    """OpenAI Chat Completions 어댑터. kwargs(temperature 등)는 두 경로(SDK·LangChain) 모두에 전달."""

    def __init__(self, model: str = "gpt-4o-mini", *, api_key: str | None = None, **kwargs: Any) -> None:
        self._model = model
        self._key = _secret(api_key, "OPENAI_API_KEY")
        self._kwargs = kwargs
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._key:
                raise ModelNotConfiguredError("OPENAI_API_KEY 환경변수가 없습니다")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ModelException("openai 패키지가 필요합니다: pip install openai") from exc
            self._client = OpenAI(api_key=self._key.get_secret_value())
        return self._client

    def __call__(self, prompt: str) -> ModelResult:
        resp = self._get_client().chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **self._kwargs,
        )
        text = resp.choices[0].message.content or ""
        u = resp.usage
        return ModelResult(text, getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))

    def chat_model(self, callbacks: Any = None, **overrides: Any) -> Any:
        """bind_tools 가능한 LangChain chat model (tool-calling·structured output용).

        callbacks는 Gateway가 주입하는 KLAFI 계측 핸들러 — 파생 Runnable에도 상속된다.
        overrides 는 alias policy(timeout/max_retries)처럼 Gateway 가 얹는 생성 인자.
        """
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=self._model, api_key=self._key, callbacks=callbacks, **{**self._kwargs, **overrides})


class AnthropicProvider:
    """Anthropic Messages 어댑터. kwargs(temperature 등)는 두 경로(SDK·LangChain) 모두에 전달."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        *,
        api_key: str | None = None,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._key = _secret(api_key, "ANTHROPIC_API_KEY")
        self._max_tokens = max_tokens
        self._kwargs = kwargs
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._key:
                raise ModelNotConfiguredError("ANTHROPIC_API_KEY 환경변수가 없습니다")
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise ModelException("anthropic 패키지가 필요합니다: pip install anthropic") from exc
            self._client = Anthropic(api_key=self._key.get_secret_value())
        return self._client

    def __call__(self, prompt: str) -> ModelResult:
        resp = self._get_client().messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **self._kwargs,
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")
        u = resp.usage
        return ModelResult(text, getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0))

    def chat_model(self, callbacks: Any = None, **overrides: Any) -> Any:
        """bind_tools 가능한 LangChain chat model (tool-calling·structured output용).

        callbacks는 Gateway가 주입하는 KLAFI 계측 핸들러 — 파생 Runnable에도 상속된다.
        overrides 는 alias policy(timeout/max_retries)처럼 Gateway 가 얹는 생성 인자.
        """
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=self._model, api_key=self._key, max_tokens=self._max_tokens, callbacks=callbacks,
            **{**self._kwargs, **overrides},
        )


# ── Provider Registry (MOD-01, checkpoint/store 와 동일한 확장 패턴) ─────────
# config(model.yaml)의 provider type → 인스턴스 팩토리. 새 provider(Bedrock/Vertex/
# 사내 게이트웨이)는 KLAFI 소스를 고치지 않고 register_provider 로 붙인다.
from typing import Callable  # noqa: E402

ProviderFactory = Callable[[dict[str, Any]], Any]
_REGISTRY: dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """프로젝트 Custom Provider 등록 (확장성 NFR)."""
    _REGISTRY[name.lower()] = factory


def resolve_provider(spec: dict[str, Any]) -> Any:
    """{type, model, ...} → Provider 인스턴스. 미등록 type 은 fail-fast."""
    name = str(spec.get("type", "")).lower()
    factory = _REGISTRY.get(name)
    if factory is None:
        raise ModelException(
            f"알 수 없는 provider type: {name!r} (가능: {sorted(_REGISTRY)})"
        )
    return factory(spec)


def _openai_factory(spec: dict[str, Any]) -> Any:
    # params: model.yaml 의 모델 생성 인자(temperature, max_tokens ...) — 두 경로 모두에 전달된다
    return OpenAIProvider(spec.get("model") or "gpt-4o-mini", **(spec.get("params") or {}))


def _anthropic_factory(spec: dict[str, Any]) -> Any:
    return AnthropicProvider(spec.get("model") or "claude-haiku-4-5-20251001", **(spec.get("params") or {}))


def _echo_factory(spec: dict[str, Any]) -> Any:  # 키 없이 테스트/데모용
    return FunctionProvider(lambda p: f"[echo] {p.strip()[:80]}")


register_provider("openai", _openai_factory)
register_provider("anthropic", _anthropic_factory)
register_provider("echo", _echo_factory)
