"""KlafiCallbackHandler — LangChain chat model 호출을 KLAFI 계측에 연결.

`init_chat_model(alias)`는 LangChain 객체를 그대로 노출하므로 `bind_tools`·`with_structured_output`
같은 파생 Runnable이 자유롭게 만들어진다. 이 핸들러를 **모델 생성 시** 주입하면 어떤 파생을
거치든 실제 LLM 호출마다 다음이 적용된다:

  · span("model.{alias}") + Token/Cost 기록 (OBS-08/11)
  · before_model / after_model Hook (HOK-05) — model-stage Guardrail 포함
  · ModelCalled Event

raise_error=True 이므로 before_model에서 Guardrail이 차단하면 호출 자체가 중단된다(fail-close).
단 이 경로는 **판정 전용**이다 — 콜백은 이미 만들어진 요청/응답의 사이드채널이라 값을 교체할
수 없다. 마스킹 가드레일이 여기 걸리면 무시되고 경고를 남긴다(노드 경계 @klafi_node 에 붙일 것).
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from klafi.core.context import get_context
from klafi.core.hook import _transform, active_hooks
from klafi.observability.tracing import _TRACER_NAME

_log = logging.getLogger("klafi.guardrail")


def _judge(hooks: Any, method: str, value: Any, rebuild: Any, stage: str, *, reverse: bool = False) -> None:
    """콜백 경로의 경계 훅 발화 — 판정(차단)은 하되 치환은 적용 불가라 경고만."""
    out = _transform(hooks, method, value, rebuild, reverse=reverse)  # 차단 시 여기서 raise
    if out is not value:
        _log.warning(
            "guardrail.mask_ignored stage=%s — chat model 콜백 경로에서는 치환이 적용되지 않습니다. "
            "마스킹은 노드 경계(@klafi_node after=[...])에 붙이세요",
            stage,
        )


def _messages_to_text(messages: Any) -> str:
    """콜백이 주는 메시지 배치에서 **마지막 메시지**만 평문으로.

    앞의 메시지는 그것이 마지막이었을 때 이미 검사됐다. 매 호출 이력 전체를 다시 스캔하면 턴마다
    비용이 N배로 늘고 system 프롬프트가 매번 오탐 대상이 된다.
    """
    last: Any = None
    for batch in messages or []:
        items = batch if isinstance(batch, list) else [batch]
        if items:
            last = items[-1]
    if last is None:
        return ""
    content = getattr(last, "content", last)
    return content if isinstance(content, str) else str(content)


class KlafiCallbackHandler(BaseCallbackHandler):
    raise_error = True  # Guardrail 차단이 실제로 호출을 막도록 예외를 전파

    def __init__(self, alias: str, cost: tuple[float, float] | None = None) -> None:
        self._alias = alias
        self._cost = cost
        self._runs: dict[UUID, dict[str, Any]] = {}

    # ── 호출 시작 ──────────────────────────────────────────────────────
    def on_chat_model_start(self, serialized: Any, messages: Any, *, run_id: UUID, **kw: Any) -> None:
        self._start(run_id, _messages_to_text(messages))

    def on_llm_start(self, serialized: Any, prompts: Any, *, run_id: UUID, **kw: Any) -> None:
        self._start(run_id, "\n".join(prompts or []))

    def _start(self, run_id: UUID, prompt: str) -> None:
        ctx = get_context()
        hooks = active_hooks()
        # Guardrail(fail-close)이 여기서 raise하면 LLM 호출 자체가 중단된다.
        _judge(hooks, "before_model", prompt, lambda p: (self._alias, p, ctx), "model")

        # 리프 span 이라 current context 에 attach 하지 않는다 — async 경로에서 langchain 이 sync 콜백을
        # 별도 copy_context 로 실행하므로 attach/detach 토큰이 다른 컨텍스트에서 풀려 오류가 났다.
        # 부모는 start_span 이 현재 컨텍스트(노드 span)에서 자동으로 잡는다.
        sp = trace.get_tracer(_TRACER_NAME).start_span(f"model.{self._alias}")
        sp.set_attribute("klafi.model", self._alias)
        self._runs[run_id] = {"span": sp, "prompt": prompt, "hooks": hooks, "ctx": ctx}

    # ── 호출 종료 ──────────────────────────────────────────────────────
    def on_llm_end(self, response: Any, *, run_id: UUID, **kw: Any) -> None:
        run = self._runs.pop(run_id, None)
        if run is None:
            return
        text, usage = self._extract(response)
        sp = run["span"]
        if usage:
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)
            sp.set_attribute("klafi.prompt_tokens", prompt_tokens)
            sp.set_attribute("klafi.completion_tokens", completion_tokens)
            sp.set_attribute("klafi.tokens", usage.get("total_tokens", prompt_tokens + completion_tokens))
            if self._cost:
                usd = (prompt_tokens / 1000) * self._cost[0] + (completion_tokens / 1000) * self._cost[1]
                sp.set_attribute("klafi.cost_usd", round(usd, 6))
        try:
            _judge(run["hooks"], "after_model", text, lambda t: (self._alias, run["prompt"], t, run["ctx"]),
                   "model_output", reverse=True)
            from klafi.events import EventType, emit

            emit(EventType.ModelCalled, model=self._alias, tokens=(usage or {}).get("total_tokens", 0))
        except BaseException as exc:
            sp.record_exception(exc)
            sp.set_status(Status(StatusCode.ERROR))
            raise
        finally:
            sp.end()

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kw: Any) -> None:
        run = self._runs.pop(run_id, None)
        if run is None:
            return
        run["span"].record_exception(error)
        run["span"].set_status(Status(StatusCode.ERROR))
        run["span"].end()

    @staticmethod
    def _extract(response: Any) -> tuple[str, dict[str, Any] | None]:
        try:
            msg = response.generations[0][0].message
        except Exception:  # noqa: BLE001 — 표준 형태가 아니면 토큰 없이 진행
            return str(response), None
        content = getattr(msg, "content", "")
        text = content if isinstance(content, str) else str(content)
        if not text.strip():
            # tool-calling 방식 structured output(예: with_structured_output)은 content가 비고
            # 실제 데이터는 tool_calls에 실린다 — 가드레일이 빈 문자열을 보지 않도록 폴백한다.
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                text = str(tool_calls)
        return text, getattr(msg, "usage_metadata", None)
