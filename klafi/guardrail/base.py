"""Guardrail Framework (요구사항 §18, F12).

Guardrail은 단순 금칙어를 넘어 Input/Output/PII/Injection 등을 다룬다.
GuardrailHook은 fail_close(=fail_open=False) Hook이라, 위반 시 실행을 중단한다.
위반은 audit성 로그로 남긴다 (GRD-07 Policy Violation Logging).

- before_agent → Input Guardrail (GRD-01)
- after_agent  → Output Guardrail (GRD-02)
Model/Tool 경계 검사(GRD-05/06)는 Model Gateway/Tool에서 같은 Guardrail을 재사용한다.
model 응답(after_model)은 provider에 따라 tool-calling 방식 structured output이 content가 아닌
tool_calls에 실리므로, KlafiCallbackHandler._extract가 content 비어있을 때 tool_calls로 폴백한다.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from klafi.core.context import ExecutionContext
from klafi.core.exceptions import GuardrailViolationError
from klafi.core.hook import Hook

_violation_log = logging.getLogger("klafi.guardrail")


BLOCK = "block"  # 위반 시 실행 중단 (GuardrailViolationError)
WARN = "warn"  # 위반을 기록만 하고 실행은 계속
MASK = "mask"  # 값을 치환하고 계속 (replacement 를 준 경우 자동으로 이 등급)

_NOMASK = object()  # "치환값 없음" 센티널 — None 자체를 치환값으로 줄 수 있어야 한다


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str | None = None
    # 위반의 처리 등급. WARN이면 차단하지 않고 경고 로그만 남긴다.
    # 오타 등 알 수 없는 값은 BLOCK으로 취급한다(fail-close가 안전한 방향).
    severity: str = BLOCK
    # 치환값(마스킹). 주면 **차단하지 않고** 이 값으로 바꿔 계속한다 — 고치는 게 목적이므로.
    # severity 보다 우선한다. 위반 기록은 severity=mask 로 남는다.
    replacement: Any = _NOMASK

    @property
    def masks(self) -> bool:
        return self.replacement is not _NOMASK

    @property
    def blocking(self) -> bool:
        return not self.allowed and not self.masks and self.severity != WARN

    @property
    def effective_severity(self) -> str:
        if self.masks:
            return MASK
        return WARN if self.severity == WARN else BLOCK


class Guardrail(Protocol):
    """가드레일 계약.

    `check`가 받는 값은 `raw` 속성이 정한다 (없으면 False = 텍스트).
      - raw=False(기본): 검사 대상을 텍스트로 변환해 받는다. 이 계약 덕에 하나의 가드레일이
        input/output/model/model_output/tool 전 스테이지에 그대로 꽂힌다 (pii, prompt_injection 등).
      - raw=True: 원본 객체(state dict, tool kwargs, 메시지 객체...)를 그대로 받는다.
        구조·타입을 봐야 하는 검사용. 대신 그 가드레일은 특정 스테이지 전용이 된다.
    """

    name: str

    def check(self, value: Any) -> GuardrailResult: ...


# ── @guardrail 데코레이터 (함수 → Guardrail 객체) ───────────────────────
CheckFn = Callable[[Any], "bool | GuardrailResult"]


class _FnGuardrail:
    def __init__(self, name: str, fn: CheckFn, raw: bool = False) -> None:
        self.name = name
        self.raw = raw
        self._fn = fn

    def check(self, value: Any) -> GuardrailResult:
        out = self._fn(value)
        if isinstance(out, GuardrailResult):
            return out
        return GuardrailResult(bool(out), None if out else f"{self.name} 위반")


def guardrail(fn: CheckFn | str | None = None, *, name: str | None = None, raw: bool = False) -> Any:
    """함수를 Guardrail 객체로 만드는 데코레이터. 코드에서 직접 참조해 쓴다.

        @guardrail
        def no_secrets(text): return "비밀번호" not in text

        @guardrail("no_secrets")          # 이름을 명시할 수도 있다
        def _(text): return "비밀번호" not in text

        @guardrail(raw=True)              # 텍스트 대신 원본 객체를 받는다
        def few_messages(state):
            return len(state.get("messages", [])) <= 50

        @klafi_node("plan", before=[no_secrets, few_messages])
        def plan(state): ...
    """
    if isinstance(fn, str):  # @guardrail("name") 형태
        name, fn = fn, None

    def make(f: CheckFn) -> _FnGuardrail:
        return _FnGuardrail(name or f.__name__, f, raw=raw)

    return make(fn) if fn is not None else make


def enforce(
    guards: list[Guardrail],
    value: Any,
    stage: str,
    ctx: ExecutionContext | None = None,
) -> Any:
    """가드레일 목록을 검사하고 **검사를 통과한 값**을 돌려준다 (GRD-07).

    가드레일은 문자열 정책이다(check(text)->판정/치환text). 값의 문자열 리프에만 정책을
    적용하는 것은 binding 이 담당한다 — 구조·타입·메시지 id를 유지한 채 되돌려 쓴다.
      - 기본: 값의 str 리프마다 check 호출(dict/list/BaseMessage 순회). LLM 경계는 값이
        이미 str이라 리프=값 전체.
      - raw=True: 순회 없이 원본을 그대로 넘긴다(구조 전체를 판정·치환해야 할 때).

    위반은 모두 audit 로그로 남기고, 처리는 등급에 따라 갈린다.
      - BLOCK(기본): GuardrailViolationError로 실행 중단 (fail-close)
      - WARN: 중단하지 않고 계속 — 감지 사실만 기록
      - MASK: GuardrailResult(replacement=...) 를 준 경우. 차단하지 않고 그 리프를 치환해 계속.
    """
    from .binding import binding_for, whole

    for g in guards:
        bind = whole if getattr(g, "raw", False) else binding_for(value)
        value = bind(value, lambda leaf, _g=g: _check_leaf(_g, leaf, stage, ctx))
    return value


def _check_leaf(g: Guardrail, leaf: Any, stage: str, ctx: ExecutionContext | None) -> Any:
    """리프 하나를 검사하고 통과/치환된 리프를 돌려준다. 위반은 로깅·등급 처리."""
    r = g.check(leaf)
    if r.allowed and not r.masks:
        return leaf
    _violation_log.warning(
        "guardrail.violation stage=%s guard=%s severity=%s execution_id=%s reason=%s",
        stage, g.name, r.effective_severity, ctx.execution_id if ctx else "-", r.reason,
    )
    if r.blocking:
        raise GuardrailViolationError(r.reason or "guardrail 위반", stage=stage, guard=g.name)
    return r.replacement if r.masks else leaf


class _WarnGuardrail:
    """기존 Guardrail을 경고 등급으로 감싼다 (검사 로직은 그대로 재사용)."""

    def __init__(self, inner: Guardrail, name: str | None = None) -> None:
        self.name = name or getattr(inner, "name", "guardrail")
        self.raw = getattr(inner, "raw", False)  # 원본/텍스트 계약은 감싸도 그대로 유지
        self._inner = inner

    def check(self, value: Any) -> GuardrailResult:
        r = self._inner.check(value)
        # 통과(allowed)나 치환(masks)은 그대로 둔다 — warn_only는 '차단→경고'만 바꿀 뿐,
        # 마스킹까지 무력화하면 안 된다(그러면 치환이 사라져 원본이 통과 = 보안 회귀).
        if r.allowed or r.masks:
            return r
        return GuardrailResult(False, r.reason, severity=WARN)


def warn_only(g: Guardrail, name: str | None = None) -> Guardrail:
    """가드레일을 '차단하지 않고 경고만' 등급으로 바꾼다.

        @klafi_node("agent", after=[warn_only(pii)])   # PII가 있어도 막지 않고 기록만
        def agent(state): ...

    직접 만드는 가드레일은 GuardrailResult(..., severity=WARN)을 반환해도 된다.
    """
    return _WarnGuardrail(g, name)


class BlocklistGuardrail:
    """금칙어 검사 (GRD-01/02 기본)."""

    def __init__(self, words: list[str], name: str = "blocklist") -> None:
        self.name = name
        self._words = [w.lower() for w in words]

    def check(self, text: str) -> GuardrailResult:
        low = text.lower()
        hit = next((w for w in self._words if w in low), None)
        if hit:
            return GuardrailResult(False, f"금칙어 '{hit}' 감지")
        return GuardrailResult(True)


class RegexGuardrail:
    """정규식 위반 검사. PII(GRD-03)/Prompt Injection(GRD-04)의 기본 구현체로 사용."""

    def __init__(self, patterns: list[str], name: str = "regex", reason: str = "패턴 위반") -> None:
        self.name = name
        self._reason = reason
        self._res = [re.compile(p, re.IGNORECASE) for p in patterns]

    def check(self, text: str) -> GuardrailResult:
        for rx in self._res:
            if rx.search(text):
                return GuardrailResult(False, f"{self._reason}: {rx.pattern}")
        return GuardrailResult(True)


# PII 예: 주민번호/이메일/카드번호 등. 프로젝트에서 확장한다.
def pii_guardrail(name: str = "pii") -> RegexGuardrail:
    return RegexGuardrail(
        patterns=[r"\d{6}-\d{7}", r"\d{16}", r"[\w.]+@[\w.]+\.\w+"],
        name=name,
        reason="PII 감지",
    )


class GuardrailHook(Hook):
    """공통 훅 — 데코레이터를 붙일 수 없는 경계(LLM·Tool)까지 가드레일을 실어 나른다.

    관측(로깅·트레이싱)은 훅, 판정·치환은 가드레일(enforce가 실행)이라는 축을 유지한다.
    model/tool 경계는 반환값이 실제 값을 교체한다(gateway/tool 이 _transform 으로 발화).
    agent 경계(before/after_agent)는 그래프 파이프라인(@klafi_graph)이 담당하므로, 여기서
    치환이 나오면 무시되고 mask_ignored 경고를 남긴다.
    """

    priority = 1  # 가장 바깥: Input을 다른 Hook보다 먼저 검사
    fail_open = False  # fail-close: 위반 시 실행 중단

    def __init__(
        self,
        input: list[Guardrail] | None = None,
        output: list[Guardrail] | None = None,
        tool: list[Guardrail] | None = None,
        tool_output: list[Guardrail] | None = None,
        model: list[Guardrail] | None = None,
        model_output: list[Guardrail] | None = None,
    ) -> None:
        self._input = input or []
        self._output = output or []
        self._tool = tool or []  # Tool 인자 경계 (GRD-05)
        self._tool_output = tool_output or []  # Tool 반환 경계
        self._model = model or []  # LLM Prompt 경계 (GRD-04/06)
        self._model_output = model_output or []  # LLM 응답 경계

    # Graph(=Agent) before/after — 그래프 경계는 치환 불가(파이프라인이 담당) → 경고
    def before_agent(self, input: Any, ctx: ExecutionContext | None) -> None:
        self._observe(self._input, input, "input", ctx)

    def after_agent(self, input: Any, result: Any, ctx: ExecutionContext | None) -> None:
        self._observe(self._output, result, "output", ctx)

    def _observe(self, guards: list[Guardrail], value: Any, stage: str, ctx: ExecutionContext | None) -> None:
        if not guards:
            return
        if enforce(guards, value, stage, ctx) is not value:
            _violation_log.warning(
                "guardrail.mask_ignored stage=%s — 공통 훅의 이 경계에서는 치환이 적용되지 않습니다. "
                "마스킹은 @klafi_node/@klafi_graph 에 붙이세요",
                stage,
            )

    # Model(LLM) before/after — 반환값이 프롬프트/응답을 교체한다
    def before_model(self, model: str, prompt: str, ctx: ExecutionContext | None) -> Any:
        return enforce(self._model, prompt, "model", ctx) if self._model else None

    def after_model(self, model: str, prompt: str, result: Any, ctx: ExecutionContext | None) -> Any:
        return enforce(self._model_output, result, "model_output", ctx) if self._model_output else None

    # Tool before/after — 반환값이 kwargs/결과를 교체한다
    def before_tool(self, tool: str, kwargs: dict, ctx: ExecutionContext | None) -> Any:
        return enforce(self._tool, kwargs, "tool", ctx) if self._tool else None

    def after_tool(self, tool: str, kwargs: dict, result: Any, ctx: ExecutionContext | None) -> Any:
        return enforce(self._tool_output, result, "tool_output", ctx) if self._tool_output else None


# ── klafi prebuilt 가드레일 (코드에서 직접 import) ───────────────────────
_PII_RES = [re.compile(p) for p in (r"\d{6}-\d{7}", r"\d{16}", r"[\w.]+@[\w.]+\.\w+")]
_INJECTION_RES = [
    re.compile(p, re.IGNORECASE)
    for p in (r"ignore (the )?(previous|above)", r"system prompt", r"너의 지시(사항)?를 무시")
]


@guardrail(name="pii")
def pii(text: str) -> GuardrailResult:
    hit = next((rx.pattern for rx in _PII_RES if rx.search(text)), None)
    return GuardrailResult(hit is None, f"PII 감지: {hit}" if hit else None)


@guardrail(name="prompt_injection")
def prompt_injection(text: str) -> GuardrailResult:
    hit = next((rx.pattern for rx in _INJECTION_RES if rx.search(text)), None)
    return GuardrailResult(hit is None, f"Prompt Injection 의심: {hit}" if hit else None)
