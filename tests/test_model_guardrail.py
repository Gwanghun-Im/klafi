"""Model Gateway(F09) + Guardrail(F12) 검증.

Model: Alias, Token/Cost span 기록, Timeout/Retry, Fallback.
Guardrail: Input/Output 차단(fail-close), Policy Violation 로깅, Guardrail은 재시도 안 됨.
"""

import logging

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from typing import TypedDict

from klafi import (
    AgentSpec,
    BlocklistGuardrail,
    ExecutionPolicy,
    GuardrailHook,
    ModelGateway,
    SimpleAgent,
    setup_tracing,
)
from klafi.core.exceptions import GuardrailException, ModelException
from klafi.model import FunctionProvider, ModelResult


@pytest.fixture(scope="session")
def exporter():
    exp = InMemorySpanExporter()
    setup_tracing(exporter=exp, simple=True)
    return exp


@pytest.fixture(autouse=True)
def _clear(exporter):
    exporter.clear()
    yield


def _spans(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


# ── Model Gateway ───────────────────────────────────────────────────────
def test_alias_hides_real_model_and_records_tokens(exporter):
    gw = ModelGateway()
    gw.register("quality-high", FunctionProvider(lambda p: "hello world"))
    model = gw.model("quality-high")  # Agent 코드엔 alias만

    agent = SimpleAgent(model=model)
    out = agent.invoke({"question": "hi there"})
    assert out["answer"] == "hello world"

    sp = _spans(exporter)["model.quality-high"]
    assert sp.attributes["klafi.model"] == "quality-high"
    assert sp.attributes["klafi.prompt_tokens"] == 2  # "hi there"
    assert sp.attributes["klafi.completion_tokens"] == 2  # "hello world"
    assert sp.attributes["klafi.tokens"] == 4


def test_cost_recorded(exporter):
    gw = ModelGateway()
    # prompt $2/1k, completion $6/1k
    gw.register("gpt", FunctionProvider(lambda p: "a b c"), cost=(2.0, 6.0))
    gw.model("gpt")("one two")
    sp = _spans(exporter)["model.gpt"]
    # prompt 2tok*2/1k + completion 3tok*6/1k = 0.004 + 0.018
    assert sp.attributes["klafi.cost_usd"] == pytest.approx(0.022)


def test_model_retry_via_policy():
    calls = []

    def flaky(p):
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("model down")
        return "ok"

    gw = ModelGateway()
    gw.register("m", FunctionProvider(flaky), policy=ExecutionPolicy(max_retries=5, backoff_base=0.0))
    assert gw.model("m")("x") == "ok"
    assert len(calls) == 3


def test_model_fallback():
    def boom(p):
        raise ValueError("primary down")

    gw = ModelGateway()
    gw.register("primary", FunctionProvider(boom), fallback="backup")
    gw.register("backup", FunctionProvider(lambda p: "from backup"))
    assert gw.model("primary")("x") == "from backup"


def test_unknown_alias_raises():
    with pytest.raises(ModelException, match="미등록"):
        ModelGateway().model("nope")("x")


# ── Guardrail ───────────────────────────────────────────────────────────
def _guarded_agent(input_guards=None, output_guards=None):
    return SimpleAgent(
        model=lambda p: p.replace("hi", "SECRET"),  # 출력에 SECRET 유발 가능
        spec=AgentSpec(id="g", name="G"),
        hooks=[GuardrailHook(input=input_guards, output=output_guards)],
    )


def test_input_guardrail_blocks(caplog):
    agent = _guarded_agent(input_guards=[BlocklistGuardrail(["나쁜말"])])
    with caplog.at_level(logging.WARNING, logger="klafi.guardrail"):
        with pytest.raises(GuardrailException):
            agent.invoke({"question": "이건 나쁜말 이다"})
    assert any("guardrail.violation" in r.message and "stage=input" in r.message for r in caplog.records)


def test_output_guardrail_blocks():
    agent = _guarded_agent(output_guards=[BlocklistGuardrail(["SECRET"])])
    with pytest.raises(GuardrailException):
        agent.invoke({"question": "hi"})  # 모델이 hi→SECRET 생성 → output 차단


def test_clean_passes():
    agent = _guarded_agent(
        input_guards=[BlocklistGuardrail(["금지"])],
        output_guards=[BlocklistGuardrail(["금지"])],
    )
    out = agent.invoke({"question": "안녕"})
    assert out["answer"] == "안녕"


def test_guardrail_not_retried():
    # GuardrailException은 결정적 → 정책이 있어도 재시도 안 함
    calls = []

    class CountGuard:
        name = "count"

        def check(self, text):
            from klafi.guardrail import GuardrailResult

            calls.append(1)
            return GuardrailResult(False, "always")

    agent = SimpleAgent(
        model=lambda p: p,
        spec=AgentSpec(id="g2", name="G2"),
        hooks=[GuardrailHook(input=[CountGuard()])],
        policy=ExecutionPolicy(max_retries=5, backoff_base=0.0),
    )
    with pytest.raises(GuardrailException):
        agent.invoke({"question": "x"})
    assert len(calls) == 1  # 재시도 없음


def test_pii_guardrail_detects_email():
    from klafi.guardrail import pii_guardrail

    r = pii_guardrail().check("연락처는 a@b.com 입니다")
    assert r.allowed is False and "PII" in r.reason
