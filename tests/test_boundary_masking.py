"""LLM·Tool 경계 치환 — GuardrailHook이 프롬프트·응답·인자·결과를 마스킹한다."""

import logging

from klafi import Tool, guardrail
from klafi.guardrail import GuardrailHook, GuardrailResult
from klafi.model import FunctionProvider, ModelGateway


@guardrail
def mask_at(text):
    return GuardrailResult(False, "마스킹", replacement=text.replace("@", "[at]")) if "@" in text else GuardrailResult(True)


def _bind(hook):
    """GuardrailHook을 활성 훅으로 묶는 컨텍스트."""
    from klafi.core.hook import bind_hooks

    return bind_hooks([hook])


# ── LLM 경계 (ModelGateway 경로) ─────────────────────────────────────────
def test_model_input_masked():
    seen = {}
    gw = ModelGateway()
    gw.register("m", FunctionProvider(lambda p: seen.setdefault("prompt", p) or "ok"))
    with _bind(GuardrailHook(model=[mask_at])):
        gw.model("m")("메일 a@b.com 확인")
    assert seen["prompt"] == "메일 a[at]b.com 확인"  # provider가 마스킹된 프롬프트를 받음


def test_model_output_masked():
    gw = ModelGateway()
    gw.register("m", FunctionProvider(lambda p: "응답 x@y.com"))
    with _bind(GuardrailHook(model_output=[mask_at])):
        out = gw.model("m")("질문")
    assert out == "응답 x[at]y.com"  # 반환값이 마스킹됨


def test_model_input_guardrail_still_blocks():
    import pytest

    from klafi.core.exceptions import GuardrailViolationError

    @guardrail
    def block_secret(text):
        return "비밀" not in text

    gw = ModelGateway()
    gw.register("m", FunctionProvider(lambda p: "ok"))
    with _bind(GuardrailHook(model=[block_secret])):
        with pytest.raises(GuardrailViolationError):
            gw.model("m")("비밀 유출")


# ── Tool 경계 ────────────────────────────────────────────────────────────
def test_tool_input_masked():
    seen = {}

    def echo(msg: str) -> str:
        seen["msg"] = msg
        return "done"

    tool = Tool(echo, name="echo")
    with _bind(GuardrailHook(tool=[mask_at])):
        tool.run(msg="연락 c@d.com")
    assert seen["msg"] == "연락 c[at]d.com"  # 함수 본문이 마스킹된 kwargs를 받음


def test_tool_output_masked():
    tool = Tool(lambda: "결과 e@f.com", name="gen")
    with _bind(GuardrailHook(tool_output=[mask_at])):
        assert tool.run() == "결과 e[at]f.com"


# ── fail_open 훅의 부분 변환 격리 ────────────────────────────────────────
def test_fail_open_hook_exception_keeps_prior_value():
    """fail_open 훅이 예외를 던지면 값은 그 훅 직전 상태를 유지한다(부분 변환 안 샘)."""
    from klafi.core.hook import Hook, _transform

    class Boom(Hook):
        fail_open = True

        def after_model(self, model, prompt, result, ctx):
            raise RuntimeError("계측 실패")

    class Mask(Hook):
        fail_open = True

        def after_model(self, model, prompt, result, ctx):
            return result.replace("@", "[at]")

    # Mask(치환) → Boom(예외). Boom이 삼켜져도 Mask 결과는 살아남는다.
    out = _transform([Mask(), Boom()], "after_model", "a@b", lambda t: ("m", "p", t, None), reverse=True)
    # reverse=True 라 Boom 먼저(예외 삼킴, 값 불변) → Mask 적용
    assert out == "a[at]b"


# ── chat model 콜백 경로: 판정만, 마스킹은 경고 ──────────────────────────
def test_callback_path_masking_warns(caplog):
    from klafi.model.callback import _judge

    with caplog.at_level(logging.WARNING, logger="klafi.guardrail"):
        out = _judge([GuardrailHook(model_output=[mask_at])], "after_model", "a@b.com",
                     lambda t: ("m", "p", t, None), "model_output", reverse=True)
    assert out is None  # 반환값 없음(콜백은 값을 못 바꿈)
    assert "mask_ignored" in caplog.text
