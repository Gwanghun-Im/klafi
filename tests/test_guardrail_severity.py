"""가드레일 위반 등급 — BLOCK(차단) vs WARN(경고만).

모든 위반이 실행을 중단해야 하는 건 아니다. PII 감지처럼 '기록은 남기되 통과'시켜야
하는 정책이 있어 GuardrailResult.severity로 조절한다.
"""

import logging

import pytest

from klafi.core.exceptions import GuardrailViolationError
from klafi.guardrail import WARN, GuardrailResult, enforce, guardrail, pii, warn_only


@guardrail(name="soft_pii")
def soft_pii(text: str) -> GuardrailResult:
    """검사 함수가 직접 severity를 지정하는 방식."""
    hit = "@" in text
    return GuardrailResult(not hit, "PII 의심" if hit else None, severity=WARN)


def test_block_is_default():
    with pytest.raises(GuardrailViolationError):
        enforce([pii], "연락처는 a@b.com 입니다", "output")


def test_warn_passes_through_and_is_reported(caplog):
    with caplog.at_level(logging.WARNING, logger="klafi.guardrail"):
        out = enforce([soft_pii], "연락처는 a@b.com 입니다", "output")  # 예외 없음
    assert out == "연락처는 a@b.com 입니다"  # 값은 그대로 통과
    assert "PII 의심" in caplog.text


def test_warn_only_wraps_existing_guardrail():
    """prebuilt 가드레일을 재작성 없이 경고 등급으로."""
    assert enforce([warn_only(pii)], "a@b.com", "output") == "a@b.com"  # 통과
    with pytest.raises(GuardrailViolationError):  # 원본은 그대로 차단
        enforce([pii], "a@b.com", "output")


def test_clean_text_passes_value_through(caplog):
    with caplog.at_level(logging.WARNING, logger="klafi.guardrail"):
        assert enforce([warn_only(pii), pii], "안녕하세요", "output") == "안녕하세요"
    assert "guardrail.violation" not in caplog.text


def test_unknown_severity_blocks():
    """오타 등 알 수 없는 값은 fail-close(차단)로 처리해야 안전하다."""

    @guardrail(name="typo")
    def typo(text: str) -> GuardrailResult:
        return GuardrailResult(False, "오타", severity="warning")  # WARN이 아님

    with pytest.raises(GuardrailViolationError):
        enforce([typo], "x", "output")


def test_warn_is_logged_with_severity(caplog):
    with caplog.at_level(logging.WARNING, logger="klafi.guardrail"):
        enforce([warn_only(pii)], "a@b.com", "output")
    assert "severity=warn" in caplog.text  # 운영에서 등급으로 필터링 가능해야 한다
