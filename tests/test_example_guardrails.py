"""examples/support_platform 의 mask_phone — 휴대폰뿐 아니라 유선·070·대표번호까지 가린다.

회귀: report 노드가 "전화: 070-4168-2900 (또는 02-523-7029)" 를 그대로 내보냄(정규식이 01x 만 매치).
"""

import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "examples" / "support_platform" / "common" / "guardrails.py"
_spec = importlib.util.spec_from_file_location("example_guardrails", _path)
_g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_g)


def test_mask_phone_covers_landline_internet_and_hotline():
    text = (
        "전화: 070-4168-2900 (또는 02-523-7029), 휴대폰 010-1234-5678, 대표 1588-1234, "
        "지역 031-123-4567, 붙여쓴 01012345678"
    )
    r = _g.mask_phone.check(text)
    assert r.allowed is False and r.reason == "전화번호 마스킹"
    assert r.replacement == (
        "전화: 070-****-**** (또는 02-****-****), 휴대폰 010-****-****, 대표 1588-****, "
        "지역 031-****-****, 붙여쓴 010-****-****"
    )


def test_mask_phone_ignores_plain_numbers():
    # 우편번호·연도·번지 같은 숫자는 건드리지 않는다
    assert _g.mask_phone.check("우편번호 06707, 효령로 176, 2010년 이전").allowed is True
