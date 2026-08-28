"""바인딩 — 값의 str 리프에만 정책을 적용하고 구조·타입·메시지 id를 유지한다."""

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph.message import add_messages

from klafi.core.exceptions import GuardrailException
from klafi.guardrail import enforce, guardrail
from klafi.guardrail.base import GuardrailResult
from klafi.guardrail.binding import bind

UP = lambda s: s.upper()


def test_str_leaf_mapped():
    assert bind("hi", UP) == "HI"


def test_dict_values_mapped_keys_untouched():
    assert bind({"a": "x", "b": "y"}, UP) == {"a": "X", "b": "Y"}


def test_non_str_scalars_are_not_leaves():
    seen = []
    bind({"a": "x", "n": 42, "f": 1.5, "b": True, "z": None}, lambda s: seen.append(s) or s)
    assert seen == ["x"]  # 문자열만 리프


def test_nested_structure_and_type_preserved():
    out = bind({"items": ("a", "b"), "meta": {"k": "v"}}, UP)
    assert out == {"items": ("A", "B"), "meta": {"k": "V"}}
    assert isinstance(out["items"], tuple)  # tuple 유지


def test_messages_only_last_is_mapped():
    seen = []
    state = {"messages": [AIMessage("첫째"), AIMessage("둘째"), AIMessage("셋째")]}
    bind(state, lambda s: seen.append(s) or s)
    assert seen == ["셋째"]  # 마지막만


def test_message_id_preserved_so_reducer_replaces():
    """가장 중요 — 치환 후에도 id가 같아 add_messages가 append 아닌 교체를 한다."""
    orig = AIMessage("내 번호 010-1234-5678", id="m1")
    out = bind({"messages": [orig]}, lambda s: s.replace("010-1234-5678", "***"))
    masked = out["messages"][-1]
    assert masked.id == "m1" and masked.content == "내 번호 ***"
    # add_messages 리듀서: 같은 id면 교체 → 대화가 중복 append되지 않는다
    merged = add_messages([orig], [masked])
    assert len(merged) == 1 and merged[0].content == "내 번호 ***"


def test_message_id_and_type_not_seen_by_guardrail():
    seen = []
    bind({"messages": [AIMessage("본문", id="run-abc")]}, lambda s: seen.append(s) or s)
    assert seen == ["본문"]  # id/type/name은 전달되지 않음


def test_unchanged_returns_same_object():
    state = {"a": "x", "meta": {"k": "v"}}
    assert bind(state, lambda s: s) is state  # 변경 없으면 재조립 안 함


def test_dict_form_message_content_is_a_leaf():
    """그래프 입력 시점의 dict 메시지({'role','content'})는 dict 브랜치라 content가 리프가 된다.

    role 값도 함께 방문되지만(임시 입력 표현), 실제 가드레일(전화/이메일 패턴)엔 무해하다.
    """
    seen = []
    bind({"messages": [{"role": "user", "content": "내 번호 010-1234-5678"}]},
         lambda s: seen.append(s) or s)
    assert "내 번호 010-1234-5678" in seen  # content가 검사 대상에 포함


def test_opaque_object_checked_but_not_masked():
    class Opaque:
        def __str__(self):
            return "secret@x.com"

    assert bind(Opaque(), lambda s: s).__class__.__name__ == "Opaque"  # 검사만 하면 원본 유지
    with pytest.raises(GuardrailException):
        bind(Opaque(), lambda s: "***")  # 치환 시도는 막힌다


# ── enforce 통합: 오탐 회귀 ──────────────────────────────────────────────
def test_metadata_numbers_do_not_trigger_pii():
    """리프 단위 검사라 id·토큰수 같은 메타데이터가 pii 패턴에 걸리지 않는다."""
    from klafi.guardrail import pii

    msg = AIMessage(
        content="주문 확인됐습니다",
        id="run-8f3a2b1c",
        response_metadata={"usage": {"output_tokens": 1234567890123456}},  # 16자리
    )
    # 통짜 투영이었다면 \d{16} 이 토큰수에 걸려 BLOCK. 이제 content만 보므로 통과.
    out = enforce([pii], {"messages": [msg]}, "output")
    assert out["messages"][-1].content == "주문 확인됐습니다"
