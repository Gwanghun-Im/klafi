"""@klafi_graph(워크플로우 경계) + raw 가드레일 검증.

- @klafi_node : 노드 단위 미들웨어·가드레일
- @klafi_graph: 워크플로우(그래프 전체) 단위. 미들웨어는 값을 교체할 수 있다(훅은 불가).
- 가드레일은 기본이 텍스트 검사기지만 raw=True 로 원본 객체를 받을 수 있다.
"""

import logging
import re
from typing import TypedDict

import pytest
from langgraph.graph import END, START

from klafi import AgentSpec, KlafiGraph, guardrail, klafi_graph, klafi_node
from klafi.core.exceptions import GuardrailException
from klafi.guardrail import GuardrailResult, WARN, enforce


class State(TypedDict):
    q: str
    a: str


def _const_guardrail(result: GuardrailResult, name: str = "fixed", raw: bool = False):
    """항상 같은 결과를 내는 가드레일 (마스킹 등 결과 조합 검증용)."""

    class _G:
        def __init__(self) -> None:
            self.name = name
            self.raw = raw

        def check(self, value):
            return result

    return _G()


# ── raw 가드레일: 텍스트 대신 원본 객체 ─────────────────────────────────
def test_raw_guardrail_receives_object_not_text():
    seen = []

    @guardrail(raw=True)
    def structural(state):
        seen.append(state)
        return isinstance(state, dict) and len(state.get("items", [])) <= 2

    enforce([structural], {"items": [1, 2]}, "input")  # 통과
    assert seen[0] == {"items": [1, 2]}  # 문자열이 아니라 dict 그대로

    with pytest.raises(GuardrailException):
        enforce([structural], {"items": [1, 2, 3]}, "input")


def test_text_guardrail_still_gets_text_from_object():
    """기존 계약 유지 — dict를 넘겨도 텍스트로 변환돼 도착한다."""
    got = []

    @guardrail
    def textual(text):
        got.append(text)
        return "비밀" not in text

    enforce([textual], {"q": "안녕"}, "input")
    assert isinstance(got[0], str) and "안녕" in got[0]


def test_text_guardrail_visits_only_string_leaves():
    """텍스트 가드레일은 값의 str 리프마다 실행되고, 비-str 스칼라(int)는 건너뛴다."""
    seen = []

    @guardrail
    def collect(text):
        seen.append(text)
        return True

    enforce([collect], {"a": "안녕", "n": 42, "nested": {"b": "잘가"}}, "input")
    assert set(seen) == {"안녕", "잘가"}  # 42는 리프 아님, 문자열만
    assert all(isinstance(x, str) for x in seen)


# ── 마스킹: 차단 대신 치환 ───────────────────────────────────────────────
def test_mask_replaces_value_instead_of_blocking():
    @guardrail
    def mask_email(text):
        if "@" not in text:
            return True
        return GuardrailResult(False, "PII 마스킹", replacement=re.sub(r"\S+@\S+", "***", text))

    out = enforce([mask_email], "연락처 a@b.com 입니다", "output")
    assert out == "연락처 *** 입니다"  # 예외 없이 치환된 값이 나온다


def test_mask_wins_over_block_severity():
    """치환값을 주면 severity가 BLOCK이어도 차단하지 않는다 — 고치는 게 목적이므로."""
    g = _const_guardrail(GuardrailResult(False, "위반", severity="block", replacement="fixed"))
    assert enforce([g], "원본", "output") == "fixed"


def test_mask_is_logged_as_mask_severity(caplog):
    g = _const_guardrail(GuardrailResult(False, "PII", replacement="***"))
    with caplog.at_level(logging.WARNING, logger="klafi.guardrail"):
        enforce([g], "a@b.com", "output")
    assert "severity=mask" in caplog.text  # 운영에서 마스킹 건수를 집계할 수 있어야 한다


def test_later_guardrail_sees_masked_value():
    """치환 후 뒤 가드레일은 치환본을 본다(텍스트 캐시가 무효화되어야 한다)."""
    seen = []

    @guardrail
    def masker(text):
        return GuardrailResult(False, "치환", replacement="깨끗함")

    @guardrail
    def observer(text):
        seen.append(text)
        return True

    enforce([masker, observer], "더러움", "output")
    assert seen == ["깨끗함"]


def test_none_can_be_a_replacement():
    """센티널을 쓰므로 None 자체도 치환값으로 줄 수 있다."""
    g = _const_guardrail(GuardrailResult(False, "제거", replacement=None))
    assert enforce([g], "원본", "output") is None


def test_mask_in_node_pipeline_replaces_result():
    @guardrail(raw=True)  # 노드 결과(dict)를 구조 그대로 받아 필드만 치환
    def mask_a(state):
        return GuardrailResult(False, "마스킹", replacement={**state, "a": "***"})

    class A(KlafiGraph):
        spec = AgentSpec(id="m1", name="M1")
        state_schema = State
        observability = False

        def define(self):
            @klafi_node("n", after=[mask_a])
            def node(state):
                return {"a": "비밀"}

            self.add_node("n", node)
            self.add_edge(START, "n")
            self.add_edge("n", END)

    assert A().invoke({"q": "x", "a": ""})["a"] == "***"


def test_common_hook_warns_that_mask_is_ignored(caplog):
    """훅은 값을 교체할 수 없다 — 마스킹이 조용히 사라지지 않도록 경고해야 한다."""
    from klafi.guardrail import GuardrailHook

    g = _const_guardrail(GuardrailResult(False, "PII", replacement={"a": "***"}), raw=True)
    hook = GuardrailHook(output=[g])
    with caplog.at_level(logging.WARNING, logger="klafi.guardrail"):
        hook.after_agent({"q": "x"}, {"a": "a@b.com"}, None)
    assert "mask_ignored" in caplog.text


def test_warn_only_preserves_raw_contract():
    from klafi.guardrail import warn_only

    @guardrail(raw=True)
    def structural(state):
        return GuardrailResult(False, "구조 위반", severity=WARN)

    wrapped = warn_only(structural)
    assert wrapped.raw is True
    assert enforce([wrapped], {"x": 1}, "input")  # 차단 없이 경고로 통과


# ── @klafi_graph: 워크플로우 경계 ────────────────────────────────────────
def _simple(cls_id: str):
    def define(self):
        @klafi_node("n")
        def node(state):
            return {"a": f"ok:{state['q']}"}

        self.add_node("n", node)
        self.add_edge(START, "n")
        self.add_edge("n", END)

    return define


def test_graph_input_guardrail_blocks():
    @guardrail
    def clean(text):
        return "금지" not in text

    @klafi_graph(before=[clean])
    class A(KlafiGraph):
        spec = AgentSpec(id="g1", name="G1")
        state_schema = State
        observability = False
        define = _simple("g1")

    assert A().invoke({"q": "정상", "a": ""})["a"] == "ok:정상"
    with pytest.raises(GuardrailException):
        A().invoke({"q": "금지 요청", "a": ""})


def test_graph_before_middleware_replaces_input():
    """훅과 달리 미들웨어는 값을 교체할 수 있다 — 워크플로우 진입 input 보강."""

    def enrich(input, ctx):
        return {**input, "q": input["q"].upper()}

    @klafi_graph(before=[enrich])
    class A(KlafiGraph):
        spec = AgentSpec(id="g2", name="G2")
        state_schema = State
        observability = False
        define = _simple("g2")

    assert A().invoke({"q": "hi", "a": ""})["a"] == "ok:HI"


def test_graph_after_middleware_replaces_result():
    def redact(result, ctx):
        return {**result, "a": "[마스킹]"}

    @klafi_graph(after=[redact])
    class A(KlafiGraph):
        spec = AgentSpec(id="g3", name="G3")
        state_schema = State
        observability = False
        define = _simple("g3")

    assert A().invoke({"q": "x", "a": ""})["a"] == "[마스킹]"


def test_graph_middleware_failure_is_fail_close():
    def require_login(input, ctx):
        raise PermissionError("로그인 필요")

    @klafi_graph(before=[require_login])
    class A(KlafiGraph):
        spec = AgentSpec(id="g4", name="G4")
        state_schema = State
        observability = False
        define = _simple("g4")

    with pytest.raises(PermissionError):
        A().invoke({"q": "x", "a": ""})


def test_graph_output_guardrail_sees_middleware_result():
    """순서 검증: after 미들웨어 → output 가드. 미들웨어가 가드를 우회할 수 없다."""

    @guardrail
    def no_secret(text):
        return "비밀" not in text

    def leak(result, ctx):
        return {**result, "a": "비밀"}

    @klafi_graph(after=[leak, no_secret])
    class A(KlafiGraph):
        spec = AgentSpec(id="g5", name="G5")
        state_schema = State
        observability = False
        define = _simple("g5")

    with pytest.raises(GuardrailException):
        A().invoke({"q": "x", "a": ""})


def test_graph_on_error_middleware_observes():
    seen = []

    def watch(exc, input, ctx):
        seen.append(type(exc).__name__)

    @klafi_graph(on_error=[watch])
    class A(KlafiGraph):
        spec = AgentSpec(id="g6", name="G6")
        state_schema = State
        observability = False

        def define(self):
            @klafi_node("boom")
            def boom(state):
                raise ValueError("터짐")

            self.add_node("n", boom)
            self.add_edge(START, "n")
            self.add_edge("n", END)

    with pytest.raises(Exception):
        A().invoke({"q": "x", "a": ""})
    assert seen == ["ValueError"]


def test_klafi_graph_rejects_functions():
    with pytest.raises(Exception):

        @klafi_graph(before=[])
        def not_a_class(state):
            return state


@pytest.mark.asyncio
async def test_graph_pipeline_works_on_ainvoke():
    async def enrich(input, ctx):
        return {**input, "q": "async"}

    @klafi_graph(before=[enrich])
    class A(KlafiGraph):
        spec = AgentSpec(id="g7", name="G7")
        state_schema = State
        observability = False
        define = _simple("g7")

    out = await A().ainvoke({"q": "x", "a": ""})
    assert out["a"] == "ok:async"


def test_text_guardrail_masks_dict_leaf_in_place():
    """텍스트 가드레일이 dict의 str 리프를 치환하면 구조·타입은 유지된 채 그 필드만 바뀐다."""

    @guardrail
    def mask(text):
        return GuardrailResult(False, "치환", replacement="***") if "@" in text else GuardrailResult(True)

    out = enforce([mask], {"q": "a@b.com", "n": 42}, "output")
    assert out == {"q": "***", "n": 42}  # 리프만 치환, int는 그대로


def test_masking_opaque_object_raises():
    """매핑 불가 객체(pydantic 등)는 검사는 되지만 치환은 막는다 — state가 조용히 깨지는 것 방지."""

    class Opaque:
        def __str__(self):
            return "a@b.com"

    @guardrail
    def mask(text):
        return GuardrailResult(False, "치환", replacement="***") if "@" in text else GuardrailResult(True)

    with pytest.raises(GuardrailException, match="치환할 수 없습니다"):
        enforce([mask], {"x": Opaque()}, "output")
