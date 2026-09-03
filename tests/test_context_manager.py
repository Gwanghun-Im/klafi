"""Context Manager 검증 (요구사항 §10.3, F05 / CNT-01~08)."""

from klafi import ContextManager


def _msgs(n_user: int) -> list:
    out = [{"role": "system", "content": "너는 도우미다"}]
    for i in range(n_user):
        out.append({"role": "user", "content": f"질문 번호 {i} 내용 채우기 텍스트"})
        out.append({"role": "assistant", "content": f"답변 번호 {i} 응답 텍스트"})
    return out


# ── Token 측정 / Threshold (CNT-01/02) ──────────────────────────────────
def test_count_tokens_and_threshold():
    cm = ContextManager(token_counter=lambda s: len(s.split()), max_tokens=10)  # 주입형 계수기
    msgs = _msgs(2)  # system + 4 메시지
    assert cm.count_tokens(msgs) == sum(len(m["content"].split()) for m in msgs)
    assert cm.over_threshold(msgs) is True
    assert cm.over_threshold([{"role": "user", "content": "짧다"}]) is False


# ── 압축: 중요·최근 보존 + 요약 (CNT-03/04) ─────────────────────────────
def test_reduce_summarizes_old_keeps_system_and_recent():
    cm = ContextManager(max_tokens=5, keep_recent=2, summarizer=lambda p: "요약본")
    msgs = _msgs(5)  # system + 10
    out = cm.reduce(msgs)

    # system 보존
    assert out[0]["role"] == "system" and out[0]["content"] == "너는 도우미다"
    # 요약 메시지 삽입
    assert any(m.get("summary") for m in out)
    # 최근 2건 원문 보존
    assert out[-1]["content"] == msgs[-1]["content"]
    assert out[-2]["content"] == msgs[-2]["content"]
    # 실제로 줄어듦
    assert len(out) < len(msgs)
    assert cm.count_tokens(out) < cm.count_tokens(msgs)


def test_reduce_without_summarizer_drops_old():
    cm = ContextManager(keep_recent=2)  # summarizer 없음
    msgs = _msgs(5)
    out = cm.reduce(msgs)
    assert not any(m.get("summary") for m in out)  # 요약 메시지 없음
    assert out[0]["role"] == "system"  # system 보존
    assert out[-1]["content"] == msgs[-1]["content"]  # 최근 보존
    assert len(out) < len(msgs)  # 오래된 건 제거


def test_important_message_preserved():
    cm = ContextManager(keep_recent=1)
    msgs = [
        {"role": "user", "content": "중요한 결정사항", "important": True},
        {"role": "user", "content": "잡담1"},
        {"role": "user", "content": "잡담2"},
        {"role": "user", "content": "최근"},
    ]
    out = cm.reduce(msgs)
    assert any(m["content"] == "중요한 결정사항" for m in out)  # important 보존
    assert out[-1]["content"] == "최근"


def test_reduce_cuts_only_on_human_turn_keeping_tool_pairs():
    """절단면이 tool_result 위에 떨어지면 tool_use 없는 tool_result 가 선두가 되어 Anthropic 400
    (support_platform report 노드 재현). human 턴까지 되감아 쌍을 보존한다."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    def turn(i):
        return [
            HumanMessage(f"질문{i} 어제 뉴스 찾아줘"),
            AIMessage("", tool_calls=[{"name": "tavily_search", "args": {}, "id": f"call_{i}"}]),
            ToolMessage("검색결과", tool_call_id=f"call_{i}"),
            AIMessage(f"보고서{i}"),
        ]

    out = ContextManager(keep_recent=2).reduce([*turn(1), *turn(2)])
    # 순진한 최근 2건 = [Tool, AI] → 선두가 tool_result. 되감아 2번 턴 전체(4건) 보존, 1번 턴만 제거
    assert [m.type for m in out] == ["human", "ai", "tool", "ai"] and out[0].content == "질문2 어제 뉴스 찾아줘"

    # dict 메시지도 동일 규칙(role=user)
    d = [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""}, {"role": "tool", "content": "r"},
         {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]
    assert [m["content"] for m in ContextManager(keep_recent=1).reduce(d)] == ["q2", "a2"]

    # 되감다 맨 앞까지 오면(human 하나뿐) 줄일 게 없음 → 원본 그대로(no-op)
    single = turn(1)
    assert ContextManager(keep_recent=2).reduce(single) is single
    # keep_recent 가 메시지 수보다 커도 no-op (음수 cut 으로 [Tool] 만 남기던 회귀 방지)
    short = turn(1)[:3]  # [Human, AI(tool_calls), Tool]
    assert ContextManager(keep_recent=4, summarizer=lambda p: "요약").reduce(short) is short


# ── manage: threshold-gated ─────────────────────────────────────────────
def test_manage_only_reduces_over_threshold():
    small = [{"role": "user", "content": "hi"}]
    cm = ContextManager(max_tokens=100)
    assert cm.manage(small) is small  # 안 넘으면 그대로


# ── Handoff Summary (CNT-08) ────────────────────────────────────────────
def test_handoff_summary_with_and_without_summarizer():
    msgs = _msgs(2)
    assert ContextManager(summarizer=lambda p: "핸드오프요약").handoff_summary(msgs) == "핸드오프요약"
    # 요약기 없으면 최근 원문 이어붙임
    plain = ContextManager(keep_recent=1).handoff_summary(msgs)
    assert msgs[-1]["content"] in plain
