"""ContextHook 검증 — 히스토리 자동 압축 (§10.3 CNT-01~04).

노드가 보는 view는 축소(모델 입력 토큰 절감), Checkpoint 원본은 보존(감사).
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState

from klafi import AgentSpec, ContextHook, KlafiApp, KlafiGraph, klafi_node
from klafi.core.exceptions import ConfigSchemaError

BASE = {
    "framework.yaml": "service: t\ncheckpoint: memory\n",
    "model.yaml": "providers:\n  main:\n    type: echo\n",
}


def _dir(tmp_path, files):
    for name, text in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return str(tmp_path)


def _agent(hooks, seen):
    class Chat(KlafiGraph):
        spec = AgentSpec(id="chat", name="Chat")
        state_schema = MessagesState
        observability = False

        def define(self):
            @klafi_node("n")
            def node(s):
                seen.append(len(s["messages"]))  # 노드가 보는 개수
                return {"messages": [AIMessage("답변")]}

            self.add_node("n", node)
            self.add_edge(START, "n")
            self.add_edge("n", END)

    return Chat(checkpointer="memory", hooks=hooks)


def test_compresses_node_view_but_keeps_checkpoint():
    seen: list[int] = []
    agent = _agent([ContextHook(max_tokens=5, keep_recent=2)], seen)
    for i in range(4):
        agent.invoke({"messages": [HumanMessage(f"질문 번호 {i} 입니다")]}, thread_id="t1")

    # 노드가 보는 히스토리는 제한됨 (threshold 초과 시 압축)
    assert max(seen) <= 4
    # Checkpoint에는 전체가 남는다 (4턴 × 2 = 8건)
    assert len(agent.get_state(thread_id="t1").values["messages"]) == 8


def test_below_threshold_untouched():
    seen: list[int] = []
    agent = _agent([ContextHook(max_tokens=10_000)], seen)
    agent.invoke({"messages": [HumanMessage("짧은 질문")]}, thread_id="t2")
    assert seen == [1]  # 압축 없음


def test_summarizer_used_when_configured():
    seen_prompts: list[str] = []

    def summarizer(prompt: str) -> str:
        seen_prompts.append(prompt)
        return "요약본"

    seen: list[int] = []
    agent = _agent([ContextHook(max_tokens=5, keep_recent=1, summarizer=summarizer)], seen)
    for i in range(3):
        agent.invoke({"messages": [HumanMessage(f"질문 번호 {i} 내용")]}, thread_id="t3")
    assert seen_prompts  # 요약기가 호출됨


def test_no_messages_state_is_noop():
    from typing import TypedDict

    class S(TypedDict):
        q: str

    class A(KlafiGraph):
        spec = AgentSpec(id="a", name="A")
        state_schema = S
        observability = False

        def define(self):
            @klafi_node("n")
            def n(s):
                return {"q": s["q"]}

            self.add_node("n", n)
            self.add_edge(START, "n")
            self.add_edge("n", END)

    # messages 키가 없는 State에서도 안전 (no-op)
    assert A(hooks=[ContextHook(max_tokens=1)]).invoke({"q": "x"})["q"] == "x"


# ── config 배선 (공통개발자 영역) ───────────────────────────────────────
def test_context_hook_from_config(tmp_path):
    files = {
        **BASE,
        "context.yaml": "max_tokens: 5\nkeep_recent: 2\nmodel: main\n",
        "hooks.yaml": "all:\n  hooks: [context]\n",
    }
    app = KlafiApp.from_config(_dir(tmp_path, files))
    hooks = app.hook_plan.for_agent("chat")
    assert any(isinstance(h, ContextHook) for h in hooks)  # YAML 이름으로 배선됨


def test_unknown_context_key_raises(tmp_path):
    files = {**BASE, "context.yaml": "max_tokenss: 5\n"}
    with pytest.raises(ConfigSchemaError, match="context 설정"):
        KlafiApp.from_config(_dir(tmp_path, files))


def test_context_hook_not_registered_without_config(tmp_path):
    # context.yaml 없이 hooks.yaml에서 참조하면 기동 시 에러
    from klafi.core.exceptions import HookNotFoundError
    from klafi.hookdefs import _NAMED

    _NAMED.pop("context", None)  # 앞선 테스트의 등록 제거
    files = {**BASE, "hooks.yaml": "all:\n  hooks: [context]\n"}
    with pytest.raises(HookNotFoundError):
        KlafiApp.from_config(_dir(tmp_path, files))
