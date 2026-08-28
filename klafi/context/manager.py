"""Context Manager (요구사항 §10.3, F05 / CNT-01~09).

장시간 Agent 실행에서 Context Window 증가를 관리한다.
- Token 사용량 측정 (CNT-01)
- Context Threshold (CNT-02)
- Auto Summarization (CNT-03): 오래된 대화를 요약 1건으로 압축
- 중요 Context 보존 (CNT-04): system/important 메시지와 최근 N건은 유지
- Agent Handoff Summary (CNT-08)

메시지는 dict({"role","content"}) 또는 LangChain BaseMessage(.type/.content) 모두 지원.
token_counter/summarizer는 주입형 — 실제로는 tiktoken/Model Gateway를 꽂는다.
"""

from __future__ import annotations

from typing import Any, Callable

Message = Any
TokenCounter = Callable[[str], int]
Summarizer = Callable[[str], str]


def _content(m: Message) -> str:
    if isinstance(m, dict):
        return str(m.get("content", ""))
    return str(getattr(m, "content", ""))


def _role(m: Message) -> str:
    if isinstance(m, dict):
        return str(m.get("role", ""))
    return str(getattr(m, "type", ""))


def _is_important(m: Message) -> bool:
    if _role(m) == "system":
        return True
    if isinstance(m, dict):
        return bool(m.get("important"))
    return bool(getattr(m, "important", False))


def _naive_tokens(s: str) -> int:
    return len(s.split())


class ContextManager:
    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        *,
        max_tokens: int = 3000,
        keep_recent: int = 4,
        summarizer: Summarizer | None = None,
    ) -> None:
        self._count = token_counter or _naive_tokens
        self._max = max_tokens
        self._keep = keep_recent
        self._summarize = summarizer

    def count_tokens(self, messages: list[Message]) -> int:  # CNT-01
        return sum(self._count(_content(m)) for m in messages)

    def over_threshold(self, messages: list[Message]) -> bool:  # CNT-02
        return self.count_tokens(messages) > self._max

    def manage(self, messages: list[Message]) -> list[Message]:
        """Threshold를 넘을 때만 압축 (CNT-02). 아니면 그대로."""
        if self.over_threshold(messages):
            return self.reduce(messages)
        return messages

    def reduce(self, messages: list[Message]) -> list[Message]:
        """오래된 대화를 요약/제거하고 중요·최근 메시지는 보존 (CNT-03/04)."""
        important = [m for m in messages if _is_important(m)]
        non_important = [m for m in messages if not _is_important(m)]

        recent = non_important[-self._keep :] if self._keep else []
        old = non_important[: len(non_important) - len(recent)]
        if not old:
            return messages  # 줄일 대상 없음

        result = list(important)
        if self._summarize is not None:  # CNT-03
            joined = "\n".join(f"{_role(m)}: {_content(m)}" for m in old)
            summary = self._summarize(f"다음 대화를 간결히 요약:\n{joined}")
            result.append({"role": "system", "content": f"[이전 대화 요약] {summary}", "summary": True})
        # summarizer가 없으면 old는 그냥 제거(압축)
        result.extend(recent)
        return result

    def handoff_summary(self, messages: list[Message]) -> str:  # CNT-08
        joined = "\n".join(f"{_role(m)}: {_content(m)}" for m in messages)
        if self._summarize is not None:
            return self._summarize(f"에이전트 인계를 위해 핵심만 요약:\n{joined}")
        # 요약기 없으면 최근 메시지 원문을 이어붙임
        return " | ".join(_content(m) for m in messages[-self._keep :])
