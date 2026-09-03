"""Context Manager (요구사항 §10.3, F05 / CNT-01~09).

장시간 Agent 실행에서 Context Window 증가를 관리한다.
- Token 사용량 측정 (CNT-01)
- Context Threshold (CNT-02)
- Auto Summarization (CNT-03): 오래된 대화를 요약 1건으로 압축
- 중요 Context 보존 (CNT-04): system/important 메시지와 최근 N건은 유지
- Agent Handoff Summary (CNT-08)

메시지는 dict({"role","content"}) 또는 LangChain BaseMessage(.type/.content) 모두 지원.
token_counter/summarizer는 주입형 — 기본 계수기는 LangChain count_tokens_approximately(메시지 단위,
tool_calls 포함, 문자/4). 정확한 값이 필요하면 tiktoken 등을 꽂는다.
"""

from __future__ import annotations

import hashlib
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


def _approx_tokens(messages: list[Message]) -> int:
    """메시지 단위 근사 계수. content 만 세던 공백 계수는 tool_calls 페이로드를 0 으로 세어 툴 루프에서
    임계를 넘지 못했고 한국어도 과소 계산했다."""
    from langchain_core.messages.utils import count_tokens_approximately

    total = 0
    for m in messages:
        if isinstance(m, dict):
            total += int(len(_content(m)) / 4) + 3
        else:
            total += int(count_tokens_approximately([m]))
    return total


def _key(m: Message) -> str:
    mid = None if isinstance(m, dict) else getattr(m, "id", None)
    return mid or hashlib.sha1(f"{_role(m)}:{_content(m)}".encode()).hexdigest()


class ContextManager:
    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        *,
        max_tokens: int = 3000,
        keep_recent: int = 4,
        summarizer: Summarizer | None = None,
    ) -> None:
        self._count = token_counter  # None 이면 메시지 단위 근사(count_tokens_approximately)
        self._max = max_tokens
        self._keep = keep_recent
        self._summarize = summarizer
        # 같은 과거 히스토리를 노드마다·턴마다 다시 요약하지 않도록 (old 메시지 키 튜플 → 요약) 기억한다.
        # 노드 view 만 압축하고 checkpoint 는 원본을 유지하므로, 이 캐시가 없으면 요약 LLM 호출이 O(노드×턴).
        self._summary_cache: dict[tuple[str, ...], str] = {}

    def count_tokens(self, messages: list[Message]) -> int:  # CNT-01
        if self._count is not None:
            return sum(self._count(_content(m)) for m in messages)
        return _approx_tokens(messages)

    def over_threshold(self, messages: list[Message]) -> bool:  # CNT-02
        return self.count_tokens(messages) > self._max

    def manage(self, messages: list[Message]) -> list[Message]:
        """Threshold를 넘을 때만 압축 (CNT-02). 아니면 그대로."""
        if self.over_threshold(messages):
            return self.reduce(messages)
        return messages

    def reduce(self, messages: list[Message]) -> list[Message]:
        """오래된 대화를 요약/제거하고 중요·최근 메시지는 **원래 순서대로** 보존 (CNT-03/04)."""
        important = [m for m in messages if _is_important(m)]
        non_important = [m for m in messages if not _is_important(m)]

        cut = max(0, len(non_important) - self._keep) if self._keep else len(non_important)  # 음수 슬라이스 금지
        # 절단면은 human 턴에서만: tool_use(assistant)↔tool_result(tool) 쌍이 갈라지거나 assistant/tool 로
        # 시작하는 히스토리가 모델에 가면 Anthropic 400("tool_result without tool_use"). LangChain
        # trim_messages(start_on="human") 와 같은 규칙. 되감은 만큼 recent 가 keep_recent 보다 길어질 수 있다.
        # ponytail: 한 human 뒤에 tool 루프가 길면 압축이 안 됨 — 필요하면 tool 쌍 단위 절단으로 완화.
        while 0 < cut < len(non_important) and _role(non_important[cut]) not in ("human", "user"):
            cut -= 1
        old, recent = non_important[:cut], non_important[cut:]
        if not old:
            return messages  # 줄일 대상 없음

        keep = {id(m) for m in important} | {id(m) for m in recent}
        result = [m for m in messages if id(m) in keep]  # important 를 앞으로 끌어올리지 않는다 — 순서 보존
        if self._summarize is not None:  # CNT-03
            summary_msg = {"role": "system", "content": f"[이전 대화 요약] {self._summary_for(old)}", "summary": True}
            at = 0
            while at < len(result) and _role(result[at]) == "system":
                at += 1
            result.insert(at, summary_msg)  # 선두 system 블록 바로 뒤 (Anthropic: system 은 연속이어야 함)
        # summarizer가 없으면 old는 그냥 제거(압축)
        return result

    def _summary_for(self, old: list[Message]) -> str:
        key = tuple(_key(m) for m in old)
        cached = self._summary_cache.get(key)
        if cached is not None:
            return cached
        joined = "\n".join(f"{_role(m)}: {_content(m)}" for m in old)
        summary = self._summarize(f"다음 대화를 간결히 요약:\n{joined}")  # type: ignore[misc]
        if len(self._summary_cache) >= 32:  # ponytail: 작은 LRU — 스레드 수만큼 커지지 않게
            self._summary_cache.pop(next(iter(self._summary_cache)))
        self._summary_cache[key] = summary
        return summary

    def handoff_summary(self, messages: list[Message]) -> str:  # CNT-08
        joined = "\n".join(f"{_role(m)}: {_content(m)}" for m in messages)
        if self._summarize is not None:
            return self._summarize(f"에이전트 인계를 위해 핵심만 요약:\n{joined}")
        # 요약기 없으면 최근 메시지 원문을 이어붙임
        return " | ".join(_content(m) for m in messages[-self._keep :])
