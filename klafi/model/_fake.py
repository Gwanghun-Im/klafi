"""FunctionChatModel — (prompt)->str 함수를 LangChain chat model 로 감싸는 어댑터.

FunctionProvider.chat_model() 이 쓴다. 키 없이(echo 등) init_chat_model 표준 경로
(bind_tools/bind_skills/with_structured_output/stream)를 테스트·데모하기 위한 것이지,
실제 tool-calling 을 흉내내지는 않는다. 마지막 메시지 content 에 함수를 적용해 응답한다.

langchain 은 선택적 의존이므로(providers 는 lazy import) 이 모듈도 import 시점에만 로드된다.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


def _last_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        c = getattr(m, "content", None)
        if isinstance(c, str):
            return c
    return ""


class FunctionChatModel(BaseChatModel):
    """(prompt: str) -> str 함수를 감싼 최소 chat model."""

    fn: Callable[[str], str]

    @property
    def _llm_type(self) -> str:
        return "klafi-function"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self.fn(_last_text(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # 스트리밍 표준 경로 검증용 — 통짜 텍스트를 한 청크로 흘린다.
        text = self.fn(_last_text(messages))
        yield ChatGenerationChunk(message=AIMessage(content=text))

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        # tool-calling 은 흉내내지 않는다. 바인딩 호출만 통과시켜 표준 스타일 코드가 돌게 한다.
        return self
