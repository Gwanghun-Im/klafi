"""Skill — 이름 붙은 Tool 묶음 + 프롬프트 (요구사항 §14.2 확장).

Tool 하나만으로는 "어떻게 쓰는 툴인지"가 LLM에게 전달되지 않는다. Skill은
**툴 묶음 + 사용 지침(prompt)** 을 한 단위로 묶어 LLM에 바인딩한다.

    refund = Skill(name="refund", tools=[lookup_order, issue_refund],
                   prompt="환불은 lookup_order로 주문 확인 후 issue_refund로 처리한다.")

    # 업무 Agent — 툴만이면 .bind_tools(...), 지침까지면 .bind_skills(...)
    llm = init_chat_model("main").bind_skills([refund])

툴은 model.bind_tools로, prompt는 SystemMessage로 자동 주입된다.
Tool과 섞어 넘겨도 된다: .bind_skills([refund, search_policy])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass
class Skill:
    """이름 붙은 툴 묶음 + 사용 지침."""

    name: str
    tools: list[Any] = field(default_factory=list)
    prompt: str | None = None

    def bind_tools(self) -> list[Any]:
        """이 스킬이 노출하는 툴 목록."""
        return list(self.tools)

    @classmethod
    def from_registry(
        cls, registry: Any, name: str, *tool_names: str, prompt: str | None = None
    ) -> "Skill":
        """ToolRegistry에 등록된 툴로 스킬을 만든다. 이름 미지정 시 전체 툴."""
        tools = registry.load(list(tool_names)) if tool_names else registry.all()
        return cls(name=name, tools=tools, prompt=prompt)

    @staticmethod
    def flatten(items: list[Any]) -> tuple[list[Any], list[str]]:
        """Skill이 섞인 목록을 (툴, 프롬프트)로 펼친다."""
        tools: list[Any] = []
        prompts: list[str] = []
        for it in items:
            if isinstance(it, Skill):
                tools.extend(it.bind_tools())
                if it.prompt:
                    prompts.append(it.prompt)
            else:
                tools.append(it)
        return tools, prompts


def bind_skills(model: Any, skills: list[Any]) -> Any:
    """ChatModel.bind_skills의 구현 — 툴은 bind_tools로, prompt는 SystemMessage로.

    업무 코드에서는 init_chat_model(alias).bind_skills([...])로 호출한다.
    """
    tools, prompts = Skill.flatten(skills)
    bound = model.bind_tools(tools)  # 변환은 ChatModel.bind_tools가 처리
    if not prompts:
        return bound
    return _with_system(bound, "\n\n".join(prompts))


def _with_system(runnable: Any, prompt: str) -> Any:
    """prompt를 SystemMessage로 선행 주입한 runnable."""
    from langchain_core.messages import SystemMessage, convert_to_messages
    from langchain_core.runnables import RunnableLambda

    def prepend(x: Any) -> list[Any]:
        return [SystemMessage(prompt), *convert_to_messages(x if isinstance(x, list) else [x])]

    return RunnableLambda(prepend) | runnable

