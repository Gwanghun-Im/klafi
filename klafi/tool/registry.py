"""Tool Registry (요구사항 §14.2 / TOL-03, FAC-05).

Agent별 Tool을 이름으로 보관·조회하고, 필요한 Tool 집합을 Loading한다.
"""

from __future__ import annotations

from klafi.core.exceptions import ToolNotFoundError

from .tool import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(f"tool '{name}' 미등록", tool=name) from None

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def load(self, names: list[str]) -> list[Tool]:
        """Agent별 Tool 자동 Loading (FAC-05)."""
        return [self.get(n) for n in names]
