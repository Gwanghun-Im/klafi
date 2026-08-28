"""klafi.core — 개발자 진입 핵심 공개 API.

도메인별 import의 표준 진입점:
    from klafi.core import KlafiGraph, AgentSpec, klafi_node, get_context, ExecutionContext, Hook

예외는 `klafi.core.exceptions`에서 가져온다.
"""

from .base_graph import BaseGraph
from .context import ExecutionContext, bind_context, get_context
from .graph import KlafiGraph, klafi_graph
from .hook import Hook, clear_hooks, register_hook
from .logging_hook import LoggingHook
from .node import klafi_node
from .spec import AgentSpec

__all__ = [
    "KlafiGraph",
    "BaseGraph",
    "AgentSpec",
    "ExecutionContext",
    "get_context",
    "bind_context",
    "Hook",
    "klafi_node",
    "klafi_graph",
    "LoggingHook",
    "register_hook",
    "clear_hooks",
]
