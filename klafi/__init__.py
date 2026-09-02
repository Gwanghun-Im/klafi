"""KLAFI — Enterprise Agentic AI Engineering Framework."""

__version__ = "0.1.4"

from klafi.app import KlafiApp
from klafi.core.base_graph import BaseGraph
from klafi.core.context import ExecutionContext, bind_context, get_context
from klafi.core.exceptions import (
    AgentExecutionException,
    AgentNotFoundError,
    ApprovalException,
    CheckpointException,
    ConfigException,
    ConfigNotFoundError,
    ConfigSchemaError,
    ConfigValueError,
    ContextException,
    GuardrailException,
    GuardrailViolationError,
    HookNotFoundError,
    KlafiException,
    ModelException,
    ModelNotConfiguredError,
    ModelNotFoundError,
    NotFoundError,
    PermissionDeniedError,
    PolicyException,
    TimeoutException,
    ToolException,
    ToolNotFoundError,
    ToolPermissionError,
    ToolValidationError,
    ValidationError,
    ViolationError,
)
from klafi.core.graph import KlafiGraph, klafi_graph
from klafi.core.hook import Hook, clear_hooks, register_hook
from klafi.core.node import klafi_node
from klafi.context.checkpoint import register_checkpointer, resolve_checkpointer
from klafi.config.layered import LayeredConfig, deep_merge
from klafi.context.hook import ContextHook
from klafi.context.manager import ContextManager
from klafi.context.memory import (
    MemoryStore,
    agent_scope,
    project_scope,
    redact_pii,
    register_store,
    resolve_store,
    user_scope,
)
from klafi.core.logging_hook import LoggingHook
from klafi.core.spec import AgentSpec
from klafi.observability.logging import setup_logging
from klafi.observability.tracing import TracingHook, setup_tracing, span
from klafi.evaluation import (
    CustomEvaluator,
    EvaluationReport,
    LLMJudgeEvaluator,
    RuleEvaluator,
    run_offline,
)
from klafi.guardrail import (
    BLOCK,
    MASK,
    WARN,
    BlocklistGuardrail,
    GuardrailHook,
    GuardrailResult,
    LLMGuardrail,
    RegexGuardrail,
    banned_topics_guardrail,
    enforce,
    guardrail,
    injection_llm_guardrail,
    pii,
    pii_guardrail,
    profanity_guardrail,
    prompt_injection,
    warn_only,
)
from klafi.hookdefs import klafi_hook, register_named_hook
from klafi.model import ModelGateway, ModelResult, init_chat_model, set_active_gateway
from klafi.events import Event, EventHook, EventType, emit, subscribe
from klafi.registry import AgentLifecycle, AgentRecord, AgentRegistry
from klafi.tool import Skill, Tool, ToolRegistry, tool
from klafi.templates import RAGAgent, SimpleAgent, SupervisorAgent
from klafi.runtime.factory import ExecutionFactory
from klafi.runtime.policy import ExecutionPolicy
from klafi.runtime.state import ExecutionState

__all__ = [
    "BaseGraph",
    "KlafiGraph",
    "KlafiApp",
    "AgentSpec",
    "ExecutionContext",
    "get_context",
    "bind_context",
    "Hook",
    "LoggingHook",
    "register_hook",
    "clear_hooks",
    "klafi_node",
    "klafi_graph",
    "ExecutionFactory",
    "ExecutionPolicy",
    "ExecutionState",
    "register_checkpointer",
    "resolve_checkpointer",
    "MemoryStore",
    "user_scope",
    "agent_scope",
    "project_scope",
    "redact_pii",
    "register_store",
    "resolve_store",
    "ContextManager",
    "ContextHook",
    "TracingHook",
    "setup_logging",
    "setup_tracing",
    "span",
    "SimpleAgent",
    "RAGAgent",
    "SupervisorAgent",
    "ModelGateway",
    "ModelResult",
    "init_chat_model",
    "set_active_gateway",
    "GuardrailHook",
    "LLMGuardrail",
    "profanity_guardrail",
    "injection_llm_guardrail",
    "banned_topics_guardrail",
    "GuardrailResult",
    "BlocklistGuardrail",
    "RegexGuardrail",
    "pii_guardrail",
    "guardrail",
    "enforce",
    "pii",
    "prompt_injection",
    "BLOCK",
    "WARN",
    "MASK",
    "warn_only",
    "klafi_hook",
    "register_named_hook",
    "RuleEvaluator",
    "LLMJudgeEvaluator",
    "CustomEvaluator",
    "EvaluationReport",
    "run_offline",
    "Tool",
    "ToolRegistry",
    "tool",
    "Skill",
    "AgentRegistry",
    "AgentRecord",
    "AgentLifecycle",
    "Event",
    "EventType",
    "EventHook",
    "emit",
    "subscribe",
    "LayeredConfig",
    "deep_merge",
    # 예외 — 도메인 축
    "KlafiException",
    "AgentExecutionException",
    "TimeoutException",
    "ModelException",
    "ToolException",
    "PolicyException",
    "GuardrailException",
    "ContextException",
    "ConfigException",
    "CheckpointException",
    "ApprovalException",
    # 예외 — 종류 축
    "NotFoundError",
    "ValidationError",
    "PermissionDeniedError",
    "ViolationError",
    # 예외 — 구체 타입
    "ConfigNotFoundError",
    "ConfigSchemaError",
    "ConfigValueError",
    "ToolNotFoundError",
    "ToolPermissionError",
    "ToolValidationError",
    "ModelNotFoundError",
    "ModelNotConfiguredError",
    "GuardrailViolationError",
    "HookNotFoundError",
    "AgentNotFoundError",
]
