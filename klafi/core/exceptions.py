"""KLAFI 공통 Exception 체계 (요구사항 §23).

프로젝트마다 Exception 처리가 갈라지는 것을 막기 위해 단일 계층을 둔다.
발생 시 error_code / execution_id / agent_id / node / tool / trace_id 등을 함께 담는다.

두 축으로 잡을 수 있다:
  · 도메인 축 — ToolException, ModelException, GuardrailException, ConfigException ...
  · 종류 축   — NotFoundError(못찾음), ValidationError(검증실패), PermissionError(권한),
               ViolationError(정책·가드레일 위반)

    except ToolException:      # tool 관련 전부
    except NotFoundError:      # tool·model·guardrail·hook·agent 미등록 전부
    except GuardrailViolationError:  # 실행 중 가드레일 차단만
"""

from __future__ import annotations

from typing import Any


class KlafiException(Exception):
    """모든 KLAFI 예외의 뿌리."""

    error_code: str = "KLAFI_ERROR"

    def __init__(self, message: str = "", **context: Any) -> None:
        super().__init__(message)
        self.message = message
        # execution_id, agent_id, node, tool, trace_id, original_exception 등 자유 부착
        self.context: dict[str, Any] = context

    def __str__(self) -> str:
        if self.context:
            extra = " ".join(f"{k}={v}" for k, v in self.context.items())
            return f"[{self.error_code}] {self.message} ({extra})"
        return f"[{self.error_code}] {self.message}"


class AgentExecutionException(KlafiException):
    error_code = "AGENT_EXECUTION_ERROR"


class ModelException(KlafiException):
    error_code = "MODEL_ERROR"


class ToolException(KlafiException):
    error_code = "TOOL_ERROR"


class PolicyException(KlafiException):
    error_code = "POLICY_ERROR"


class GuardrailException(KlafiException):
    error_code = "GUARDRAIL_ERROR"


class ContextException(KlafiException):
    error_code = "CONTEXT_ERROR"


class ConfigException(KlafiException):
    error_code = "CONFIG_ERROR"


class CheckpointException(KlafiException):
    error_code = "CHECKPOINT_ERROR"


class ApprovalException(KlafiException):
    error_code = "APPROVAL_ERROR"


class TimeoutException(AgentExecutionException):
    error_code = "TIMEOUT_ERROR"


# ── 종류 축 (도메인과 교차해서 잡을 때) ────────────────────────────────
class NotFoundError(KlafiException):
    """이름으로 찾지 못함 (tool/model alias/guardrail/hook/agent). 대개 설정 오류."""

    error_code = "NOT_FOUND"


class ValidationError(KlafiException):
    """입력·출력·스키마 검증 실패."""

    error_code = "VALIDATION_ERROR"


class PermissionDeniedError(KlafiException):
    """권한 부족 (최소권한 원칙 위반)."""

    error_code = "PERMISSION_DENIED"


class ViolationError(KlafiException):
    """실행 중 정책·가드레일 위반으로 차단됨."""

    error_code = "VIOLATION"


# ── Config (설정) ──────────────────────────────────────────────────────
class ConfigNotFoundError(ConfigException, NotFoundError):
    """설정 파일·디렉터리 미설정."""

    error_code = "CONFIG_NOT_FOUND"


class ConfigSchemaError(ConfigException, ValidationError):
    """알 수 없는 키·stage 등 스키마 위반(오타)."""

    error_code = "CONFIG_SCHEMA_ERROR"


class ConfigValueError(ConfigException, ValidationError):
    """설정 값이 유효하지 않음(지원하지 않는 type 등)."""

    error_code = "CONFIG_VALUE_ERROR"


# ── 못찾음 (도메인별) ──────────────────────────────────────────────────
class ToolNotFoundError(ToolException, NotFoundError):
    error_code = "TOOL_NOT_FOUND"


class ModelNotFoundError(ModelException, NotFoundError):
    """model alias 미등록."""

    error_code = "MODEL_NOT_FOUND"


class ModelNotConfiguredError(ModelException, ConfigException):
    """모델은 등록됐지만 API 키 등 설정이 없어 호출 불가."""

    error_code = "MODEL_NOT_CONFIGURED"


class HookNotFoundError(NotFoundError):
    error_code = "HOOK_NOT_FOUND"


class AgentNotFoundError(NotFoundError):
    """Agent가 Runtime 또는 Registry에 없음."""

    error_code = "AGENT_NOT_FOUND"


# ── 실행 중 ────────────────────────────────────────────────────────────
class GuardrailViolationError(GuardrailException, ViolationError):
    """실행 중 가드레일을 통과하지 못해 차단됨 (fail-close)."""

    error_code = "GUARDRAIL_VIOLATION"


class ToolPermissionError(ToolException, PermissionDeniedError):
    error_code = "TOOL_PERMISSION_DENIED"


class ToolValidationError(ToolException, ValidationError):
    """Tool 입력·출력 검증 실패."""

    error_code = "TOOL_VALIDATION_ERROR"
