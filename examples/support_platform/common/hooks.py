"""공통 Hook — 프로젝트 전 에이전트에 항상 적용되는 훅을 코드로 관리한다.

Hook = 전역 관측/정책(Hook 서브클래스). YAML(hooks.yaml)로 배치하지 않고 여기 한 곳에서만
코드로 선언한다(가드레일과 동일 방침). 특정 에이전트/노드 전용은 공통 훅에 넣지 않고 그 노드에
@klafi_node 미들웨어·가드레일로 붙인다(예: audit → agents/triage_agent.py).

  PLATFORM_HOOKS   config 불필요한 공통 훅(metrics · 가드레일 · event). from_config 에 그대로 전달.
  context_hook(gw) 히스토리 자동압축 훅. summarizer 가 gateway 요약모델을 필요로 하므로
                   조립(from_config) 이후 bootstrap 에서 gateway 를 주입해 base_hooks 에 더한다.
"""

from klafi import ContextHook
from klafi.core import Hook
from klafi.events import EventHook
from klafi.guardrail import GuardrailHook, pii, prompt_injection, warn_only

from .guardrails import no_secrets


class MetricsHook(Hook):
    """플랫폼 공통 실행 지표 수집 — 모든 Node/Agent 실행을 계측."""

    priority = 50
    fail_open = True  # 계측 실패가 업무를 막지 않음

    def __init__(self) -> None:
        self.agent_runs = 0
        self.node_calls = 0
        self.errors = 0

    def before_agent(self, inp, ctx):
        self.agent_runs += 1

    def before_node(self, node, state, ctx):
        self.node_calls += 1

    def on_node_error(self, node, state, exc, ctx):
        self.errors += 1

    def snapshot(self) -> dict:
        return {"agent_runs": self.agent_runs, "node_calls": self.node_calls, "errors": self.errors}


# 상태 공유 훅(스냅샷용)은 인스턴스로.
metrics = MetricsHook()

# 플랫폼 공통 가드레일 — 전 에이전트 4스테이지.
#   input        = 사용자 입력 금칙어(no_secrets)               — before_agent
#   output       = 최종 응답 PII                                — after_agent
#   model        = LLM 프롬프트 인젝션                          — before_model
#   model_output = LLM 응답 PII(structured output 포함)         — after_model
#
# 등급 정책: 금칙어·인젝션은 차단(BLOCK), PII는 경고(WARN).
# 상담 답변에는 주문자 이메일 등이 정상적으로 등장할 수 있어 전면 차단하면 오탐이 크다.
# 대신 위반은 severity=warn 으로 전부 기록되므로 운영에서 탐지·집계할 수 있다.
platform_guardrails = GuardrailHook(
    input=[no_secrets],
    output=[warn_only(pii)],
    model=[prompt_injection],
    model_output=[warn_only(pii)],
)

# config 불필요한 공통 훅 — 이 리스트로만 관리한다.
#   event = 실행 생명주기 이벤트 훅(ExecutionStarted/NodeStarted ... → EventBus)
PLATFORM_HOOKS = [metrics, platform_guardrails, EventHook()]


def context_hook(gateway) -> ContextHook:
    """히스토리 자동압축 훅 (§10.3) — 대화가 임계를 넘으면 오래된 부분을 요약해 입력 토큰을 줄인다.

    summarizer 는 요약용 모델(alias 'fast')이라 gateway 가 있어야 만든다 → PLATFORM_HOOKS 에 미리
    못 넣고, from_config 로 gateway 가 조립된 뒤 bootstrap 이 base_hooks 에 더한다. Checkpoint 원본은
    보존(감사·재현). (이전 context.yaml 의 값을 코드로 옮긴 것 — 훅은 YAML 로 관리하지 않는다.)
    """
    return ContextHook(max_tokens=400, keep_recent=4, summarizer=gateway.model("fast"))
