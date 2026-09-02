"""공통 Hook — 프로젝트 전 에이전트에 항상 적용되는 훅을 코드로 관리한다.

Hook = 전역 관측/정책(Hook 서브클래스). YAML(hooks.yaml)로 배치하지 않고 여기 한 곳에서만
코드로 선언한다(가드레일과 동일 방침). 특정 에이전트/노드 전용은 공통 훅에 넣지 않고 그 노드에
@klafi_node 미들웨어·가드레일로 붙인다(예: audit → agents/triage_agent.py).

  PLATFORM_HOOKS 가 유일한 배선 지점이다 — 항목은 두 모양 중 하나(통일 계약):
    · Hook 인스턴스               (gateway 불필요: metrics · 가드레일 · event)
    · (gateway) -> Hook 팩토리    (gateway 필요: context_hook · moderation_hook)
  해석은 KlafiApp.from_config 가 한다 — bootstrap 에 append 없음, 여기 리스트만 고치면 된다.
"""

from klafi import ContextHook
from klafi.core import Hook
from klafi.events import EventHook
from klafi.core.exceptions import ModelNotFoundError
from klafi.guardrail import (
    GuardrailHook,
    injection_llm_guardrail,
    pii,
    profanity_guardrail,
    prompt_injection,
    warn_only,
)

from .guardrails import mask_phone, no_secrets


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
#   input        = 사용자 입력 금칙어(no_secrets)                    — before_agent
#   output       = 최종 응답 전화번호 마스킹(mask_phone) + PII 경고    — after_agent
#   model        = LLM 프롬프트 인젝션                                — before_model
#   model_output = LLM 응답 PII(structured output 포함)              — after_model
#
# 등급 정책: 금칙어·인젝션은 차단(BLOCK), 전화번호는 마스킹(MASK), 그 외 PII는 경고(WARN).
# 상담 답변에는 주문자 이메일 등이 정상적으로 등장할 수 있어 전면 차단하면 오탐이 크다.
# mask_phone 은 차단 대신 치환 → after_agent 가 값 스레딩(_transform)이라 전 에이전트 출력에 적용된다.
platform_guardrails = GuardrailHook(
    input=[no_secrets],
    output=[mask_phone, warn_only(pii)],
    model=[prompt_injection],
    model_output=[warn_only(pii)],
)

# ── 플랫폼 공통 훅 배선(유일한 지점) — 인스턴스든 gateway 팩토리든 여기 한 리스트에 ──
#   event = 실행 생명주기 이벤트 훅(ExecutionStarted/NodeStarted ... → EventBus)
#   (리스트 정의는 파일 하단 — 팩토리 함수 선언 뒤)


def moderation_hook(gateway) -> GuardrailHook:
    """LLM 판정 모더레이션 — 비속어(입력 차단·출력 마스킹) + 인젝션 의미 탐지.

    judge 모델(alias 'judge')이 필요해 context_hook 과 같은 패턴으로 조립 후 주입한다.
    입력은 차단, 출력은 욕설 부분만 *** 마스킹(LLM 이 치환문 생성 — 리프 전체를 날리지 않는다).
    주의: 스트리밍(/stream)은 after_agent 미발화라 출력 검사가 빠진다(입력 차단만) — 프레임워크
    stream TODO 와 동일한 갭. judge alias 는 model.yaml 에서 timeout 3s·재시도 0 으로 등록돼 있다.
    """
    if not gateway.has("judge"):  # fail-fast — 오타/미등록이면 가드레일이 조용히 꺼지는 사고 방지
        raise ModelNotFoundError("moderation_hook: model.yaml 에 'judge' alias 를 등록하세요", model="judge")
    judge = gateway.model("judge")
    return GuardrailHook(
        input=[profanity_guardrail(judge), injection_llm_guardrail(judge)],
        output=[profanity_guardrail(judge, action="mask")],
    )


def context_hook(gateway) -> ContextHook:
    """히스토리 자동압축 훅 (§10.3) — 대화가 임계를 넘으면 오래된 부분을 요약해 입력 토큰을 줄인다.

    summarizer 는 요약용 모델(alias 'fast')이라 gateway 가 있어야 만든다 → PLATFORM_HOOKS 에 미리
    못 넣고, from_config 로 gateway 가 조립된 뒤 bootstrap 이 base_hooks 에 더한다. Checkpoint 원본은
    보존(감사·재현). (이전 context.yaml 의 값을 코드로 옮긴 것 — 훅은 YAML 로 관리하지 않는다.)
    """
    return ContextHook(max_tokens=400, keep_recent=4, summarizer=gateway.model("fast"))


PLATFORM_HOOKS = [
    metrics,               # 인스턴스 — 실행 지표(스냅샷 공유)
    platform_guardrails,   # 인스턴스 — 정규식/블록리스트 4스테이지
    EventHook(),           # 인스턴스 — 생명주기 이벤트
    context_hook,          # 팩토리 — 히스토리 자동압축(요약모델 필요)
    moderation_hook,       # 팩토리 — LLM 모더레이션(judge 모델 필요)
]
