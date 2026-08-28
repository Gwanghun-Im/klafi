"""공통 Guardrail — 프로젝트의 모든 가드레일을 여기서 관리한다.

가드레일 = 순수 문자열 정책(.check 보유). @klafi_node/@klafi_graph 의 input·output 이나
GuardrailHook(공통 훅)에 넣는다. klafi prebuilt(pii, prompt_injection)는 klafi.guardrail 에서.

세 가지 처리 등급을 모두 보여준다:
  no_secrets    BLOCK — 차단
  refund_policy BLOCK — 차단
  mask_phone    MASK  — 차단하지 않고 치환(마스킹)
"""

import re

from klafi.guardrail import GuardrailResult, guardrail

_PHONE = re.compile(r"01[016-9]-?\d{3,4}-?\d{4}")


@guardrail
def no_secrets(text: str) -> GuardrailResult:
    hit = next((w for w in ("비밀번호", "주민번호") if w in text), None)
    return GuardrailResult(hit is None, f"금칙어 '{hit}'" if hit else None)


@guardrail
def refund_policy(text: str) -> GuardrailResult:
    return GuardrailResult("무단환불" not in text, "무단환불 정책 위반")


@guardrail
def mask_phone(text: str) -> GuardrailResult:
    """휴대폰 번호는 막지 않고 가린다 — replacement 를 주면 차단 대신 치환된다.

    순수 문자열 정책이다. 값의 모양(state dict, 메시지 객체, tool kwargs, LLM str)은 바인딩이
    처리하므로, 이 하나가 노드·그래프·tool·gateway 경계 어디에나 그대로 꽂힌다. 메시지 id 유지·
    add_messages 리듀서 같은 LangGraph 지식은 몰라도 된다.
    """
    if not _PHONE.search(text):
        return GuardrailResult(True)
    return GuardrailResult(False, "휴대폰 번호 마스킹", replacement=_PHONE.sub("010-****-****", text))
