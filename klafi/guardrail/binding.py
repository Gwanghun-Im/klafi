"""바인딩 — 값의 문자열 리프에만 정책(str→str)을 적용하고 구조·타입·id를 유지한다.

가드레일은 "문자열 정책" 하나로 통일된다(check(text)->판정/치환text). 값의 모양(state dict,
BaseMessage, tool kwargs, LLM str)을 아는 것은 이 바인딩의 몫이다. 덕분에 통짜 json.dumps
투영이 사라져 역변환이 필요 없고, 메시지 id가 보존되며(add_messages 리듀서가 append 아닌 교체),
엉뚱한 필드(id·metadata·토큰수)를 스캔·치환하지 않는다.

    Binding = Callable[[value, Callable[[str], str]], value']

판정만 하는 가드레일은 fn이 리프를 그대로 돌려주고, 그러면 바인딩도 원본을 그대로 반환한다
(하위가 전부 is-동일하면 재조립하지 않는다).
"""

from __future__ import annotations

from typing import Any, Callable

from klafi.core.exceptions import GuardrailException

Leaf = Callable[[str], str]


def _is_message(v: Any) -> bool:
    # LangChain BaseMessage: .content(str 또는 블록리스트) + .model_copy(pydantic)
    return hasattr(v, "content") and hasattr(v, "model_copy")


def _is_model(v: Any) -> bool:
    # pydantic BaseModel(툴의 구조화 출력 등) — dict 로 펼쳐 순회하고 같은 타입으로 재조립한다
    return hasattr(v, "model_dump") and hasattr(v, "model_validate")


def _role(m: Any) -> Any:
    return m.get("role") if isinstance(m, dict) else getattr(m, "type", None)


def _tail_start(msgs: list) -> int:
    """검사 대상 꼬리의 시작 인덱스. 기본은 마지막 메시지 하나지만, 병렬 툴콜은 한 스텝에 ToolMessage
    여러 개를 붙이므로 마지막이 tool 이면 연속한 tool 메시지 전부를 포함한다."""
    i = len(msgs) - 1
    while i > 0 and _role(msgs[i]) == "tool" and _role(msgs[i - 1]) == "tool":
        i -= 1
    return i


def _opaque(v: Any, fn: Leaf) -> Any:
    """매핑 불가 리프(pydantic 툴 출력·커스텀 객체 등). 검사는 하되 치환은 막는다.

    조용히 건너뛰면 '가드레일이 안 돌았는데 통과'가 되어 보안 회귀 → str(v)로 검사 커버리지는
    현행(json.dumps default=str)과 같게 유지하고, 치환 시도만 예외로 막는다.
    """
    s = str(v)
    if fn(s) != s:
        raise GuardrailException(
            f"{type(v).__name__} 은 치환할 수 없습니다 — 구조를 치환하려면 "
            "@guardrail(raw=True) 로 원본을 받아 같은 타입으로 반환하세요"
        )
    return v


def bind(v: Any, fn: Leaf) -> Any:
    """값의 str 리프를 fn으로 매핑. 변경이 없으면 원본 객체를 그대로 반환한다."""
    if isinstance(v, str):
        return fn(v)
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if k == "messages" and isinstance(x, list) and x:
                # ponytail: messages는 방금 들어온/만들어진 꼬리만 본다 — 앞의 것은 마지막이었을 때
                # 이미 검사됐다. 꼬리 = 마지막 메시지, 단 병렬 툴콜의 연속 ToolMessage 는 전부.
                i = _tail_start(x)
                tail = [bind(m, fn) for m in x[i:]]
                out[k] = x if all(a is b for a, b in zip(tail, x[i:])) else [*x[:i], *tail]
            else:
                out[k] = bind(x, fn)
        return out if any(out[k] is not v[k] for k in v) else v
    if isinstance(v, (list, tuple)):
        new = [bind(x, fn) for x in v]
        return type(v)(new) if any(a is not b for a, b in zip(new, v)) else v
    if _is_message(v):
        c = bind(v.content, fn)  # content 하나만 — id·name·type은 건드리지 않는다
        return v.model_copy(update={"content": c}) if c is not v.content else v
    if _is_model(v):
        data = v.model_dump()
        new = bind(data, fn)
        return v if new is data else type(v).model_validate(new)
    if v is None or isinstance(v, (int, float, bool)):
        return v  # 텍스트 리프 아님 — 건너뜀
    return _opaque(v, fn)


def whole(v: Any, fn: Leaf) -> Any:
    """raw=True 가드레일용 — 리프가 값 전체다(순회하지 않고 그대로 넘긴다)."""
    return fn(v)


def binding_for(value: Any) -> Callable[[Any, Leaf], Any]:
    """값의 모양으로 바인딩을 고른다. str이면 항등(리프=값), 아니면 구조 순회."""
    return whole if isinstance(value, str) else bind
