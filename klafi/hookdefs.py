"""명명 Hook 레지스트리 — YAML(config/hooks.yaml)에서 이름으로 훅을 참조.

- prebuilt: event
- 프로젝트 공통 훅 파일에서 @klafi_hook("audit") 로 등록(코드) → YAML에서 이름 참조.
- Logging/Tracing은 KlafiGraph 기본 탑재이므로 여기서 다루지 않는다.
"""

from __future__ import annotations

from typing import Callable

from klafi.core.exceptions import HookNotFoundError
from klafi.core.hook import Hook

HookFactory = Callable[[], Hook]
_NAMED: dict[str, HookFactory] = {}


def register_named_hook(name: str, factory: HookFactory) -> None:
    _NAMED[name] = factory


def klafi_hook(name: str) -> Callable[[type], type]:
    """Hook 하위 클래스를 이름으로 등록하는 데코레이터."""

    def deco(cls: type) -> type:
        register_named_hook(name, cls)  # cls() 로 인스턴스 생성
        return cls

    return deco


def resolve_named_hooks(names: list[str]) -> list[Hook]:
    out: list[Hook] = []
    for n in names:
        f = _NAMED.get(n)
        if f is None:
            raise HookNotFoundError(f"hook '{n}' 미등록 (@klafi_hook 또는 register_named_hook 필요)")
        out.append(f())
    return out


def _event_hook() -> Hook:
    from klafi.events import EventHook

    return EventHook()


register_named_hook("event", _event_hook)
