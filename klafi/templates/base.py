"""Template 공통 타입.

Template들은 KlafiGraph를 상속하고 define()에서 그래프를 조립한다.
model/retriever는 주입형 callable (Model Gateway 자리).
"""

from __future__ import annotations

from typing import Protocol


class Model(Protocol):
    def __call__(self, prompt: str) -> str: ...


class Retriever(Protocol):
    def __call__(self, query: str) -> list[str]: ...
