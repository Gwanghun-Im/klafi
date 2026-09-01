"""AgentSpec / Config (요구사항 F01 SDK-02·03, §22).

Agent Metadata 표준과 Config 구조. YAML 또는 객체 기반 모두 허용.
"""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel, Field


class AgentSpec(BaseModel):
    """Agent Metadata 표준 (SDK-02). Registry/Trace/Context가 공유하는 신원."""

    id: str
    name: str
    version: str = "0.1.0"
    project: str | None = None
    owner: str | None = None
    description: str | None = None
    agent_type: str | None = None  # simple / rag / supervisor / plan-executor / hitl
    print: bool = False  # True 면 서버 부팅(등록) 시 컴파일된 그래프를 터미널에 그린다(디버깅/문서화)

    # 자동 주입 대상 설정 (실제 Adapter 연결은 이후 WS2/WS3에서). 지금은 식별자만.
    model: str | None = None  # Model Alias (MOD-02)
    tools: list[str] = Field(default_factory=list)

    # 실행정책·기타 설정. 코드 하드코딩 금지 원칙에 따라 여기로 외부화.
    config: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "AgentSpec":
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))
