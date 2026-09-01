"""공통개발자 영역 — 플랫폼 부트스트랩 (server.py / demo.py 공용).

config로 인프라를 조립하고, 플랫폼 공통 Hook(Event/Metrics)·인증·공통 메모리를 구성한 뒤
업무개발자의 agents 를 등록한다. Spring Boot의 @Configuration 에 해당.
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent  # platform/ — config 와 .env 탐색 기준


def _load_env() -> None:
    # 프로젝트 루트(support_platform/)의 .env 를 우선(독립 실행/zip 배포 시 여기).
    # klafi 저장소 안에서 개발할 때는 저장소 루트 .env 로 폴백(기존 워크플로 유지).
    for p in (HERE.parent / ".env", HERE.parents[2] / ".env"):
        if p.exists():
            break
    else:
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.removeprefix("export ").partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# 공통 훅 파일 (코드로 관리 — metrics · 가드레일 · event · context)
from .hooks import PLATFORM_HOOKS, context_hook, metrics  # noqa: E402


def auth(request) -> dict:
    """Authentication Adapter (API-10) — 요청에서 사용자·권한을 security_context로.

    데모: 헤더 X-User 가 있으면 인증 사용자로 보고 권한 부여, 없으면 anon(권한 없음).
    실제로는 토큰 검증 후 역할→권한 매핑. (미인증 anon은 노드 auth 미들웨어에서 차단됨)
    """
    user = request.headers.get("x-user")
    return {
        "user_id": user or "anon",
        "permissions": ["orders:read", "policy:read", "trades:write", "web:search"] if user else [],
    }


def build_app():
    _load_env()  # .env 를 os.environ 에 먼저 로드 → from_config 의 setup_logging 이 KLAFI_LOG_LEVEL 을 본다
    from klafi.app import KlafiApp
    from klafi.context.memory import user_scope

    # 훅·가드레일은 전부 코드로 관리한다(hooks.yaml 없음). platform_hooks 는 항상 적용.
    # KLAFI_ENV=postgres → config/environments/postgres.yaml 계층이 얹혀 실 DB 사용.
    app = KlafiApp.from_config(
        str(HERE / "config"),
        environment=os.environ.get("KLAFI_ENV"),
        platform_hooks=PLATFORM_HOOKS,
    )
    # context 훅은 gateway 요약모델이 필요해 조립 후 주입한다(등록되는 전 에이전트에 적용).
    app.factory.base_hooks.append(context_hook(app.gateway))

    # 업무 Agent 자동 등록 — app/agents/<name>/ 폴더를 훑는다(convention).
    # 업무개발자는 폴더만 떨구면 서비스된다. owner 는 각 agentSpec.py 의 spec.owner 사용.
    app.register_package("app.agents")

    # 플랫폼 공통 Long-Term Memory 사전 시드 (운영: 사용자 온보딩 시 기록)
    app.memory().remember(user_scope("u1"), "pref", {"lang": "ko"})

    return app
