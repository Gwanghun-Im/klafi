"""klafi.observability.logging — 앱 로거 부트스트랩 (setup_tracing 의 형제).

KlafiApp.from_config 가 자동 호출한다. 로깅 설정은 공통 제어라 프레임워크가 소유한다
— 프로젝트마다 복붙하던 uvicorn 포매터 재사용·SDK 로거 억제를 여기 한 곳에 둔다.
"""

from __future__ import annotations

import logging
import os


def setup_logging(level: str | None = None) -> None:
    """앱 로거 출력을 구성한다 (KLAFI_LOG_LEVEL, 기본 INFO).

    설정하지 않으면 파이썬은 logging.lastResort(WARNING)로만 출력하므로, 가드레일 위반(WARNING)은
    보이지만 audit 같은 INFO 로그는 조용히 사라진다. uvicorn 은 uvicorn.* 로거만 구성하고 root 는
    건드리지 않아(--log-level 로도 앱 로거는 안 바뀜) 여기서 직접 잡아준다.

    호스트 앱 존중: KLAFI_LOG_SETUP=0 이면 건너뛴다. root 로거가 이미 구성돼 있으면
    logging.basicConfig(force 없음)가 자동으로 no-op 이 되어 기존 설정을 덮지 않는다.
    """
    if os.environ.get("KLAFI_LOG_SETUP") == "0":
        return
    level = level or os.environ.get("KLAFI_LOG_LEVEL", "INFO")
    handler = logging.StreamHandler()
    # uvicorn 로그만 색이 있는 이유: uvicorn 은 자기 로거(uvicorn/uvicorn.access, propagate=False)에
    # ColourizedFormatter 를 단다. 앱 로거는 root 핸들러를 타는데 표준 Formatter 엔 색이 없어,
    # 같은 포매터를 재사용해 레벨 색을 맞춘다(터미널일 때만 자동 적용).
    try:
        from uvicorn.logging import DefaultFormatter

        handler.setFormatter(DefaultFormatter("%(levelprefix)s %(name)s: %(message)s"))
    except ImportError:  # server extra 없이 돌 때
        handler.setFormatter(logging.Formatter("%(levelname)-8s %(name)s: %(message)s"))
    logging.basicConfig(level=level, handlers=[handler])
    # SDK 내부 HTTP 로그는 앱 로그를 가려서 WARNING 이상만.
    for noisy in ("httpx", "httpcore", "anthropic", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
