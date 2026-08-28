"""KLAFI 채팅 웹 클라이언트 — 패키지 정적 자산으로 배포된다 (API-09 부속).

에이전트를 만드는 프로젝트(support_platform 등)는 이 파일을 소스로 갖지 않는다.
`pip install klafi[server]`로 klafi 안에 함께 설치되므로, 앱 쪽은 mount_frontend(app) 한 줄이면 된다
— 저장소 상대경로(`../../frontend`)에 의존하지 않아, 프로젝트를 어디로 복사·zip해도 그대로 동작한다.
"""

from __future__ import annotations

import importlib.resources as resources
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


class _NoCacheStatic(StaticFiles):
    """개발용 정적 서빙 — 브라우저 캐시 금지.

    캐시가 남으면 index.html을 고쳐도 옛 UI가 그대로 떠서 '수정이 안 먹는다'로 오인된다.
    """

    def file_response(self, *args: Any, **kwargs: Any) -> Any:
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        return resp


def mount_frontend(app: FastAPI, path: str = "/app") -> None:
    """웹 채팅 클라이언트를 API와 same-origin으로 마운트 (CORS 설정 불필요).

        app = build_app().http_app(auth=auth)
        mount_frontend(app)   # http://.../app 에서 채팅 UI
    """
    static_dir = resources.files("klafi.server") / "static"
    app.mount(path, _NoCacheStatic(directory=str(static_dir), html=True), name="klafi-frontend")
