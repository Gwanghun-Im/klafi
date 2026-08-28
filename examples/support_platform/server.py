"""공통개발자 영역 — ASGI 엔트리. 등록된 전 Agent를 HTTP로 서비스.

실행:  uvicorn support_platform.server:app --app-dir examples --port 8078
  - http://127.0.0.1:8078/docs  (Swagger)
  - http://127.0.0.1:8078/app   (웹 채팅 클라이언트, klafi 패키지에 내장)

호출 시 헤더에 X-User 를 넣으면 auth가 사용자·권한(orders:read)을 주입한다.

이 파일은 klafi 저장소 바깥으로 프로젝트 전체를 복사·zip 해도 그대로 동작한다
(웹 클라이언트가 저장소 상대경로가 아니라 `pip install klafi[server]`로 함께 설치되므로).

--app-dir 은 support_platform 의 부모(examples)를 가리킨다: support_platform 을 직접 path 에
올리면 그 안의 platform/ 하위폴더가 stdlib `platform` 모듈을 가려버리기 때문이다.
"""

from klafi.server import mount_frontend

from .platform.bootstrap import auth, build_app

app = build_app().http_app(auth=auth)
mount_frontend(app)
