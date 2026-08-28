"""공통개발자 영역 — ASGI 엔트리. 등록된 전 Agent를 HTTP로 서비스.

실행(이 폴더 안에서):  uvicorn server:app --port 8078
  - http://127.0.0.1:8078/docs  (Swagger)
  - http://127.0.0.1:8078/app   (웹 채팅 클라이언트, klafi 패키지에 내장)

호출 시 헤더에 X-User 를 넣으면 auth가 사용자·권한(orders:read)을 주입한다.

이 파일은 klafi 저장소 바깥으로 프로젝트 전체를 복사·zip 해도 그대로 동작한다
(웹 클라이언트가 저장소 상대경로가 아니라 `pip install klafi[server]`로 함께 설치되므로).
top-level 모듈(server·demo)과 패키지(app·common)는 이 폴더가 cwd 면 그대로 잡힌다 —
래퍼 패키지도 --app-dir 도 필요 없다(공통개발자 폴더를 platform 이 아니라 common 으로 둔 이유:
platform 은 stdlib 모듈명과 충돌해 감싸야 했다).
"""

from klafi.server import mount_frontend

from common.bootstrap import auth, build_app

app = build_app().http_app(auth=auth)
mount_frontend(app)
