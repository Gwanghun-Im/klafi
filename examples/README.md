# KLAFI 예제

예제는 **프로젝트 구조**다 (단일 파일 아님). 3-역할 분리(공통개발자=config/bootstrap, 업무개발자=agents)를 따른다.

| 프로젝트 | 내용 |
|----------|------|
| [support_platform/](support_platform/) | **전 기능** — Factory·Engine·Context·Hook + Model·Tool·Skill·Memory·Guardrail·Eval·Registry·Event, CLI + 서버 (에이전트 3종) |

## 실행

```bash
PYTHONPATH=examples python -m support_platform.demo                        # 로컬 데모
uvicorn support_platform.server:app --app-dir examples --port 8078         # HTTP 서비스
```

서버 기동 후 브라우저로 `/docs`(Swagger UI)를 열면 등록된 Agent 3종(`support`·`triage`·`schedule`)을 API로 시험할 수 있다. 호출 시 헤더 `X-User: u1`로 사용자·권한이 주입된다. 에이전트별 curl 예제는 [support_platform/README.md](support_platform/README.md#실행).

## 실제 LLM

`config/model.yaml`의 provider `type`을 `echo`(키 불필요) ↔ `anthropic`/`openai`로 바꾸면 된다. 키는 `support_platform/.env`에 두고(`.env.example` 참고) 서버가 자동 로드한다. 업무 코드(`app/agents/`)는 변경하지 않는다.

구조와 역할 분리는 [support_platform/README.md](support_platform/README.md) 참고.
