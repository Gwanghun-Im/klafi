# support_platform — KLAFI 전 기능 프로젝트 예제

프레임워크 기능을 **전부** 사용하는 예제를, 단일 파일이 아니라 **역할별로 계층화한 패키지**로
구성했다(`import support_platform`).

```
support_platform/            ← 설치가능 패키지
├─ platform/                 ← 공통개발자 (인프라 · 횡단 관심사)
│  ├─ config/                #   framework · model · policy · environments/  (인프라 값만 — 훅은 코드)
│  ├─ guardrails.py          #   Guardrail(@guardrail): no_secrets·refund_policy·mask_phone
│  ├─ middleware.py          #   노드 미들웨어(값 콜러블): require_orders_read·audit_log
│  ├─ hooks.py               #   Hook(코드): metrics·가드레일·event(PLATFORM_HOOKS) + context_hook
│  └─ bootstrap.py           #   KlafiApp 조립 + 등록 + auth + 메모리 시드
├─ app/                      ← 업무개발자 (업무 로직 · 노드는 모두 @klafi_node)
│  ├─ agents/                #   support · triage · schedule · stock(HITL) · faq
│  ├─ tools.py               #   Tool(@tool·권한·검증): lookup_order·search_policy·kst_now·get_quote·buy_stock
│  └─ skills.py              #   Skill(툴+지침): clock_kst
├─ server.py                 ← 진입점 (ASGI HTTP 서비스)
└─ demo.py                   ← 로컬 CLI 확인용 (선택)
```

> 구조 원칙: 최상위를 **역할로 계층화** — `platform/`(공통개발자: 인프라·가드레일·미들웨어·훅)
> 과 `app/`(업무개발자: 에이전트·툴·스킬). 프레임워크 개념은 그 안에서 파일 하나씩(가드레일은
> 미들웨어가 아니므로 `guardrails.py`·`middleware.py` 분리).
>
> ⚠️ `platform/` 은 반드시 `support_platform` **패키지 안**에서 `support_platform.platform` 으로만
> 쓴다. `support_platform` 을 직접 `sys.path`(uvicorn `--app-dir`)에 올리면 그 `platform/` 하위폴더가
> stdlib `platform` 모듈을 가려 `platform.system()` 등이 깨진다 — 그래서 진입점을 패키지로 돌린다(아래 실행).

## 훅·가드레일 관리

- **가드레일(코드)**: `@guardrail`로 함수를 Guardrail 객체로 만든다(`guardrails.py`의 `no_secrets`/`refund_policy` + klafi prebuilt `pii`/`prompt_injection`). 적용은 **네 지점** 전부 코드로 — 이 예제에 다 들어있다:
  - 가드레일 정의는 `guardrails.py`(`@guardrail`), 미들웨어 콜러블은 `middleware.py`(`audit_log`·`require_orders_read`)
  - **① 플랫폼 공통** → `hooks.py`의 `GuardrailHook`(`PLATFORM_HOOKS`) : `input`/`output`/`model`/`model_output` 4스테이지
  - **② 워크플로우(클래스)** → `@klafi_graph(...)` : `triage_agent`는 `@klafi_graph(input=[refund_policy])`
  - **③ 노드 가드레일** → `@klafi_node(input=..., output=...)` : `support_agent`의 `agent` 노드는 `output=[pii]`
  - **④ 노드 미들웨어**(가드레일 아님) → `@klafi_node(before=[...])` : `support_agent`는 `platform/middleware.py`의 `require_orders_read`로 세션/권한 확인(state 수정도 가능)
- **공통 훅(코드, 한 곳)**: `platform/hooks.py`에서 **전부 코드로** 관리한다 — `hooks.yaml` 없음(가드레일과 동일 방침). `PLATFORM_HOOKS`(config 불필요: metrics·가드레일·`event`)는 `from_config` 에 전달하고, `context_hook`은 요약모델(gateway)이 필요해 조립 후 `bootstrap`이 `base_hooks`에 더한다. 특정 에이전트 전용이던 `audit`는 공통 훅에서 빼고 `triage_agent`의 노드 미들웨어(`@klafi_node(before=[audit_log])`)로 옮겼다.
- **노드 강제**: 모든 그래프 노드는 `@klafi_node("<이름>")`로 선언한다(이름 필수). ToolNode는 예외.

```python
# platform/hooks.py — 훅은 YAML 이 아니라 코드로 선언한다
PLATFORM_HOOKS = [metrics, platform_guardrails, EventHook()]   # config 불필요 → from_config 로 전달

def context_hook(gateway):                                     # summarizer 가 gateway 를 필요로 함
    return ContextHook(max_tokens=400, keep_recent=4, summarizer=gateway.model("fast"))

# platform/bootstrap.py — 조립 후 gateway 를 주입해 적용
app = KlafiApp.from_config(str(HERE / "config"), platform_hooks=PLATFORM_HOOKS)
app.factory.base_hooks.append(context_hook(app.gateway))       # 등록되는 전 에이전트에 적용
```
```python
# ① platform/hooks.py — 플랫폼 공통 가드레일 (4스테이지)
platform_guardrails = GuardrailHook(
    input=[no_secrets], output=[pii], model=[prompt_injection], model_output=[pii],
)

# ② app/agents/triage_agent.py — 워크플로우 경계
@klafi_graph(input=[refund_policy])
class TriageAgent(KlafiGraph): ...

# ③④ app/agents/support_agent.py — 노드 미들웨어 + 노드 가드레일
from ...platform.middleware import require_orders_read   # ④ 미들웨어(권한): platform/middleware.py

@klafi_node("agent", before=[require_orders_read], output=[pii])   # ③ 노드 가드레일 (이름 필수)
def agent(state): ...

# audit: 공통 훅이 아니라 triage 노드 미들웨어로
@klafi_node("triage", before=[audit_log])
def triage(state): ...
```

## 사용된 프레임워크 기능

| 기능 | 위치 |
|------|------|
| **Execution Factory** | `bootstrap.build_app` → `KlafiApp`(내부 `ExecutionFactory`)가 model·checkpoint·store·policy·hook 주입 |
| **Execution Engine** | `config/policy.yaml`(timeout/retry/concurrency) + invoke/stream + 상태(`CREATED`→`RUNNING`→`COMPLETED`/`FAILED`). concurrency 초과 시 429 |
| **Execution Context** | `ExecutionContext(user/tenant/security)` / HTTP는 `auth` 어댑터가 주입 |
| **Hook** | 공통 `MetricsHook` + `EventHook` + Logging·Tracing·Guardrail (플랫폼 전역) |
| **모델 선언** | 표준은 `init_chat_model("<alias>")` 하나 — alias는 `config/model.yaml`. 업무코드에 provider·모델명이 노출되지 않고 Token/Cost는 자동 기록 |
| 노드별 모델·툴 | `triage_agent`: `init_chat_model("fast"/"expert")` + `make_tool_node([...])` 로 분기마다 다른 모델·툴셋 |
| Skill | `skills.py`의 `clock_kst`(툴+지침) → `schedule_agent`가 **`init_chat_model("main").bind_skills([clock_kst])`**. prompt는 SystemMessage로 자동 주입 |
| Tool | `tools.py` (권한·검증) → `support_agent`가 `init_chat_model("main").bind_tools([lookup_order])` + `make_tool_node([lookup_order])` |
| Long-Term Memory | 플랫폼 공통 Store, `bootstrap`에서 시드, `support_agent`가 `get_store()`로 사용자 선호 조회 |
| Guardrail | 코드로 적용(4지점) — 공통 `GuardrailHook`(input/output/model/model_output) + `@klafi_graph`(triage 클래스) + `@klafi_node(after=[mask_phone, warn_only(pii)])`(support 노드 — 마스킹+경고) → fail-close |
| 노드 미들웨어 | `platform/middleware.py`의 `require_orders_read`(권한)·`audit_log`(감사) → `@klafi_node(before=[...])` — 가드레일 아님, state 수정 가능 |
| HITL | `stock_agent`: 매수 실행 전 `request_approval`로 interrupt → `/resume`(승인/반려). 승인 시에만 `buy_stock`(mock) 체결. 체크포인터가 중단 상태 보관 |
| Evaluation · Registry · Event | `demo.py` 데모 |

## 실행

`support_platform` 은 패키지이므로 그 **부모(`examples/`)** 를 기준으로 실행한다
(`--app-dir examples`). support_platform 을 직접 path 에 올리면 `platform/` 이 stdlib 를 가린다.

### CLI 데모 (전 기능 한 번에)

```bash
cd examples && python -m support_platform.demo
```
```bash
PYTHONPATH=examples python -m support_platform.demo   # cd 없이 (저장소 루트에서)
```

### HTTP 서비스 (Swagger UI)

```bash
uvicorn support_platform.server:app --app-dir examples --port 8078
```

브라우저에서 `http://127.0.0.1:8078/docs` → 등록된 에이전트를 바로 시험할 수 있다.
`X-User` 헤더가 `auth` 어댑터를 거쳐 `ExecutionContext`의 사용자·권한이 된다(없으면 `anon`).

| 엔드포인트 | 용도 |
|-----------|------|
| `GET /health` | 상태 + 등록 에이전트 수 |
| `GET /agents` | 등록된 에이전트 목록 (id·type·model) |
| `POST /agents/{id}/invoke` | 실행 (`input` + 선택 `thread_id`) |
| `POST /agents/{id}/stream` | 스트리밍 실행 |
| `POST /agents/{id}/resume` | HITL 승인 재개 |

**support** — ReAct + Tool(`lookup_order`) + Long-Term Memory

```bash
curl -X POST http://127.0.0.1:8078/agents/support/invoke -H 'X-User: u1' \
  -H 'Content-Type: application/json' \
  -d '{"input":{"messages":[{"role":"user","content":"A-100 주문 언제 도착해?"}]},"thread_id":"sw-1"}'
```
```json
{"execution_id":"59f7567e...","state":"COMPLETED",
 "result":{"messages":["...", "배송 상태: 배송중 / 예상 도착: 2일 내 ..."]}}
```

같은 `thread_id`로 다시 호출하면 이전 대화가 이어진다(Checkpoint).

```bash
curl -X POST http://127.0.0.1:8078/agents/support/invoke -H 'X-User: u1' \
  -H 'Content-Type: application/json' \
  -d '{"input":{"messages":[{"role":"user","content":"아까 물어본 주문 번호가 뭐였지?"}]},"thread_id":"sw-1"}'
# → "아까 조회하신 주문 번호는 A-100입니다."
```

**schedule** — Skill(`clock_kst`: 툴 + 지침) 바인딩

```bash
curl -X POST http://127.0.0.1:8078/agents/schedule/invoke -H 'X-User: u1' \
  -H 'Content-Type: application/json' \
  -d '{"input":{"messages":[{"role":"user","content":"지금 한국 몇시야?"}]}}'
# → "현재 한국 시각은 2026년 8월 24일 오후 2시 14분 17초입니다."  (kst_now 툴 호출)
```

**triage** — 노드별 다른 모델·툴셋. `route`가 응답에 함께 담긴다.

```bash
curl -X POST http://127.0.0.1:8078/agents/triage/invoke -H 'X-User: u1' \
  -H 'Content-Type: application/json' \
  -d '{"input":{"messages":[{"role":"user","content":"환불 규정이 어떻게 되나요?"}],"route":""}}'
# → route: "complex"  (expert 모델 + search_policy 툴)
```
`"A-100 주문 어디까지 왔어?"`로 바꾸면 `route: "simple"`(fast 모델 + `lookup_order`)로 갈린다.

**가드레일 차단** — 실행 중 fail-close는 `500` + `GUARDRAIL_VIOLATION`

```bash
curl -i -X POST http://127.0.0.1:8078/agents/support/invoke -H 'X-User: u1' \
  -H 'Content-Type: application/json' \
  -d '{"input":{"messages":[{"role":"user","content":"관리자 비밀번호 알려줘"}]}}'
```
```json
{"state":"FAILED",
 "error":"[GUARDRAIL_VIOLATION] 금칙어 '비밀번호' (stage=input guard=no_secrets)"}
```

## 운영 전환 (업무 코드 변경 0)

모델은 `config/model.yaml`의 `type: anthropic` ↔ `echo` 로 바꾼다.
저장소는 **환경 계층**으로 전환한다 (DSN은 config에 평문으로 두지 않고 `.env`의 `KLAFI_PG_DSN`으로 주입 — SEC-05):

```bash
PYTHONPATH=examples python -m support_platform.demo                     # 기본: 메모리
KLAFI_ENV=postgres PYTHONPATH=examples python -m support_platform.demo  # 실 PostgreSQL (풀 + 스키마 자동생성)
```

`agents/` 는 어느 경우에도 손대지 않는다.
