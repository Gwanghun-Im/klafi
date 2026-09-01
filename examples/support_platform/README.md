# support_platform — KLAFI 전 기능 프로젝트 예제

프레임워크 기능을 **전부** 사용하는 예제를, 단일 파일이 아니라 **역할별 flat 프로젝트**로 구성했다.
이 폴더째 복사·zip 해 klafi 만 `pip install` 하면 그대로 돈다.

```
support_platform/            ← 프로젝트 (flat — 이 폴더 안에서 실행)
├─ common/                   ← 공통개발자 (인프라 · 횡단 관심사)
│  ├─ config/                #   framework · model · policy · mcp · environments/  (인프라 값만 — 훅은 코드)
│  ├─ guardrails.py          #   Guardrail(@guardrail): no_secrets·refund_policy·mask_phone
│  ├─ middleware.py          #   노드 미들웨어(값 콜러블): require_orders_read·audit_log
│  ├─ hooks.py               #   Hook(코드): metrics·가드레일·event(PLATFORM_HOOKS) + context_hook
│  ├─ mcp.py                 #   MCP 외부 도구 연결(connect_mcp) — 에이전트가 import 해 bind. 없으면 degrade
│  └─ bootstrap.py           #   KlafiApp 조립 + 등록 + auth + 메모리 시드
├─ app/                      ← 업무개발자 (업무 로직 · 노드는 모두 @klafi_node)
│  ├─ agents/                #   에이전트별 패키지: support · triage · schedule · stock(HITL) · faq
│  │  └─ <name>/             #     agentSpec.py(스펙) · state.py(상태) · prompt.py(프롬프트) · config.yaml(per-agent policy) · <name>_agent.py(그래프)
│  ├─ tools.py               #   Tool(@tool·권한·검증): lookup_order·search_policy·kst_now·get_quote·buy_stock
│  └─ skills.py              #   Skill(툴+지침): clock_kst
├─ server.py                 ← 진입점 (ASGI HTTP 서비스)
└─ demo.py                   ← 로컬 CLI 확인용 (선택)
```

> 구조 원칙: 역할 분리 — `common/`(공통개발자: 인프라·가드레일·미들웨어·훅)과 `app/`(업무개발자:
> 에이전트·툴·스킬). 둘 다 top-level 패키지, `server.py`·`demo.py`는 top-level 모듈이라 **이 폴더가
> cwd 면 그대로 잡힌다** — 래퍼 패키지도 `--app-dir` 도 없다.
>
> 공통개발자 폴더를 `platform`이 아니라 **`common`** 으로 둔 이유: `platform`은 파이썬 stdlib 모듈명과
> 충돌해, 그대로 두면 전체를 별도 패키지로 감싸야 했다(`support_platform.platform`). `common`으로 바꿔
> flat 구조가 됐다.

## 훅·가드레일 관리

- **가드레일(코드)**: `@guardrail`로 함수를 Guardrail 객체로 만든다(`guardrails.py`의 `no_secrets`/`refund_policy` + klafi prebuilt `pii`/`prompt_injection`). 적용은 **네 지점** 전부 코드로 — 이 예제에 다 들어있다:
  - 가드레일 정의는 `guardrails.py`(`@guardrail`), 미들웨어 콜러블은 `middleware.py`(`audit_log`·`require_orders_read`)
  - **① 플랫폼 공통** → `hooks.py`의 `GuardrailHook`(`PLATFORM_HOOKS`) : `input`/`output`/`model`/`model_output` 4스테이지. `input`·`output` 경계도 값 스레딩(_transform)이라 **마스킹까지 적용**된다 — 이 예제는 출력 전화번호 마스킹(`mask_phone`)을 여기서 전 에이전트에 일괄 적용
  - **② 워크플로우(클래스)** → `@klafi_graph(...)` : `triage_agent`는 `@klafi_graph(input=[refund_policy])`
  - **③ 노드 가드레일** → `@klafi_node(input=..., output=...)` : 노드 단위로 붙이는 지점(스트리밍에서도 동작). 이 예제는 출력 마스킹을 ①(플랫폼)으로 올려 노드엔 두지 않았다 — 특정 노드만 다르게 검사할 때 쓴다
  - **④ 노드 미들웨어**(가드레일 아님) → `@klafi_node(before=[...])` : `support_agent`는 `common/middleware.py`의 `require_orders_read`로 세션/권한 확인(state 수정도 가능)
- **공통 훅(코드, 한 곳)**: `common/hooks.py`에서 **전부 코드로** 관리한다 — `hooks.yaml` 없음(가드레일과 동일 방침). `PLATFORM_HOOKS`(config 불필요: metrics·가드레일·`event`)는 `from_config` 에 전달하고, `context_hook`은 요약모델(gateway)이 필요해 조립 후 `bootstrap`이 `base_hooks`에 더한다. 특정 에이전트 전용이던 `audit`는 공통 훅에서 빼고 `triage_agent`의 노드 미들웨어(`@klafi_node(before=[audit_log])`)로 옮겼다.
- **노드 강제**: 모든 그래프 노드는 `@klafi_node("<이름>")`로 선언한다(이름 필수). ToolNode는 예외.

```python
# common/hooks.py — 훅은 YAML 이 아니라 코드로 선언한다
PLATFORM_HOOKS = [metrics, platform_guardrails, EventHook()]   # config 불필요 → from_config 로 전달

def context_hook(gateway):                                     # summarizer 가 gateway 를 필요로 함
    return ContextHook(max_tokens=400, keep_recent=4, summarizer=gateway.model("fast"))

# common/bootstrap.py — 조립 후 gateway 를 주입해 적용
app = KlafiApp.from_config(str(HERE / "config"), platform_hooks=PLATFORM_HOOKS)
app.factory.base_hooks.append(context_hook(app.gateway))       # 등록되는 전 에이전트에 적용
```
```python
# ① common/hooks.py — 플랫폼 공통 가드레일 (4스테이지, output 은 마스킹까지)
platform_guardrails = GuardrailHook(
    input=[no_secrets], output=[mask_phone, warn_only(pii)],   # 전 에이전트 출력에 전화번호 마스킹
    model=[prompt_injection], model_output=[warn_only(pii)],
)

# ② app/agents/triage/triage_agent.py — 워크플로우 경계
@klafi_graph(input=[refund_policy])
class TriageAgent(KlafiGraph): ...

# ④ app/agents/support/support_agent.py — 노드 미들웨어(권한). 출력 마스킹은 ①(플랫폼)으로 올림
from common.middleware import require_orders_read

@klafi_node("agent", before=[require_orders_read])   # 이름 필수
def agent(state): ...

# audit: 공통 훅이 아니라 triage 노드 미들웨어로
@klafi_node("triage", before=[audit_log])
def triage(state): ...
```

## 사용된 프레임워크 기능

| 기능 | 위치 |
|------|------|
| **Execution Factory** | `bootstrap.build_app` → `KlafiApp`(내부 `ExecutionFactory`)가 model·checkpoint·store·policy·hook 주입 |
| **자동 등록(convention)** | `app.register_package("app.agents")` — `app/agents/<name>/` 폴더를 훑어 자동 등록. 업무개발자는 폴더만 떨구면 서비스됨(bootstrap 무수정). owner 는 `agentSpec.py` 의 `spec.owner`, `_` 접두 폴더는 스킵 |
| **Execution Engine** | `config/policy.yaml`(timeout/retry/concurrency) + invoke/stream + 상태(`CREATED`→`RUNNING`→`COMPLETED`/`FAILED`). concurrency 초과 시 429 |
| **에이전트별 policy** | `app/agents/<name>/config.yaml`의 `policy:` 가 전역 위를 **명시 키만 덮음**(나머지 상속). 예: `stock`은 timeout 120·retry 0·concurrency 1, `triage`는 concurrency 3. **동시성은 2단계** — 요청은 전역 총량 캡(policy.yaml) AND 에이전트별 캡을 둘 다 통과해야 실행(둘 중 하나 초과 시 429) |
| **Execution Context** | `ExecutionContext(user/tenant/security)` / HTTP는 `auth` 어댑터가 주입 |
| **Hook** | 공통 `MetricsHook` + `EventHook` + Logging·Tracing·Guardrail (플랫폼 전역) |
| **모델 선언** | 표준은 `init_chat_model("<alias>")` 하나 — alias는 `config/model.yaml`. 업무코드에 provider·모델명이 노출되지 않고 Token/Cost는 자동 기록 |
| 노드별 모델·툴 | `triage_agent`: `init_chat_model("fast"/"expert")` + `make_tool_node([...])` 로 분기마다 다른 모델·툴셋 |
| Skill | `skills.py`의 `clock_kst`(툴+지침) → `schedule_agent`가 **`init_chat_model("main").bind_skills([clock_kst])`**. prompt는 SystemMessage로 자동 주입 |
| Tool | `tools.py` (권한·검증) → `support_agent`가 `init_chat_model("main").bind_tools([lookup_order])` + `make_tool_node([lookup_order])` |
| MCP(외부 도구) | `config/mcp.yaml`(Tavily 웹검색 서버, `permission: web:search`, 키는 `${TAVILY_API_KEY}`로 .env 주입) → `common/mcp.py`의 `connect_mcp`가 `from_langchain_tool`로 감싸 KLAFI Tool화 → `support_agent`가 `from common.mcp import search` 후 `bind_tools([lookup_order, *search])`. MCP 도구도 **권한(web:search)·검증·audit·가드레일**을 그대로 탄다. `pip install 'klafi[mcp]'`+npx+키 필요, 없으면 자동 degrade |
| Long-Term Memory | 플랫폼 공통 Store, `bootstrap`에서 시드, `support_agent`가 `get_store()`로 사용자 선호 조회 |
| Guardrail | 코드로 적용 — 공통 `GuardrailHook`(input/output/model/model_output; **output 은 값 스레딩이라 `mask_phone` 마스킹까지 전 에이전트 적용**) + `@klafi_graph`(triage 클래스) + `@klafi_node`(노드 단위) → 차단은 fail-close, 전화번호는 MASK |
| 노드 미들웨어 | `common/middleware.py`의 `require_orders_read`(권한)·`audit_log`(감사) → `@klafi_node(before=[...])` — 가드레일 아님, state 수정 가능 |
| HITL | `stock_agent`: 매수 실행 전 `request_approval`로 interrupt → `/resume`(승인/반려). 승인 시에만 `buy_stock`(mock) 체결. 체크포인터가 중단 상태 보관 |
| Evaluation · Registry · Event | `demo.py` 데모 |

## 실행

**이 폴더 안에서** 실행한다 — 래퍼 패키지도 `--app-dir` 도 없다(폴더째 복사·zip 해도 동일).

### CLI 데모 (전 기능 한 번에)

```bash
cd examples/support_platform && python demo.py
```

### HTTP 서비스 (Swagger UI)

```bash
cd examples/support_platform && uvicorn server:app --port 8078
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
python demo.py                     # 기본: 메모리 (examples/support_platform 안에서)
KLAFI_ENV=postgres python demo.py  # 실 PostgreSQL (풀 + 스키마 자동생성)
```

`agents/` 는 어느 경우에도 손대지 않는다.
