# KLAFI Enterprise Agentic AI Framework — 개발계획서 v1.0

> 근거 문서: `KLAFI Enterprise Agentic AI Framework 요구사항 정의서 v1.0`
> 대상: KLAFI 개발팀 / AX사업2팀 · 기준: 2026년 개발 착수
> **상태: 2026-08-24 기준 P0/P1 요구사항 전 영역 구현 완료 (M-A~M-F 통과, 테스트 186개 통과)**

---

## 0. 구현 현황 (2026-08-24)

계획서의 2026 로드맵(P0~P5, M-A~M-F)과 요구사항 문서 F01~F14 + 공통영역(Security/Exception/Event/Config)의 **P0/P1이 전부 구현·검증 완료**되었다.

**마일스톤: M-A ✅ · M-B ✅ · M-C ✅ · M-D ✅ · M-E ✅ · M-F ✅** — 전 게이트 통과.
**규모: 15개 서브패키지, 테스트 186개 전부 통과, 레퍼런스 프로젝트 1개(에이전트 3종, 실제 LLM 연동 검증).**

### 요구사항 → 구현 매핑

| 요구사항 | 구현 모듈 | 상태 |
|------|----------|:----:|
| F01 SDK/BaseGraph · F04 Execution Context | `klafi/core` (graph·base_graph·context·spec·exceptions) — 개발자 진입은 `KlafiGraph` | ✅ |
| F02 Factory · F03 Engine · F07 Policy(Timeout/Retry/Concurrency) | `klafi/runtime` (factory·engine·policy·state) | ✅ |
| F05 State/Checkpoint · Long-Term Memory · Context Manager | `klafi/context` (checkpoint·memory·manager) | ✅ |
| F06 Hook / AOP | `klafi/core/hook` + logging_hook + `klafi/hookdefs` — **Graph·Node·Tool·LLM 4지점 before/after** | ✅ |
| F08 Human-in-the-Loop | `klafi/hitl` (approval) — 승인/검수/질문 3패턴 | ✅ |
| F09 Model Gateway · Tool Framework | `klafi/model`(alias·`init_chat_model`) · `klafi/tool`(권한·검증·Skill) | ✅ |
| F10 Agent Registry (Control Plane) | `klafi/registry` (record·registry) | ✅ |
| F11 Observability (OTel/Trace/Token/Cost) | `klafi/observability` (tracing) | ✅ |
| F12 Evaluation · Guardrail | `klafi/evaluation` · `klafi/guardrail`(`@guardrail`+`@guard` 코드 적용) | ✅ |
| F13 API / Agent Server (invoke/stream/resume) | `klafi/server` (registry·http) + `klafi/app`(부트스트랩) | ✅ |
| F14 Template (Simple/RAG/Supervisor) | `klafi/templates` (KlafiGraph 하위 클래스) | ✅ |
| §21 Security(권한·PII·Audit·Secret 외부화) | `core`(security_context) + `tool`(권한) + `guardrail`(PII) + auth 어댑터 + config `${ENV}` 치환 | ✅ |
| §23 Exception 체계 | `klafi/core/exceptions` — 도메인 축 + 종류 축(NotFound/Validation/Permission/Violation) | ✅ |
| §22 Config Framework (5계층 병합) | `klafi/config` (layered) + `klafi/app/hookplan`(hooks.yaml) | ✅ |
| §24 Event Framework | `klafi/events` (bus·hook) | ✅ |

### 계획 대비 조정 사항 (구현 중 확정)

- **패키지 구조**: 계획의 8개 배포 패키지(§3) 대신 **단일 `klafi` 패키지 + 서브모듈**로 개발. `core`가 어디에도 의존하지 않는 원칙은 **lazy import**로 유지(정책·체크포인트·스토어 미사용 시 해당 모듈 미로딩). 별도 git 분기 시 패키지 분리 예정.
- **SPI 사전 정의**: 8팀 병렬 전제의 "SPI 전부 선정의"(WS1) 대신, **어댑터가 실제로 필요한 시점에 인터페이스 정의**(YAGNI). 결과적으로 Checkpointer=BaseCheckpointSaver, Store=BaseStore 등 **LangGraph 표준을 그대로 채택**해 재구현 0.
- **Guardrail 조기 탑재**: P0 등급인 Input/Output/Violation Logging을 Hook의 fail-close 기반으로 조기 구현(§WS6 주의사항 반영).
- **Docker/API 스캐폴드**: Template의 Docker/API는 Agent Server(F13) 완성 후 채우는 것으로 순서 조정(선행 의존 정직 반영).
- **개발자 진입 API 전환**: `build() -> StateGraph` 반환 패턴을 폐기하고 **`KlafiGraph(BaseGraph)` 상속 + `define()`** 으로 통일. 개발자는 `class MyAgent(KlafiGraph)`에서 `state_schema` + `add_node/add_edge`로 조립. LangGraph 실행엔진(checkpoint/interrupt/streaming)은 계속 내부 활용(재구현 안 함). "Native 감싸기만" 원칙 문구는 폐기.
- **Hook 지점 확장(F06)**: 계획의 Agent/Node에서 **Graph·Node·Tool·LLM 4지점 before/after**로 확대. Node는 `wrap_node`, Tool/Model은 실행 중 활성 Hook(`bind_hooks` ContextVar)을 `Tool.run`/`ModelGateway`가 발화한다.
- **Guardrail 운영모델 전환(F12)**: 초기에는 가드레일을 `config/hooks.yaml`의 `guardrails:` 블록으로 배치했으나, **가드레일을 코드에서 직접 적용하는 방식으로 전환**했다(YAML 배치·이름 레지스트리·`GuardrailNotFoundError` 폐기). `@guardrail`은 함수를 `Guardrail` 객체로 만들고(prebuilt: `pii`/`prompt_injection`), 적용은 세 지점 — ① 노드 `@guard(input=..., output=...)`, ② 워크플로우 전체 `@guard(...)`를 `KlafiGraph` 하위 클래스에, ③ 플랫폼 공통 `GuardrailHook`(hooks 주입). `hooks.yaml`은 명명 훅(`hooks:`)만 담고 `guardrails:` 키가 있으면 기동 시 fail-fast. 훅은 **공통 훅 파일(코드) + YAML(선언) 이중관리** 유지.
- **model_output(응답) 가드레일 추가(F12)**: 기존 `model` stage는 prompt(`before_model`)만 검사하고 응답은 전혀 검사하지 않던 구멍을 발견해 `model_output` stage(`after_model`)를 추가했다. `with_structured_output` 등 tool-calling 방식 structured output은 provider에 따라 `content`가 비고 데이터가 `tool_calls`에만 실리므로, `KlafiCallbackHandler._extract`가 `content`가 비어 있으면 `tool_calls`로 폴백하도록 수정했다(그전엔 provider별로 우연히 되거나 빈 문자열을 검사하는 상태였다). Stream 응답은 `on_llm_end`가 전체 누적 텍스트로 발화해 검사 자체는 되지만, 청크를 실시간으로 외부에 흘려보내는 노드에서는 사후 차단이 이미 나간 내용을 되돌리지 못한다 — 구조적 한계로 남겨둔다.
- **Tool 연결 방식(F09)**: KLAFI 자체 호출 대신 **LangGraph 네이티브 `bind_tools` + `ToolNode`** 채택. KLAFI Tool은 `bind_tools([...])`·`make_tool_node([...])`에 그대로 넣으면 되고(변환은 두 진입점 내부에서 수행), 실행은 `Tool.run` 경유라 권한·검증·Timeout·Hook·Metric이 그대로 적용된다. **노드별로 다른 모델·다른 툴셋**(`init_chat_model(alias)` + `make_tool_node([...])`)도 지원.
- **애플리케이션 계층 신설(계획 외)**: 에이전트 수 증가에 대응해 `klafi/app`(`KlafiApp`)을 추가. Spring Boot 방식의 **3-역할 분리**(프레임워크개발자=`klafi/`, 공통개발자=`config/`+`bootstrap.py`, 업무개발자=`agents/*.py`)를 표준 프로젝트 구조로 확정.
- **예제 정책 변경**: 단일 파일 샘플을 전부 폐기하고 **레퍼런스 프로젝트 1개**(`examples/support_platform/`)로 통합. 기능 하나당 에이전트 하나만 남겨 **3종**(ReAct·노드별 모델/툴 라우팅·Skill)으로 정리했고, 실제 LLM(Claude)과 HTTP(Swagger)로 검증. HITL·실패 후 Resume은 프레임워크와 테스트(`tests/test_hitl_supervisor.py`·`tests/test_policy.py`)에서 계속 검증한다.
- **Config Fail-Fast(§22)**: 잘못된 설정이 조용히 무시되던 문제를 제거. **부트스트랩(`KlafiApp.from_config`) 시점**에 config 디렉터리·environment 미설정, policy 키 오타, hooks.yaml 스키마·미등록 이름, 지원하지 않는 type을 전부 검출한다(각 메시지에 허용 값 목록 포함).
- **Exception 체계 구체화(§23)**: 계획의 단일 계층에서 **도메인 축 × 종류 축**으로 확장(아래 §0.1). 설정 오류(NotFound/Schema)와 실행 중 위반(Violation)이 타입으로 갈린다.
- **chat_model 경로 계측 통합(F09/F11)**: `init_chat_model(alias)`가 돌려주는 LangChain 객체를 통한 호출은 Gateway를 우회해 Token/Cost·훅이 누락되던 문제를 해결. Gateway가 모델 생성 시 `KlafiCallbackHandler`를 주입해 **`bind_tools`·`with_structured_output` 등 파생 Runnable을 통한 호출까지** span·Token/Cost·`before/after_model` 훅·`ModelCalled` 이벤트 대상이 된다. model-stage 가드레일의 fail-close 차단도 이 경로에서 동작한다.
- **PostgreSQL 운영 배선(MEM-03)**: `from_conn_string()`이 context manager를 반환해 실제로는 동작하지 않던 문제를 수정. **ConnectionPool + 스키마 `setup()`** 으로 배선하고, 접속 실패는 5초 내 `CheckpointException`으로 fail-fast(비밀번호 미노출). 실 DB로 프로세스 간 Resume·Long-Term Memory 영속성 검증 완료.
- **Config 환경변수 치환(SEC-05)**: DB Credential을 config 파일에 평문 저장하지 않도록 `${VAR}` / `${VAR:default}` 치환을 추가. 미설정 시 기동 실패. 예제는 `config/environments/postgres.yaml`이 `${KLAFI_PG_DSN}`을 참조하고 값은 `.env`에 둔다(커밋 제외).
- **모델 선언 표준 확정(F03/F09)**: 업무 코드가 모델을 얻는 길을 `init_chat_model("<alias>")` **하나로 통일**. `self.chat_model`(단일 편의 주입)·`self.gateway.chat_model(alias)`(노드별) 두 경로는 제거했다. Gateway는 `ExecutionFactory` 조립 시 프로세스 활성 Gateway로 지정되고, `init_chat_model`은 계측 핸들러가 붙은 `ChatModel` 래퍼(`bind_tools`·`bind_skills` 외 전 속성은 원본 위임)를 돌려준다. 같은 일을 하는 API가 둘이면 프로젝트마다 스타일이 갈리므로 진입점을 하나로 고정했다.
- **Skill 도입(F09 확장)**: **Skill = 툴 묶음 + 사용 지침(prompt)**. 툴만 바인딩하면 "언제 쓰는 툴인지"가 업무 코드의 SystemMessage에 매번 중복 작성된다. `init_chat_model(alias).bind_skills([skill])`은 툴을 `bind_tools`로 붙이고 `skill.prompt`를 SystemMessage로 선행 주입한다. Skill은 값(dataclass)이라 모듈에서 임포트해 쓰며 별도 레지스트리를 두지 않는다(이름으로 참조하는 소비자가 없어 폐기). `bind_tools`에 Skill을 넣으면 지침이 조용히 버려지는 것을 막기 위해 거부한다.
- **동시 실행 상한 노브(F07/§12 확장)**: 계획의 P1 항목 "Concurrency Limit"을 조기 구현. `policy.yaml: concurrency: N` → `ExecutionPolicy.concurrency` → `create_app(max_concurrency=N)`가 서버 전역 `threading.BoundedSemaphore(N)`로 슬롯을 관리한다. invoke/resume(sync 스레드풀)·stream(async) 진입에서 non-blocking acquire, 초과분은 대기 없이 **429 Too Many Requests**(`Retry-After: 1`, 백프레셔)로 즉시 거절하고 슬롯은 요청 종료 시 반납. 미설정 시 무제한. 상한은 uvicorn worker 1개 기준(실질 `N×K`)이고 postgres면 pool `max_size`와 함께 산정한다. memory·postgres 양쪽 + 실서버(동시 6건→200×2/429×4)로 검증.
- **Structured Output**: 에이전트 응답 구조화는 LangChain 네이티브 `chat_model.with_structured_output(Schema)`를 그대로 사용한다(Open Framework). 위 계측 통합으로 관측·가드레일 손실 없이 쓸 수 있다. Tool 출력은 `@tool(output_schema=...)`로 별도 검증.

### 0.1 Exception 체계 (§23 구현)

요구사항의 `KlafiException` 단일 계층을 **두 축**으로 확장했다. 새 타입이 기존 도메인 예외를 상속하므로 `except ToolException` 같은 코드는 그대로 동작한다(하위 호환).

```
KlafiException
├─ 도메인 축: AgentExecution · Model · Tool · Policy · Guardrail · Context · Config · Checkpoint · Approval
└─ 종류  축: NotFoundError · ValidationError · PermissionDeniedError · ViolationError
```

| 상황 | 예외 | error_code | 발생 시점 |
|------|------|-----------|----------|
| config 디렉터리·environment 미설정 | `ConfigNotFoundError` | `CONFIG_NOT_FOUND` | 기동 |
| 설정 키·stage 오타 | `ConfigSchemaError` | `CONFIG_SCHEMA_ERROR` | 기동 |
| 지원하지 않는 설정 값 | `ConfigValueError` | `CONFIG_VALUE_ERROR` | 기동 |
| tool 이름 못찾음 | `ToolNotFoundError` | `TOOL_NOT_FOUND` | 기동/실행 |
| model alias 못찾음 | `ModelNotFoundError` | `MODEL_NOT_FOUND` | 기동 |
| API 키 미설정 | `ModelNotConfiguredError` | `MODEL_NOT_CONFIGURED` | 최초 호출 |
| hook 이름 못찾음 | `HookNotFoundError` | `HOOK_NOT_FOUND` | 기동 |
| agent 미등록(Runtime·Registry) | `AgentNotFoundError` | `AGENT_NOT_FOUND` | 실행 |
| **실행 중 가드레일 차단** | `GuardrailViolationError` | `GUARDRAIL_VIOLATION` | **실행** |
| tool 권한 부족 | `ToolPermissionError` | `TOOL_PERMISSION_DENIED` | 실행 |
| tool 입출력 검증 실패 | `ToolValidationError` | `TOOL_VALIDATION_ERROR` | 실행 |
| 실행 timeout | `TimeoutException` | `TIMEOUT_ERROR` | 실행 |

**핵심 구분**: 설정에 없는 이름 참조는 기동 시 `NotFound`로 실패하고, 실행 중 가드레일 차단은 `GuardrailViolationError`(정상 동작)다. 가드레일은 코드로 참조하므로 "guardrail 이름 못찾음" 예외는 없다. 모든 예외는 `error_code`와 컨텍스트(`stage`·`guard`·`tool`·`execution_id`)를 함께 담아 Trace·Audit과 연결된다.

### 레퍼런스 프로젝트 (`examples/support_platform/`)

3-역할 분리 구조로 프레임워크 전 기능을 검증하는 Pilot. 기능당 에이전트 하나 — 3종:

| Agent | 검증 대상 |
|-------|----------|
| `support` | ReAct (Tool 권한·검증 · Long-Term Memory · Model alias) |
| `triage` | 노드별 다른 모델(`fast`/`expert`) + 다른 툴셋 라우팅 |
| `schedule` | Skill(툴 + 지침) 바인딩 |

전 에이전트가 실제 Claude 연동 및 HTTP(Swagger `/invoke`)로 동작 확인됨. HITL(`/resume`)·실패 후 Checkpoint Resume은 프레임워크와 테스트에서 검증한다. 공통개발자 config(`model.yaml`·`policy.yaml`·`hooks.yaml`)로 모델·정책·명명 훅이 적용되고, 가드레일은 코드(`@guard`/`GuardrailHook`)로 적용된다.

### 남은 것 = 계획서 2027 H1 (P2 고급)

Context Fork/Merge · Online Evaluation · Model Intelligent Router · Distributed Queue/Job · Self-Reflection·자동최적화 · Redis 어댑터 · Grafana Dashboard. 전부 P2로 2026 목표("재사용 프레임워크 뼈대 + Pilot 검증") 범위 밖의 고도화.

---

## 1. 계획 요약 (Executive Summary)

KLAFI는 LangGraph/LangChain **Code-Native 개발 방식을 유지**하면서, Enterprise SI에서 반복되는 실행·통제·관측·평가·운영 기능을 **SDK + Adapter 형태의 재사용 패키지**로 표준화한다.

**2026년 목표는 "완성된 Platform"이 아니라 "재사용 가능한 Framework 뼈대 + Pilot 검증"이다.**

- **2026 (12개월)**: Core → Enterprise Runtime → Observability → Template → Pilot 검증
- **2027 H1**: Registry / Guardrail / Evaluation Platform / HITL / Control Plane 로 Platform 수준 완성

핵심 성공 기준 한 문장: **"LangGraph의 자유도는 유지하고, Enterprise 개발의 반복은 제거한다."**

---

## 2. 설계 원칙 (개발 내내 고정)

| # | 원칙 | 개발 시 강제 규칙 |
|---|------|------------------|
| 1 | LangGraph 엔진 활용 | Checkpoint/Interrupt/StateGraph/Streaming 재구현 금지(내부 활용). 단 개발자 진입은 `KlafiGraph` 상속으로 통일 — "감싸기만" 제약은 없음 |
| 2 | Business Logic 집중 | 개발자는 State/Node/Edge/Prompt/Tool만, 공통기능은 Framework |
| 3 | Open Framework | KLAFI 전용 DSL 강제 금지, LangGraph 직접 접근 항상 허용 |
| 4 | Plug-in / Adapter | Core ↔ 제품 Adapter 분리, 특정 LLM/제품 종속 금지 |
| 5 | Fail-Open/Close 분리 | Observability/Eval Backend 장애가 Agent 업무를 중단시키지 않음 |
| 6 | Framework 후보 판단 | "여러 프로젝트가 같은 코드를 반복하는가?" → YES만 Core에 넣음 |

---

## 3. 패키지 구조 & 소유권

요구사항의 8개 Workstream을 그대로 **8개 배포 패키지**로 확정한다. 각 패키지는 독립 버전·독립 릴리즈.

```
klafi/
├─ klafi-core            # WS1: BaseGraph / AgentSpec / Config / Exception / SPI
├─ klafi-runtime         # WS2: Execution Factory / Engine / Policy
├─ klafi-context         # WS3: State / Checkpoint / Memory / Context Manager
├─ klafi-hook            # WS4: Hook / Decorator / Event
├─ klafi-observability   # WS5: OTel / Logging / Trace / Langfuse·LangSmith Adapter
├─ klafi-intelligence    # WS6: Guardrail / Evaluation
├─ klafi-server          # WS7: API / Agent Server / Registry
└─ klafi-template        # WS8: Agent Template / Sample
```

**의존 규칙**: `core`는 누구에게도 의존하지 않는다(SPI 인터페이스만 제공). 나머지 패키지는 `core`에만 의존하고 서로 직접 의존하지 않으며, 필요한 연결은 `core`의 SPI/Event를 통해서만 한다. → 순환 의존 원천 차단, 패키지별 독립 개발 가능.

---

## 4. 단계별 로드맵 (2026)

전제: 착수 2026-Q1, 2주 스프린트. 아래 기간은 순차가 아니라 **Foundation 이후 병렬 진행**을 전제로 한다.

| Phase | 기간(안) | 목표 | 핵심 산출물 | 게이트(DoD) |
|-------|---------|------|------------|------------|
| **P0. Foundation** | Q1 (M1–M3) | 모든 패키지가 의존할 최소 기반 | BaseGraph, AgentSpec, Config, Exception, SPI 인터페이스, ExecutionContext | Simple LangGraph Agent가 BaseGraph로 실행됨 |
| **P1. Enterprise Runtime** | Q2 (M4–M6) | 실행방식·실행정책 표준화 | Execution Factory/Engine, invoke/async/stream, Timeout/Retry/Cancel, Checkpoint 자동주입, Hook 엔진 | **코드 변경 없이 Config로 실행정책 변경** / Resume 성공 |
| **P2. Observability** | Q2–Q3 (M5–M7) | 실행~Business Exception 단일 Trace 연결 | OTel 적용, Trace/Metric/Log, Langfuse·LangSmith·Loki Adapter | Execution→Agent→Node→Tool→Model→Error 추적 가능 |
| **P3. Template & Toolkit** | Q3 (M7–M9) | 프로젝트 생산성 즉시 효과 | T01 Simple / T02 RAG / T03 Supervisor / T04 Plan-Executor / T05 HITL 템플릿 | 신규 개발자가 Template 복사로 첫 Agent 생성 |
| **P4. Runtime & API** | Q3–Q4 (M8–M10) | Agent를 Enterprise 서비스로 제공 | Agent Server, invoke/stream/resume/cancel API, OpenAPI 자동생성, Health, Auth Adapter | 2개 이상 Agent를 동일 Runtime에서 API 서비스 |
| **P5. Pilot 검증** | Q4 (M10–M12) | Reference Agent로 요구사항 통합검증 | Supervisor Multi-Agent + HITL Reference Agent | 아래 §6 Reference Agent DoD 전 항목 통과 |

> **병렬 원칙**: P0 종료 후 WS2/WS3/WS4/WS5는 동시에 시작한다. Observability(P2)는 Runtime(P1) 완성을 기다리지 않고 Hook 인터페이스 확정 시점부터 착수한다.

---

## 5. Workstream별 실행 계획

각 WS는 **P0 산출물(우선)**, **완료조건**, **P1/P2 확장**을 갖는다.

### WS1 · klafi-core (Framework 기반)
- **P0**: BaseGraph, AgentSpec, 표준 Config 구조(YAML/객체), Agent Metadata(ID/Name/Version/Project), KlafiException 계층, **SPI/Adapter 인터페이스 전부 선정의**(Model/Checkpoint/Store/Trace/Evaluator/Guardrail/Approval).
- **완료조건**: Simple LangGraph Agent를 KLAFI BaseGraph로 실행.
- **주의**: 이 팀이 SPI를 먼저 확정해야 다른 WS가 병렬로 붙는다. **최우선 착수·최우선 동결.**

### WS2 · klafi-runtime (Execution)
- **P0**: Execution Factory(Checkpointer/Model/Store/Hook/Policy 자동 주입), Execution Engine(invoke/ainvoke/stream/astream), Execution ID 발급, Timeout, Retry+Backoff, Trace ID 연계, Execution 상태머신(CREATED→QUEUED→RUNNING→WAITING_APPROVAL→COMPLETED / FAILED·CANCELLED·TIMEOUT).
- **P1**: Batch/abatch, Cancellation, ~~Concurrency Limit~~(완료 — `policy.yaml: concurrency`, 429 백프레셔), Queue/Job, Rate/Token/Cost Policy, Policy Override 체계(Enterprise→Project→Agent→Execution).
- **완료조건**: Agent 코드 변경 없이 Config로 실행정책 변경.

### WS3 · klafi-context (State/Memory/Checkpoint)
- **P0**: ExecutionContext(ContextVar/LangGraph Runtime Context 기반, **Global 변수 금지**), Checkpointer Interface, Memory·PostgreSQL Checkpoint Adapter, Thread ID 표준, Async Context 유지.
- **P1**: Redis Adapter, Resume, Checkpoint 조회, TTL, Long-Term Memory(User/Agent Scope, TTL, 삭제 API, PII 정책), Context Manager(Token 측정, Auto Summarization, 중요 Context 보존, Handoff Summary).
- **P2**: Context Fork/Merge, Advanced Compression, 장기 Task Checklist.
- **완료조건**: 실행 중단 후 동일 Thread Resume 성공.

### WS4 · klafi-hook (Hook/AOP)
- **P0**: Agent/Graph/Node Before·After·Error Hook, Global Hook 등록, Agent별 Hook 등록, Hook 엔진(Decorator Metadata를 Runtime이 읽어 Hook·Policy 연결).
- **P1**: Tool/Model Hook, Finally Hook, Node별 Hook, Hook Priority, Enable/Disable, `@klafi_node`·`@hitl` Decorator.
- **완료조건**: 개발자가 Node 안에 Logging 코드를 쓰지 않아도 Node 실행 로그 자동 생성.

### WS5 · klafi-observability
- **P0**: OpenTelemetry 기본 적용, Agent/Node/Tool/Model Trace, Execution/Error Log, Token·Latency Metric, **필수 Correlation ID**(request/execution/trace/session/agent/thread), Business Exception ↔ Trace 연결.
- **P1**: Model Usage·Cost Metric, Langfuse·LangSmith Adapter, Loki 연계, Grafana Dashboard.
- **완료조건**: Execution ID→Agent→Node→Tool→Model→Error 추적. **Backend 장애 시 Fail-Open(업무 지속).**

### WS6 · klafi-intelligence (Guardrail/Evaluation)
- **P1(주력)**: Evaluator Interface, Rule Evaluator, LLM Judge, Custom Evaluator, Offline Evaluation, 평가결과 저장, Trace-평가 연결 / Input·Output Guardrail(P0), PII Detection, Prompt Injection Guard, Tool Access Guard, Policy Violation Logging(P0), Guardrail Plugin.
- **P2**: Online Evaluation, Agent Version 비교, Dataset 관리.
- **완료조건**: 동일 Agent Version별 품질 비교 가능한 평가결과 구조 확정.
- **주의**: Guardrail의 Input/Output/Policy Violation Logging은 **P0** 등급 → P1로 미루지 말고 Hook(WS4) 위에 조기 탑재.

### WS7 · klafi-server (API/Runtime/Registry)
- **P0**: Agent Invoke API, Stream API, Health API, OpenAPI 자동생성. **Runtime과 HTTP Layer 분리**(FastAPI 교체가 Agent 코드에 영향 없게).
- **P1**: Async Execution/Cancel/Resume/Metadata API, Authentication Adapter, Agent Registry(등록/조회/Version/Owner/상태/Endpoint), Agent Lifecycle(DEVELOPMENT→TEST→APPROVED→PRODUCTION→DEPRECATED→RETIRED).
- **완료조건**: `/agents/{id}/invoke`·`/stream`으로 2개 이상 Agent를 동일 Runtime 서비스.

### WS8 · klafi-template
- **P0/P3**: T01 Simple, T02 RAG, T03 Supervisor, T04 Plan-Executor, T05 HITL, (T06 Batch는 P1).
- **각 Template 필수 포함**: README / Project Structure / Sample Agent / State 정의 / Node 예제 / Config / Checkpoint / Logging / Tracing / Test / Docker / API.
- **완료조건**: 신규 개발자가 Template 복사로 첫 Agent 생성.

---

## 6. Reference Agent (Framework와 동시 개발 — 필수)

Framework만으로는 요구사항 검증이 불가하므로 **Integration Test 역할의 Reference Agent를 병행**한다.

**아키텍처**: Supervisor → (Research / Analysis / Report Agent) → Human Approval

**적용 필수 기능**: Checkpoint · Memory · Streaming · Retry · Timeout · Hook · Observability · Evaluation · Guardrail · HITL · API — **전 항목 실동작.**

이 Reference Agent의 통과가 곧 **2026 P5 게이트**이며, KLAFI Core 1차 릴리즈 조건이다.

> **✅ 달성**: `tests/test_mf_reference_agent.py`(통합 테스트) + 레퍼런스 프로젝트 `examples/support_platform/`(에이전트 3종)로 구현. Supervisor/HITL/Model Gateway/Tool/Skill/Guardrail/Checkpoint/Memory/Observability/Policy/Evaluation을 결합해 통합검증 통과했고, 실제 Claude 및 HTTP(Swagger)로도 동작 확인함. HITL·Resume은 통합 테스트가 담당하고 레퍼런스 프로젝트는 기능당 에이전트 하나 원칙으로 3종만 유지한다.

---

## 7. Definition of Done (릴리즈 게이트) — ✅ 충족

Library 존재만으로 완료로 보지 않는다. 세 관점 전부 충족 (테스트로 검증됨):

**개발자 관점** — Node에 Logging 미작성 / Checkpoint 연결코드 미작성 / Retry·Timeout 미작성 / 모델정보 하드코딩 없음 / Trace 직접전달 없음
**Framework 관점** — 모든 실행에 Execution ID / Agent·Node·Tool·Model 추적 / 공통 Exception 체계(도메인×종류 2축, §0.1) / invoke·async·stream 동일 실행 / Agent별 정책변경 / Checkpoint Resume / 잘못된 설정은 기동 시 Fail-Fast
**운영 관점** — 사용자·Agent Version 실행 추적 / 실패 Node 식별 / 실행 Model·Tool 식별 / Token·Latency·Error 확인

---

## 8. 기술 스택 결정 (초기 확정 대상)

| 영역 | 기본 채택 | Adapter 후보 (교체 가능) |
|------|----------|------------------------|
| Agent Core | LangGraph / LangChain | — (대체 안 함) |
| LLM | 사내 LLM / Azure OpenAI | OpenAI / Claude / vLLM |
| Checkpoint | PostgreSQL | Redis |
| Store / Memory | PostgreSQL | Redis / Vector DB |
| Trace | OpenTelemetry | LangSmith / Langfuse |
| Log | Loki | ELK / APM |
| Evaluation | 자체 Evaluator | LangSmith / Langfuse |
| API Runtime | FastAPI | (HTTP Layer 분리로 교체 가능) |
| Deploy | Kubernetes | — |

**Config 체계(초기 동결 필수)**: `framework.yaml / model.yaml / policy.yaml / observability.yaml / security.yaml / agents/*.yaml`
**우선순위**: Framework Default → Environment → Project → Agent → Runtime Override

**Security 초기 필수(P0)**: Authentication Context, Secret 외부화(코드·Config 저장 금지), 개인정보 Log Masking, Audit Log.

---

## 9. 마일스톤 요약

| 마일스톤 | 시점(안) | 조건 | 상태 |
|---------|---------|------|:----:|
| **M-A** Foundation Freeze | Q1 말 | SPI 인터페이스 동결 + BaseGraph 실행 | ✅ |
| **M-B** Runtime Ready | Q2 말 | Config로 정책변경 + Resume | ✅ |
| **M-C** Observability Ready | Q3 초 | 단일 Trace 연결 | ✅ |
| **M-D** Template Ready | Q3 말 | Template 복사로 첫 Agent | ✅ |
| **M-E** API Serving | Q4 초 | 2개+ Agent 동일 Runtime 서비스 | ✅ |
| **M-F** Pilot Pass (Core 1차 릴리즈) | Q4 말 | Reference Agent DoD 전 통과 | ✅ |

---

## 10. 리스크 & 대응

| 리스크 | 영향 | 대응 |
|--------|------|------|
| SPI 인터페이스 늦은 확정 | 전 WS 병렬화 실패 → 일정 붕괴 | WS1을 Q1 최우선, M-A에서 인터페이스 **동결**, 이후 변경은 RFC |
| LangGraph 과도한 래핑 | Open Framework 원칙 위반 | 코드리뷰 체크리스트에 "Native 접근 가능?" 항목 강제 |
| Observability/Eval Backend 종속·장애 | Agent 전체 장애로 확대 | Adapter 분리 + Component별 Fail-Open/Close 정책 명문화 |
| Framework Overhead | 성능 NFR 미달 | 벤치마크를 M-B/M-F 게이트에 포함, Connection Pool 재사용 |
| Reference Agent 지연 | 통합검증 공백 | Framework와 **동시 착수**, P0부터 최소 버전 병행 |
| Business Logic이 Core에 유입 | 재사용성 훼손 | §2 원칙6 판단기준을 PR 템플릿에 삽입 |

---

## 11. 2027 H1 확장 (참고)

Registry · Advanced Policy Engine · HITL 고도화 · Evaluation Platform · Guardrail 강화 · Queue/Job · Advanced Context(Fork/Merge/Compression) · Agent Governance · Deployment Automation · **Control Plane** → Enterprise Agent Engineering Framework 수준 완성.
```

---

**부록: 우선순위 매핑** — P0(Core, 2026 필수): SDK/Execution/Factory/Context/Hook/Policy(Retry·Timeout)/State·Checkpoint/Model Adapter/Observability/Exception/Config/API/Template · P1(Enterprise): Memory/HITL/Registry/Cost·Concurrency/Tool Registry/Evaluation/Guardrail/Langfuse·LangSmith/Agent Server/RBAC·Audit · P2(고급): Context Fork·Merge/Online Eval/Model Router/Distributed Queue/Approval Workflow/Self Reflection·자동최적화.
