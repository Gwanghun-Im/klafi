# KLAFI 아키텍처

> 근거 요구사항: `요구사항 정의서 v1.0` §4(목표 아키텍처). 이 문서는 실제 구현 기준.

---

## 1. 레이어

```
┌─────────────────────────────────────────────────────────┐
│ Enterprise Agent Application                            │
│  Simple / RAG / Supervisor / Reference (개발자 코드)     │
└───────────────────────────┬─────────────────────────────┘
                            │  class MyAgent(KlafiGraph): define()
┌───────────────────────────▼─────────────────────────────┐
│ KLAFI SDK           klafi/core, klafi/templates          │
│  KlafiGraph · BaseGraph · ExecutionContext · Hook 엔진    │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│ KLAFI Execution      klafi/runtime                       │
│  Execution Engine (invoke/async/stream) · Policy · State │
│  Hook onion(Before/After/Error/Finally) · Event 발행      │
└──────────────┬────────────────────────┬──────────────────┘
               │                        │
┌──────────────▼─────────┐   ┌──────────▼───────────────────┐
│ Context/State          │   │ Intelligence                 │
│ klafi/context          │   │ klafi/guardrail·evaluation    │
│  Checkpoint·Memory·CtxM │   │  Guardrail(fail-close)·Eval  │
└──────────────┬─────────┘   └──────────┬───────────────────┘
               │                        │
┌──────────────▼────────────────────────▼──────────────────┐
│ LangGraph (재구현 안 함)                                   │
│  StateGraph · Checkpoint · Interrupt · Streaming · Store  │
└──────────────────────────┬────────────────────────────────┘
                           │  Adapter Layer
┌──────────────────────────▼────────────────────────────────┐
│ Adapters   model gateway · checkpointer · store · tracing  │
│            · approval · evaluator · guardrail · event      │
└──────────────────────────┬────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────┐
│ Platform  LLM · Postgres · Redis · OTel/Loki/Langfuse · K8s│
└────────────────────────────────────────────────────────────┘
```

주변 서비스: **klafi/server**(FastAPI 어댑터) · **klafi/registry**(Control Plane) · **klafi/config**(5계층 설정).

---

## 1.5 개발자 진입점 (KlafiGraph 상속)

업무개발자는 `KlafiGraph`를 상속해 Agent를 만든다. KlafiGraph는 BaseGraph(실행)를 상속하고 LangGraph StateGraph 빌더 API를 노출한다.

```
   개발자 코드      ┌───────────────────────────────────────────┐
                    │ class MyAgent(KlafiGraph):                 │
                    │   state_schema = State                     │
                    │   def define(self):                        │
                    │     self.add_node(...); self.add_edge(...) │
                    └───────────────────┬───────────────────────┘
   KlafiGraph       ┌───────────────────▼───────────────────────┐
                    │ StateGraph 빌더(add_node/add_edge) 위임     │
                    │ + LoggingHook/TracingHook 기본 탑재         │
                    └───────────────────┬───────────────────────┘
   BaseGraph        ┌───────────────────▼───────────────────────┐
   (실행 기반)       │ invoke/stream · Hook · 정책 · Context ·      │
                    │ Checkpoint · Store 배선                     │
                    └────────────────────────────────────────────┘
```

`define()` 안에서 업무개발자가 쓰는 API는 넷뿐이다.

```python
llm = init_chat_model("main")           # 모델 선언 (alias는 config/model.yaml)
llm = llm.bind_tools([lookup_order])    # 툴 (KLAFI Tool 그대로)
llm = llm.bind_skills([clock_kst])      # 툴 + 지침 (prompt는 SystemMessage로 주입)
self.add_node("tools", self.make_tool_node([lookup_order]))   # ToolNode
```

- **Template**(`SimpleAgent`/`RAGAgent`/`SupervisorAgent`)은 KlafiGraph 하위 클래스로, `define()`이 정해진 흐름을 갖는다.
- **저수준 제어**가 필요하면 `BaseGraph`를 직접 상속(`build() -> StateGraph`) — Hook 기본 탑재 없이 완전 수동.
- `.builder`(원본 StateGraph)/`.compiled`로 LangGraph에 직접 접근 가능(Open Framework). Business Logic(State/Node/Edge/Prompt/Tool)만 다르고 Enterprise 배선은 동일.

---

## 1.6 애플리케이션 계층 & 3-역할 분리 (`klafi/app`)

Agent가 많아지면 인프라 설정(모델·정책·보안)을 Agent마다 반복하면 안 된다. `KlafiApp`이 **공통 설정을 config 한곳에서 조립**해 각 Agent에 주입한다 — Spring Boot의 `application.yml` + `@Configuration` Bean에 해당.

```
config/*.yaml ──┐
                │ KlafiApp.from_config()
                ▼
   ┌─────────────────────────────────────────────┐
   │ 공유 리소스 (공통개발자 소유)                   │
   │  ModelGateway(model.yaml) · ExecutionPolicy   │
   │  (policy.yaml) · GuardrailHook(코드/공통 훅)   │
   │  · Logging/Tracing · checkpoint/store(framework)│
   └───────────────────────┬─────────────────────┘
                           │ app.create(spec, build) — 주입
   agents/*.py (업무개발자) │  class XAgent(KlafiGraph)
                           ▼
   ┌─────────────────────────────────────────────┐
   │ KlafiGraph 인스턴스 : 업무 그래프 + 공유 인프라 │
   └───────────────────────┬─────────────────────┘
              app.register()│
                           ▼  Registry(governance) + AgentServer(runtime) + http_app()(FastAPI)
```

**역할 경계**

| 역할 | 소유 | 코드/설정 |
|------|------|-----------|
| 프레임워크 개발자 | `klafi/` | SDK·Adapter·실행 파이프라인 |
| 공통개발자 | `config/*.yaml` + `main.py` | 모델 연결·정책·Guardrail·Checkpoint·서비스 부트스트랩 |
| 업무개발자 | `agents/*.py` | `KlafiGraph` 하위 클래스 (`spec`+`state_schema`+`define()`) |

업무개발자의 Agent 파일에는 gateway/tracing/checkpointer/guardrail 코드가 없다. 모델은 alias(`spec.model="main"`)로만 참조하고, 실제 provider·키·정책·보안은 config가 결정한다. → 운영 전환(echo→anthropic, memory→postgres)이 **업무 코드 변경 없이 config만**으로 이뤄진다.

---

## 2. 의존 규칙 (순환 차단)

**`klafi/core`는 어디에도 하드 의존하지 않는다.** 상위 기능(runtime/context/observability 등)은 core에만 의존한다.

- BaseGraph가 정책·체크포인트·스토어를 쓸 때는 **lazy import**로 해당 모듈을 불러온다. 미사용 시 그 모듈은 로드조차 되지 않아 core의 독립성이 유지된다.

  ```python
  def _run_sync(self, fn, ctx, policy):
      if policy is None:
          return fn()                          # runtime 미로딩
      from klafi.runtime.engine import run_sync # 필요할 때만
      return run_sync(fn, policy, ...)
  ```

- `klafi/server`(FastAPI)는 top-level `klafi/__init__`에서 import하지 않는다 → core 사용자는 FastAPI 없이 동작.

---

## 3. 실행 흐름 (`agent.invoke`)

```
invoke(input, thread_id)
 │
 ├─ ExecutionContext 생성/보강 (execution_id, agent_id from spec)
 ├─ with bind_context(ctx)          ← ContextVar 바인딩 (async/thread 안전)
 │   ├─ Hook.before_agent           (Guardrail input, Event: AgentStarted …)
 │   ├─ Policy 적용 (있으면)          run_sync/run_async: retry가 timeout을 감쌈
 │   │    └─ compiled.invoke(...)    ← LangGraph 실행
 │   │         └─ 각 Node = wrap_node(원본)
 │   │              ├─ Hook.before_node   (Logging/Tracing span 시작)
 │   │              ├─ 원본 Node 실행       (span("tool.x")/model.x 자동 중첩)
 │   │              ├─ Hook.after_node
 │   │              └─ Hook.finally_node   (span 종료)
 │   ├─ interrupt 시 → state=WAITING_APPROVAL (결과에 __interrupt__)
 │   ├─ Hook.after_agent            (Guardrail output, Event: AgentCompleted)
 │   ├─ Hook.on_agent_error         (제어흐름=interrupt는 제외)
 │   └─ Hook.finally_agent
 └─ 결과 반환
```

**Hook onion 순서**: before는 priority 오름차순, after/finally는 역순. `is_control_flow()`로 LangGraph interrupt(`GraphBubbleUp`)를 오류 처리·재시도에서 제외한다.

---

## 4. 확장점 (Adapter / SPI)

특정 제품 종속 금지 — 전부 교체 가능:

| 확장점 | 표준 인터페이스 | 등록/주입 |
|--------|----------------|-----------|
| Checkpointer | LangGraph `BaseCheckpointSaver` | `register_checkpointer(name, factory)` / `checkpointer="postgres"` |
| Store (Memory) | LangGraph `BaseStore` | `register_store(...)` / `store="memory"` |
| Model Provider | `ModelProvider` (`(prompt)->ModelResult`) | `ModelGateway.register(alias, provider)` |
| Trace Backend | OTel `SpanExporter` | `setup_tracing(exporter=...)` |
| Approval | `ApprovalAdapter` | `register_approval_adapter(...)` |
| Evaluator | `Evaluator` | `run_offline(..., evaluators=[...])` |
| Guardrail | `@guardrail` → `Guardrail`(`check(text)->GuardrailResult`) | 코드: `@guard`(노드·워크플로우) / `GuardrailHook`(공통 훅) |
| Named Hook | `@klafi_hook("name")` | `config/hooks.yaml` `hooks: [...]` |
| Registry Store | `RegistryStore` | `AgentRegistry(store=...)` |
| Event Subscriber | `Subscriber` | `subscribe(handler, types=[...])` |

**LangGraph 실행엔진 활용**: Checkpoint/Store/Interrupt/Streaming은 재구현하지 않고 LangGraph 엔진을 내부에서 사용한다(`CheckpointerSPI = BaseCheckpointSaver`, `MemoryStoreSPI = BaseStore`). 단, 개발자 진입은 KlafiGraph 상속으로 통일했고 "감싸기만 한다"는 제약은 두지 않는다.

---

## 5. 핵심 설계 결정

- **ContextVar 실행 Context** — 전역변수 금지(§9). Node가 Logger·인증·Trace를 인자로 받지 않고 `get_context()`로 꺼낸다. Timeout worker thread에는 `copy_context()`로 전파, async는 자연 전파.

- **Hook 기반 횡단관심사** — Before/After가 **Graph·Node·Tool·LLM** 4지점에서 실행된다. Node는 `wrap_node`가, Tool/Model은 실행 중 활성 Hook(`bind_hooks` contextvar)을 `Tool.run`/`ModelGateway`가 발화한다. 관측은 `fail_open=True`, Guardrail은 `fail_open=False`(차단). → §25 Fail-Open/Close를 Component별로 실현.
- **축: 전역=훅(관찰), 지역=가드레일(판정·치환·관측)** — 전역 관심사(Logging/Metrics/Tracing)는 훅으로 모든 에이전트에 붙고 **값을 바꾸지 않는다**(`_fire`가 반환값을 버림). 노드/그래프 **지역**에 붙는 판정·치환·관측은 `@klafi_node`/`@klafi_graph`의 `before`/`after` 리스트로 표현한다.
- **가드레일은 문자열 정책, 바인딩이 모양을 안다** — 가드레일은 `check(text) -> 판정/치환text` 하나로 통일된다. 값의 모양(state dict, `BaseMessage` 객체, tool kwargs, LLM str)을 아는 것은 **바인딩**(`klafi/guardrail/binding.py`)의 몫으로, 값의 **문자열 리프에만** 정책을 적용하고 구조·타입·메시지 `id`를 유지한 채 되돌려 쓴다. 덕분에 통짜 `json.dumps` 투영이 사라져 (a) 역변환이 불필요하고 (b) 메시지 id가 보존되어 `add_messages`가 append 아닌 **교체**를 하며 (c) id·metadata·토큰수 같은 엉뚱한 필드를 스캔·치환하지 않는다. 구조 전체를 판정·치환해야 할 때만 `@guardrail(raw=True)`로 원본을 받는다. **의도된 축소**: `messages`는 마지막 메시지만 검사한다(앞의 것은 마지막이었을 때 이미 검사됨).
- **노드는 모두 @klafi_node(강제)** — 모든 그래프 노드는 `@klafi_node("<이름>")`로 선언해야 한다(이름 필수). `KlafiGraph.build()`가 `define()` 직후 검사해 미선언 노드면 기동 실패시킨다(ToolNode 등 Runnable은 예외). `before`/`after` 리스트에 **두 종류**가 섞인다(원소 타입으로 구분): ① **가드레일**(`.check` 보유) — 바인딩이 문자열 리프마다 판정(BLOCK/WARN)·치환(MASK: `GuardrailResult(replacement=...)`); ② **값 콜러블** — 값 전체를 한 번 받아 반환하면 교체(세션·권한 검증, 지역 관측). 값 콜러블은 리프 유무와 무관하게 **항상 한 번** 도는 반면 가드레일은 문자열 리프가 없으면 호출되지 않으므로, `require_orders_read` 같은 '내용과 무관하게 반드시 도는' 검증은 값 콜러블이어야 한다(조용한 권한 우회 방지). 리스트 순서대로 적용된다(정규화 후 검사 순서도 표현 가능). `wrap_node`(common→agent) 안쪽에서 도므로 어니언은 `common → agent → before → fn → after → agent → common`. **입력(before) 마스킹은 노드 본문이 보는 뷰만 바꾸고 저장되지 않는다** — 저장까지 가리려면 after에 둔다.
- **워크플로우 경계는 @klafi_graph** — 같은 규칙(가드레일·값 콜러블 혼합)을 그래프 전체에 적용한다: `@klafi_graph(before=[...], after=[...], on_error=[...])`. `BaseGraph`의 실행 파이프라인이 `before → 그래프 → after` 순으로 적용한다(훅이 값을 못 바꾸므로 훅이 아니라 파이프라인). **스트리밍에서는 after가 미적용**(단일 결과가 없어 검사 지점이 없음 — `after_agent` 훅도 같은 이유). 그래프 after는 응답만 가리고 **체크포인트·스트림은 원본**이다. → **마스킹은 노드 after에 붙여라**: 응답·저장·스트림 셋 다 커버된다.
- **LLM·Tool 경계 치환** — 데코레이터를 붙일 수 없는 두 경계는 `GuardrailHook`(공통 훅)이 가드레일을 실어 나른다. `before_model`/`after_model`/`before_tool`/`after_tool` 네 콜백이 `_transform`으로 발화되어 **반환값이 프롬프트·응답·인자·결과를 교체**한다(관측 훅은 `None`을 반환해 무영향). `fail_open` 훅이 예외를 던지면 값은 그 훅 직전 상태를 유지한다(부분 변환 격리). 단 **chat_model 콜백 경로**(`init_chat_model`)는 이미 만들어진 요청/응답의 사이드채널이라 치환이 구조적으로 불가능 — 판정(차단)만 되고 마스킹은 `guardrail.mask_ignored` 경고와 함께 무시된다(노드 after에 붙일 것). 하나의 문자열 정책(`mask_phone`)이 노드·그래프·gateway·tool 경계 전부에 그대로 꽂힌다.
- **가드레일은 코드로 적용** — `@guardrail`이 함수를 `Guardrail` 객체로 만든다(prebuilt: `pii`/`prompt_injection`). 적용 지점 셋: ① 노드/워크플로우 `@klafi_node`/`@klafi_graph`의 `before`/`after`, ② 플랫폼 공통 `GuardrailHook`(hooks로 주입, input/output/model/model_output/tool/tool_output 스테이지). 이름 레지스트리·YAML 배치는 폐기 — `hooks.yaml`은 명명 훅(`hooks:`)만 담고, `guardrails:` 키가 있으면 기동 시 fail-fast.
- **model_output(응답) 가드레일** — `model`은 프롬프트(`before_model`)만, `model_output`은 응답(`after_model`)을 검사한다. `with_structured_output` 등 tool-calling 방식 structured output은 `content`가 비고 실데이터가 `tool_calls`에 실리므로, `KlafiCallbackHandler._extract`가 `content`가 비어 있으면 `tool_calls`로 폴백해 가드레일이 빈 문자열을 보지 않게 한다. Stream 응답도 `on_llm_end`가 전체 누적 텍스트로 발화하므로 검사 자체는 되지만, 청크를 실시간으로 외부에 흘려보내는 노드에서는 이미 나간 내용을 사후 차단으로 되돌릴 수 없다(구조적 한계).
- **chat_model 계측** — `init_chat_model(alias)`(= `gateway.chat_model(alias)`)는 LangChain 모델에 `KlafiCallbackHandler`를 주입해 돌려준다. 개발자가 `bind_tools`·`with_structured_output`로 파생 Runnable을 만들어도 실제 LLM 호출마다 span·Token/Cost·`before/after_model` 훅·`ModelCalled` 이벤트가 적용된다(`raise_error=True`라 model-stage Guardrail이 호출을 차단할 수 있다). **Structured Output은 LangChain 네이티브를 그대로 쓰되 관측은 유지**된다.
- **모델 선언은 한 가지** — `init_chat_model("<alias>")`. alias는 `config/model.yaml`이 정의하고, 활성 Gateway는 `ExecutionFactory` 조립 시 정해진다. `self.chat_model`·`self.gateway` 같은 별도 주입 경로는 두지 않는다(같은 일을 하는 길이 둘이면 코드베이스가 갈라진다).
- **Tool 연결은 LangGraph 네이티브** — 바인딩은 `model.bind_tools([...])`, 실행 노드는 `self.make_tool_node([...])`(=`ToolNode`)다. KLAFI Tool→LangChain 변환(`as_langchain()`)은 두 진입점 안에서 일어나므로 업무 코드에 변환 호출이 나타나지 않는다. ToolNode가 실행해도 권한·검증·Timeout·Hook은 KLAFI `Tool.run`에서 그대로 적용된다.
- **Skill = Tool + 지침** — `init_chat_model(alias).bind_skills([skill])`은 툴을 `bind_tools`로 붙이고 `skill.prompt`를 `SystemMessage`로 선행 주입한다. 툴 사용법을 업무 코드마다 다시 적지 않게 하는 것이 목적이다.

- **정책은 코드 밖** — Timeout/Retry는 `ExecutionPolicy`로 주입(Agent 코드 하드코딩 금지). retry가 timeout을 감싸 재시도마다 새 timeout. Guardrail/Policy 예외는 결정적이라 재시도 제외.
- **동시 실행 상한(Concurrency)** — `policy.yaml: concurrency: N` → `create_app(max_concurrency=N)`가 서버 전역 `threading.BoundedSemaphore(N)`로 슬롯을 관리한다. invoke/resume(sync 스레드풀)·stream(async) 모두 진입 시 non-blocking acquire, 초과분은 **429**(백프레셔)로 즉시 거절하고 슬롯은 요청 종료 시 반납. 상한은 **worker 1개 기준**이라 `--workers K`면 실질 `N×K`이고, postgres면 pool `max_size`와 함께 산정한다(구조적 한계는 게이트웨이 레이트리미터로 보완).

- **Runtime ≠ HTTP** — `AgentServer`(런타임 레지스트리)는 FastAPI를 모른다. `create_app`이 유일한 FastAPI 의존 지점 → Runtime 교체가 Agent 코드에 무영향(§19).

- **관측/이벤트 연결** — Token/Cost는 `model.{alias}` span 속성, Business Exception은 span에 record → `Execution ID → Agent → Node → Tool → Model → Error` 단일 Trace(§16). Event Bus로 Monitoring/Audit/Billing이 결합 없이 구독(§24).

---

## 5.5 예외 체계 (§23)

모든 예외는 `KlafiException` 하위이며 **두 축**으로 잡을 수 있다. 새 타입은 기존 도메인 예외를 상속하므로 `except ToolException` 같은 기존 코드는 그대로 동작한다.

```
KlafiException
├─ 도메인 축: AgentExecution · Model · Tool · Policy · Guardrail · Context · Config · Checkpoint · Approval
└─ 종류  축: NotFoundError · ValidationError · PermissionDeniedError · ViolationError
```

| 상황 | 예외 | error_code | 두 축 |
|------|------|-----------|-------|
| config 디렉터리·environment 미설정 | `ConfigNotFoundError` | `CONFIG_NOT_FOUND` | Config + NotFound |
| 설정 키·stage 오타 | `ConfigSchemaError` | `CONFIG_SCHEMA_ERROR` | Config + Validation |
| 지원하지 않는 설정 값 | `ConfigValueError` | `CONFIG_VALUE_ERROR` | Config + Validation |
| tool 이름 못찾음 | `ToolNotFoundError` | `TOOL_NOT_FOUND` | Tool + NotFound |
| model alias 못찾음 | `ModelNotFoundError` | `MODEL_NOT_FOUND` | Model + NotFound |
| API 키 미설정 | `ModelNotConfiguredError` | `MODEL_NOT_CONFIGURED` | Model + Config |
| hook 이름 못찾음 | `HookNotFoundError` | `HOOK_NOT_FOUND` | NotFound |
| agent 미등록 (Runtime·Registry) | `AgentNotFoundError` | `AGENT_NOT_FOUND` | NotFound |
| **실행 중 가드레일 차단** | `GuardrailViolationError` | `GUARDRAIL_VIOLATION` | Guardrail + **Violation** |
| tool 권한 부족 | `ToolPermissionError` | `TOOL_PERMISSION_DENIED` | Tool + PermissionDenied |
| tool 입출력 검증 실패 | `ToolValidationError` | `TOOL_VALIDATION_ERROR` | Tool + Validation |
| 실행 timeout | `TimeoutException` | `TIMEOUT_ERROR` | AgentExecution |

**설정 오류 vs 실행 위반 구분이 핵심**: 설정에 없는 이름 참조는 기동 시 `NotFound`로 실패하고, 실행 중 가드레일 차단은 `GuardrailViolationError`(정상 동작)다. 가드레일은 코드로 참조하므로 "guardrail 이름 못찾음" 예외는 없다. 모든 예외는 `error_code`와 컨텍스트(`stage`, `guard`, `tool`, `execution_id` 등)를 함께 담는다.

---

## 6. 상태 · 생명주기

- **Execution 상태**(§8): `CREATED → RUNNING → COMPLETED` / 예외 `FAILED·TIMEOUT`, HITL 시 `WAITING_APPROVAL`.
- **Agent Lifecycle**(§15, Registry): `DEVELOPMENT → TEST → APPROVED → PRODUCTION → DEPRECATED → RETIRED`. 불가능한 전이는 거부.

---

## 7. 설정 우선순위 (§22)

```
Framework Default → Environment → Project → Agent → Runtime Override
(config/framework·model·policy.yaml + environments/ + agents/)
```
`LayeredConfig`가 deep-merge로 실효 설정을 산출하고, `ExecutionPolicy.from_config` 등에 공급.

**Fail-fast 검증** — 잘못된 설정은 첫 요청이 아니라 **부트스트랩(`KlafiApp.from_config`) 시점**에 `ConfigException` 등으로 즉시 실패한다:

| 대상 | 검출 |
|------|------|
| config 디렉터리 경로 / `environment` 파일 | 존재하지 않으면 에러 (빈 설정으로 조용히 기동 금지) |
| `policy.yaml` 키 | `timeout`·`max_retries`·`backoff_*`·`concurrency` 외 오타는 에러 — 정책 미적용을 숨기지 않음 |
| `hooks.yaml` 스키마 | 최상위(`all`/`agents`)·블록(`hooks`)만 허용. `guardrails` 키가 있으면 에러(가드레일은 코드로) |
| `hooks.yaml` 이름 | 미등록 훅·가드레일 이름 에러 (`validate_names`가 기동 시 전 에이전트 확인) |
| provider / checkpoint / store `type` | 미지원 값 에러 |
| Agent의 `spec.model` alias | 미등록 alias면 `app.register()`(=기동) 시 에러 |

---

## 8. 모듈 책임 요약

| 패키지 | 책임 | 주요 타입 |
|--------|------|-----------|
| `core` | SDK·실행 진입·Context·Hook·Exception | `BaseGraph` `ExecutionContext` `Hook` `KlafiException` |
| `core.graph` | 개발자 진입 그래프 클래스(상속 + define) | `KlafiGraph` |
| `app` | 애플리케이션 부트스트랩(config→인프라 주입) | `KlafiApp` |
| `runtime` | 실행 엔진·정책·상태 | `run_sync/async` `ExecutionPolicy` `ExecutionState` |
| `context` | 체크포인트·장기기억·히스토리 관리 | `resolve_checkpointer` `MemoryStore` `ContextManager` `ContextHook` |
| `observability` | OTel 트레이싱 | `setup_tracing` `TracingHook` `span` |
| `model` | 모델 게이트웨이·모델 선언 | `ModelGateway` `init_chat_model` `ChatModel` |
| `tool` | 툴 프레임워크·스킬 | `Tool` `Skill` `ToolRegistry` |
| `guardrail` | 입출력 가드레일 | `GuardrailHook` `Guardrail` |
| `evaluation` | 평가·버전비교 | `Evaluator` `run_offline` `EvaluationReport` |
| `hitl` | 승인 프로세스 | `request_approval` `resume_approval` |
| `templates` | 재사용 Agent | `SimpleAgent` `RAGAgent` `SupervisorAgent` |
| `server` | 서비스화 | `AgentServer` `create_app` |
| `registry` | Control Plane | `AgentRegistry` `AgentRecord` `AgentLifecycle` |
| `events` | 이벤트 버스 | `EventBus` `EventHook` `emit/subscribe` |
| `config` | 설정 병합 | `LayeredConfig` `deep_merge` |
