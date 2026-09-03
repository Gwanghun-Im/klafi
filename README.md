# KLAFI

**Enterprise Agentic AI Engineering Framework** — LangGraph/LangChain 기반.

KLAFI는 LangGraph의 자유도를 유지하면서, Enterprise SI에서 반복되는 실행·통제·관측·평가·운영 기능을 표준화한다. 개발자는 **State·Node·Edge·Prompt·Tool** 만 작성하고, 나머지(Logging·Checkpoint·Retry·Timeout·Trace·Model·Guardrail·Evaluation·API·Memory·Security)는 KLAFI가 담당한다.

> **핵심 원칙 한 문장: "LangGraph의 자유도는 유지하고, Enterprise 개발의 반복은 제거한다."**

설계 원칙과 내부 구조는 [ARCHITECTURE.md](ARCHITECTURE.md), 개발 경과는 [개발계획서](KLAFI_개발계획_v1.0.md) 참고.

---

## 설치

```bash
pip install -e .                # core (FastAPI Agent Server 포함)
pip install -e ".[anthropic]"   # + Claude  (OpenAI 는 [openai])
```

의존: `langgraph`, `langchain-core`, `pydantic`, `pyyaml`, `opentelemetry-api/sdk`, `fastapi`(Agent Server, 코어). 모델 SDK(`anthropic`/`openai`)와 ASGI 실행기(`uvicorn`)는 앱이 고르는 별도 항목. (`[server]` extra 는 하위호환용으로 남아 있음 — fastapi 가 이미 코어라 no-op)

---

## 30초 Quickstart

```python
from klafi.templates import SimpleAgent
from klafi.model import ModelGateway, FunctionProvider
from klafi.runtime import ExecutionPolicy
from klafi.observability import setup_tracing

setup_tracing(service_name="my-agent")           # Observability (선택)

gw = ModelGateway()
gw.register("quality-high", FunctionProvider(lambda p: f"답변: {p}"), cost=(1.0, 3.0))

agent = SimpleAgent(
    model=gw.model("quality-high"),              # Alias만 노출, Token/Cost 자동 기록
    checkpointer="memory",                       # Resume 가능
    policy=ExecutionPolicy(timeout=30, max_retries=2),  # Timeout/Retry
)
print(agent.invoke({"question": "KLAFI가 뭐야?"}, thread_id="t1")["answer"])
```

개발자가 짠 것은 모델 함수 하나. **Logging·Tracing·Checkpoint·Retry·Timeout·Token 집계는 자동.**

---

## 핵심 개념

| 개념 | 설명 |
|------|------|
| `KlafiGraph` | 개발자가 상속하는 표준 그래프 클래스. `state_schema`를 지정하고 `define()`에서 `self.add_node/add_edge`로 그래프를 조립하면 execution_id·Hook·정책·체크포인트·스토어가 자동으로 붙는다. LangGraph StateGraph 빌더 API를 그대로 노출하고, `.builder`/`.compiled`로 native 접근도 열려 있다. |
| `BaseGraph` | KlafiGraph의 실행 기반(저수준). invoke/async/stream·Hook·정책·Context를 제공. |
| `ExecutionContext` | execution_id/trace_id/agent/user/tenant/security 등 실행정보. **ContextVar 기반**(전역변수 금지)이라 Node가 `get_context()`로 꺼내 쓴다. |
| `Hook` | Before/After를 **Graph·Node·Tool·LLM**에 자동 부착(AOP) + Error/Finally. `fail_open`으로 관측(Fail-Open) vs 차단(Fail-Close, Guardrail) 구분. |
| Guardrail | `@guardrail`로 함수를 Guardrail 객체로 만들고(prebuilt: `pii`/`prompt_injection`), **코드에서 직접 적용** — 노드·워크플로우는 `@guard(input=..., output=...)`, 플랫폼 공통은 `GuardrailHook`. 위반 시 fail-close. |
| 모델 선언 | `init_chat_model("<alias>")` **하나**. alias는 `config/model.yaml`이 매핑하고, 업무 코드에는 provider·모델명이 나오지 않는다. Token/Cost·훅·가드레일 계측이 붙은 모델이 나온다. |
| Tool / Skill | `@tool`로 만든 KLAFI Tool을 `bind_tools([...])`에 그대로 넣는다(변환 불필요). **Skill = 툴 + 사용 지침**이며 `bind_skills([...])`로 붙이면 지침이 SystemMessage로 자동 주입된다. |
| `ExecutionPolicy` | Timeout·Retry·Backoff·**Concurrency**(서버 전역 동시 실행 상한). Agent 코드가 아닌 Config/인자로 주입. |
| Adapter | Checkpointer/Store/Model/Trace/Approval/Evaluator/Guardrail을 전부 교체 가능. |
| 예외 | 전부 `KlafiException` 하위. **도메인 축**(Tool/Model/Guardrail/Config…)과 **종류 축**(NotFound/Validation/Permission/Violation) 두 방향으로 잡는다. 잘못된 설정은 기동 시 Fail-Fast. |

---

## Agent 만드는 법 — KlafiGraph 상속

개발자는 **`KlafiGraph`를 상속**하고 `state_schema` + `define()`만 작성한다. LangGraph의 `add_node/add_edge/add_conditional_edges` 를 `self`에서 그대로 쓰고, 실행·Hook·정책·체크포인트는 상속으로 자동 획득한다. `.builder`/`.compiled`로 native 접근도 열려 있다(Open Framework).

```python
from klafi.core import KlafiGraph, AgentSpec
from klafi.runtime import ExecutionPolicy
from langgraph.graph import START, END

class MyAgent(KlafiGraph):
    spec = AgentSpec(id="my", name="My", model="main")   # model은 alias로 주입
    state_schema = MyState
    def define(self):                                    # 노드는 모두 @klafi_node
        @klafi_node("plan")
        def plan(state): ...
        @klafi_node("review")
        def review(state): ...
        self.add_node("plan", plan); self.add_node("review", review)
        self.add_edge(START, "plan")
        self.add_conditional_edges("review", route, {"plan": "plan", END: END})

agent = MyAgent(model=my_model, checkpointer="memory", policy=ExecutionPolicy(timeout=30))
agent.invoke({...})
```

- **표준 패턴이 필요하면** 제공되는 KlafiGraph 하위 Template을 쓴다: `SimpleAgent` · `RAGAgent` · `SupervisorAgent`.
- **저수준 제어**가 필요하면 `BaseGraph`를 직접 상속(`build() -> StateGraph`)한다 — Hook 기본 탑재 없이 완전 수동.

예제: [examples/support_platform/app/agents/](examples/support_platform/app/agents/).

---

## 프로젝트 구조 (Agent가 많아질 때) — 3-역할 분리

Agent가 늘면 모델 연결·정책·보안 같은 공통 설정이 파일마다 흩어진다. **Spring Boot처럼** 공통 설정은 `config/`로 모으고 업무개발자는 업무 로직만 짜도록 `KlafiApp`을 쓴다.

```
my-project/
├─ config/                 ← 공통개발자 (≈ application.yml)
│  ├─ framework.yaml       #   service, checkpoint, store
│  ├─ model.yaml           #   모델 연결 (= DB 연결처럼 한곳에서)
│  └─ policy.yaml          #   timeout · retry · concurrency(동시 실행 상한)
├─ guardrails.py           #   Guardrail(@guardrail) — 문자열 정책
├─ middleware.py           #   노드 미들웨어(값 콜러블) — 권한·감사
├─ hooks.py                #   Hook(코드) — 관측·정책 + 히스토리 압축(context). YAML 아님
├─ agents/                 ← 업무개발자 (≈ @Controller/@Service)
│  ├─ qa_agent.py          #   class QAAgent(KlafiGraph). 인프라 코드 0줄
│  └─ summarize_agent.py
├─ bootstrap.py            ← 공통개발자 (≈ @Configuration: KlafiApp 조립·등록)
├─ server.py               ← 공통개발자 (실제 진입점: app.http_app())
└─ demo.py                 ← 로컬 CLI 확인용 (선택)
```

| Spring Boot | KLAFI | 담당 |
|-------------|-------|------|
| `application.yml` | `config/*.yaml` | 공통개발자 |
| `@Configuration` / DataSource Bean | `bootstrap.py` (`KlafiApp.from_config()`) | 공통개발자 |
| `main()` 진입점 | `server.py` (ASGI) / `demo.py` (CLI) | 공통개발자 |
| `@Controller` / `@Service` | `agents/*.py` (`KlafiGraph` 하위 클래스) | 업무개발자 |

**업무개발자** — 모델 alias만 선언하고 업무 그래프만 짠다 (gateway/tracing/checkpointer/guardrail 코드 없음):

```python
# agents/qa_agent.py
class QAAgent(KlafiGraph):
    spec = AgentSpec(id="qa", name="QA Agent", model="main")   # 'main' = config가 매핑할 alias
    state_schema = State
    def define(self):                                          # model은 KlafiApp이 주입
        @klafi_node("answer")
        def answer(s):
            return {"answer": self.model(s["question"])}
        self.add_node("answer", answer)
        self.add_edge(START, "answer"); self.add_edge("answer", END)
```

**공통개발자** — config로 인프라를 통제하고 등록만 한다:

```python
# bootstrap.py  (조립) — server.py/demo.py 공용
from klafi.app import KlafiApp

def build_app():
    app = KlafiApp.from_config("config")       # 모델·정책·Checkpoint 조립
    app.register_package("agents")             # agents/ 하위를 자동 등록(convention)
    return app                                 #   owner 는 각 spec.owner 사용, _접두 폴더는 스킵

# server.py  (실제 진입점)
app = build_app().http_app()                   # 등록된 전 Agent를 HTTP 서비스
```

> 업무개발자는 `agents/` 에 파일/폴더만 떨구면 서비스된다 — bootstrap(공통개발자 영역)을 안 건드린다.
> 명시적으로 통제하고 싶으면 `app.register(QAAgent, owner="team-qa")` 로 하나씩 등록해도 된다.

```yaml
# config/model.yaml — 운영 전환은 업무코드 변경 없이 여기만
providers:
  main: { type: anthropic, model: claude-haiku-4-5-20251001, cost: [0.001, 0.005] }
  fast: { type: openai,    model: gpt-4o-mini }
```

각 Agent에는 config의 정책·Checkpoint가 자동 적용되고(가드레일은 코드로 부착), Registry(owner/version)에도 등록된다. 동작하는 예제: [examples/support_platform/](examples/support_platform/) (`cd examples/support_platform && python demo.py`).

---

## 기능별 사용

<details><summary><b>실행 — invoke / async / stream</b></summary>

```python
agent.invoke(inp)                 # sync
await agent.ainvoke(inp)          # async
for ev in agent.stream(inp): ...  # streaming
```
모든 실행에 고유 `execution_id` 발급, `state`(CREATED→RUNNING→COMPLETED/FAILED/TIMEOUT/WAITING_APPROVAL) 추적.
</details>

<details><summary><b>Hook (Graph·Node·Tool·LLM before/after)</b></summary>

```python
class MyHook(Hook):
    def before_tool(self, tool, kwargs, ctx): ...    # Tool 진입
    def before_model(self, model, prompt, ctx): ...  # LLM 호출 직전 (프롬프트 검사 등)
    def before_node(self, node, state, ctx): ...
    def before_agent(self, input, ctx): ...          # = Graph
```
공통 훅은 **공통 훅 파일 한 곳**(코드, `PLATFORM_HOOKS`)에서 관리한다. 특정 노드 전용 훅은 `@klafi_node(before=/after=)` 미들웨어로 노드 옆에 둔다.
`KlafiGraph`는 Logging/Tracing을 기본 탑재 — Node 코드에 로깅 한 줄 없이 `node.start`/`node.end`가 자동 생성.

**모든 그래프 노드는 `@klafi_node("<이름>")`로 선언한다(이름 필수, ToolNode는 예외).** KlafiGraph가 강제한다.
- `before`/`after`/`on_error` — 노드 전용 **미들웨어 콜러블**(가드레일 아님도 됨). `before`가 값을 반환하면 body로 넘어가는 state를 **교체**, `after`가 반환하면 노드 출력을 교체. 검증 실패는 그냥 예외를 던진다(fail-close).
- `before`/`after` — 한 리스트에 **가드레일과 미들웨어를 섞어** 넣는다(`.check` 유무로 구분). 리스트 순서대로 적용된다.
- 가드레일 등급: `BLOCK`(차단) / `WARN`(기록만) / `MASK`(`GuardrailResult(replacement=...)` → 차단 대신 치환).
- 가드레일은 기본이 텍스트 검사기이고, `@guardrail(raw=True)`면 원본 객체(state dict 등)를 그대로 받는다. 구조를 치환하려면 raw=True 가 필요하다.

```python
def require_login(state, ctx):                    # 세션 검증 + state 보강
    if not ctx.security_context.get("user_id"):
        raise PermissionError("로그인 필요")
    return {**state, "verified": True}            # state 교체

@klafi_node("plan", before=[require_login], input=[pii], output=[pii])
def plan(state): ...
self.add_node("plan", plan)
# 어니언: common → agent → input가드 → before → fn → after → output가드 → agent → common
```
</details>

<details><summary><b>Checkpoint / Resume · Long-Term Memory</b></summary>

```python
agent = SimpleAgent(model=m, checkpointer="memory", store="memory")
agent.invoke(inp, thread_id="t1")          # 중단되면
agent.invoke(None, thread_id="t1")         # 동일 thread로 Resume

mem = agent.memory()                        # 세션 넘는 장기 기억
from klafi.context.memory import user_scope
mem.remember(user_scope("u1"), "pref", {"lang": "ko"})
mem.recall(user_scope("u1"), "pref")
```
Checkpoint=실행상태(Thread), Memory=지속 지식(Scope). 별개.

**운영(PostgreSQL)** — ConnectionPool + 스키마 자동 생성. DSN은 config에 평문으로 두지 않고 환경변수로 주입한다(SEC-05).

```yaml
# config/environments/postgres.yaml
checkpoint: { type: postgres, conn_string: ${KLAFI_PG_DSN}, min_size: 1, max_size: 10 }
store:      { type: postgres, conn_string: ${KLAFI_PG_DSN} }
```
접속 실패는 기동 시 5초 내 `CheckpointException`으로 끝난다(비밀번호 미노출).

**히스토리 자동 관리** — 대화가 길어지면 `ContextHook`이 Node 진입 시 오래된 부분을 요약·압축한다. 모델 입력 토큰만 줄고 **Checkpoint 원본은 보존**된다(감사·재현).

```python
# hooks.py (공통개발자) — 훅은 YAML 이 아니라 코드로 선언한다
def context_hook(gateway):
    return ContextHook(max_tokens=400,      # 임계 초과 시 압축
                       keep_recent=4,        # 최근 N건은 원문 보존
                       summarizer=gateway.model("fast"))  # 요약용 alias (생략 시 트림만)
# bootstrap: 조립 후 gateway 를 주입해 적용
app.factory.base_hooks.append(context_hook(app.gateway))
```
</details>

<details><summary><b>Observability</b></summary>

```python
from klafi.observability import setup_tracing, span
setup_tracing(exporter=my_otlp_exporter)   # Loki/Tempo/Langfuse(OTLP)
with span("tool.search", **{"klafi.tokens": 42}):  # Node 하위 자동 중첩
    ...
```
`Execution → Agent → Node → Tool → Model → Error`가 하나의 Trace. Backend 장애 시 Fail-Open(업무 지속).
</details>

<details><summary><b>Human-in-the-Loop</b></summary>

```python
from klafi.hitl import request_approval, resume_approval
def report_node(state):
    d = request_approval("publish", approver="editor")   # interrupt로 중단
    return {"result": "발행" if d.approved else "보류"}

resume_approval(agent, "t1", approved=True)              # 승인 재개
```
</details>

<details><summary><b>Guardrail (@guardrail + @guard) / Model Gateway / Tool</b></summary>

```python
# 1) 가드레일 선언 — 함수를 Guardrail 객체로 (klafi prebuilt: pii, prompt_injection)
from klafi.guardrail import guardrail, guard, pii, prompt_injection
@guardrail
def no_secrets(text): return "비밀번호" not in text

# 2) 코드에서 직접 적용. 세 지점:
#    · 노드           @klafi_node("n", before=[require_login, no_secrets], after=[mask_pii]) def node(state): ...
#    · 전달 계약       @klafi_node("route", visibility="internal")   → 스트림에 토큰·updates 미전달(내부 판단 노드)
#                     @klafi_node("extract", output=Report)          → 반환값을 스키마로 검증·강제, 스트림엔 structured 청크 1회,
#                                                                  /agents/{id} 의 nodes 에 output_schema(JSON Schema) 노출
#    · 워크플로우 전체  @klafi_graph(before=[require_login, no_secrets]) class MyAgent(KlafiGraph): ...
#    · 공통 훅(플랫폼)  GuardrailHook(input=[no_secrets], output=[pii], model=[prompt_injection],
#                                   model_output=[pii])  →  KlafiApp platform_hooks 로 주입
#    위반 시 GuardrailViolationError(fail-close). input=들어온 값, output=반환값.

@tool(required_permission="db:write", policy=ExecutionPolicy(timeout=5))
def writer(x: str) -> str: ...              # 권한·검증·Timeout·Metric 자동

# 모델 선언은 init_chat_model(alias) 하나, Tool 연결은 LangGraph 네이티브(bind_tools + ToolNode)
class Agent(KlafiGraph):
    state_schema = MessagesState
    def define(self):
        llm = init_chat_model("main").bind_tools([writer])   # KLAFI Tool 그대로
        self.add_node("agent", lambda s: {"messages": [llm.invoke(s["messages"])]})
        self.add_node("tools", self.make_tool_node([writer]))     # LangGraph ToolNode
        self.add_conditional_edges("agent", tools_condition)
        self.add_edge("tools", "agent"); self.add_edge(START, "agent")
# ToolNode가 실행해도 권한·검증·Hook은 KLAFI Tool.run에서 그대로 적용

# Structured Output — LangChain 네이티브를 그대로 사용
class Ticket(BaseModel):
    category: str
    urgency: int

llm = init_chat_model("main").with_structured_output(Ticket)   # → Ticket 인스턴스 반환

# Skill — 툴 + 사용 지침을 한 단위로 (지침은 SystemMessage로 자동 주입)
clock = Skill(name="clock_kst", tools=[kst_now], prompt="한국 시각이 필요하면 kst_now 툴을 사용한다.")
llm = init_chat_model("main").bind_skills([clock])
```
`init_chat_model(alias)`가 돌려주는 모델에는 KLAFI 계측 핸들러가 주입돼 있어, `bind_tools`·`with_structured_output` 어느 쪽으로 파생시켜도 **span·Token/Cost·`before/after_model` 훅·model-stage 가드레일이 그대로 적용**된다.
</details>

<details><summary><b>예외 처리 — 도메인 축 / 종류 축</b></summary>

모든 예외는 `KlafiException` 하위이며 **두 방향**으로 잡을 수 있다.

```python
from klafi.core.exceptions import ToolException, NotFoundError, GuardrailViolationError

try:
    agent.invoke(...)
except GuardrailViolationError as e:      # 실행 중 가드레일 차단만
    log.warning("blocked at %s by %s", e.context["stage"], e.context["guard"])
except NotFoundError:                      # tool·model·hook·agent 미등록 (대개 설정 오류)
    ...
except ToolException:                      # tool 관련 전부 (도메인)
    ...
```

| 상황 | 예외 | error_code |
|------|------|-----------|
| config 디렉터리·environment 미설정 | `ConfigNotFoundError` | `CONFIG_NOT_FOUND` |
| 설정 키·stage 오타 | `ConfigSchemaError` | `CONFIG_SCHEMA_ERROR` |
| tool / model alias / hook 못찾음 | `ToolNotFoundError` · `ModelNotFoundError` · `HookNotFoundError` | `*_NOT_FOUND` |
| API 키 미설정 | `ModelNotConfiguredError` | `MODEL_NOT_CONFIGURED` |
| **실행 중 가드레일 차단** | `GuardrailViolationError` | `GUARDRAIL_VIOLATION` |
| tool 권한 부족 / 입출력 검증 실패 | `ToolPermissionError` · `ToolValidationError` | `TOOL_PERMISSION_DENIED` · `TOOL_VALIDATION_ERROR` |
| 실행 timeout | `TimeoutException` | `TIMEOUT_ERROR` |

가드레일은 코드로 참조하므로 "미등록" 예외가 없다. 위반은 실행 중 `GuardrailViolationError`(정상 fail-close)로만 발생한다. 모든 예외가 `error_code`와 컨텍스트(`stage`·`guard`·`tool`·`execution_id`)를 담는다.

**Config Fail-Fast**: 잘못된 설정은 첫 요청이 아니라 `KlafiApp.from_config()` 시점에 즉시 실패한다 — 디렉터리·environment 미설정, policy 키 오타, hooks.yaml 스키마·미등록 이름, 지원하지 않는 `type`. 메시지에 허용 값 목록이 함께 나온다.
</details>

<details><summary><b>Evaluation · Registry · Config · Event</b></summary>

```python
from klafi.evaluation import run_offline, RuleEvaluator
from klafi.registry import AgentRegistry
from klafi.config import LayeredConfig
from klafi.events import subscribe, EventType
report = run_offline(agent, dataset, [RuleEvaluator(rule)])
report.compare_versions("quality")          # Agent Version별 품질 비교

reg = AgentRegistry(); reg.register_agent(agent, owner="team-a"); reg.approve("id","1.0.0")
cfg = LayeredConfig.from_dir("config", environment="prod", agent_id="qa")   # 5계층 병합
subscribe(handler, types=[EventType.ModelCalled])                          # 결합 저감 확장점
```
</details>

---

## Agent 서비스화 (API)

`config/policy.yaml`의 `concurrency`로 **서버 전역 동시 실행 상한**을 둔다(백프레셔). 초과 요청은 대기 없이 **429 Too Many Requests**(`Retry-After: 1`)로 즉시 거절된다 — invoke·resume·stream 모두 적용, `KlafiApp.http_app()`이 자동 전달.

```yaml
# config/policy.yaml
concurrency: 20    # 미설정 시 무제한. 동시 20건까지 실행, 21번째부터 429
```

주의: 이 상한은 **uvicorn worker 1개 기준**이다. `--workers K`면 실질 상한은 `N×K`. postgres 사용 시 `store/checkpoint`의 `max_size`(기본 10)와 함께 올려야 DB가 새 병목이 되지 않는다(대략 `concurrency ≤ max_size` 권장).


```python
from klafi.server import AgentServer, create_app
server = AgentServer()
server.register(agent, agent_id="qa")
app = create_app(server)     # FastAPI: /agents/{id}/invoke · /stream · /resume · /health · OpenAPI
```

---

## 예제 (프로젝트 구조)

모든 예제는 **3-역할 분리 프로젝트**다 (단일 파일 아님). Simple/RAG/Supervisor 같은 Template 클래스는 코드로 제공된다 (`from klafi.templates import SimpleAgent, RAGAgent, SupervisorAgent`).

| 프로젝트 | 내용 |
|----------|------|
| [examples/support_platform/](examples/support_platform/) | **전 기능** — Factory·Engine·Context·Hook + Model·Tool·Skill·Memory·Guardrail·Eval·Registry·Event |

### 전 기능 프로젝트 — `support_platform/`

프레임워크 기능 전부를 **3-역할 분리 프로젝트**로 구성한 예제. 자세한 구조는 [examples/support_platform/README.md](examples/support_platform/README.md).

```bash
cd examples/support_platform && python demo.py                # CLI 데모
cd examples/support_platform && uvicorn server:app --port 8078   # HTTP 서비스 → /docs
```

`/docs`(Swagger)에서 `support`(ReAct+Tool) · `triage`(노드별 모델·툴) · `schedule`(Skill) 3종을 바로 호출할 수 있다. 에이전트별 curl 예제는 [examples/support_platform/README.md](examples/support_platform/README.md#실행).

| 필수 기능 | 이 프로젝트에서 |
|-----------|----------------|
| **Execution Factory** | `KlafiApp`(내부 `ExecutionFactory`)가 config로 model·checkpoint·store·policy·hook 주입 |
| **Execution Engine** | `config/policy.yaml`(timeout/retry/concurrency) + invoke/stream + 상태머신 |
| **Execution Context** | `ExecutionContext(user/tenant/security)` / HTTP는 `auth` 어댑터가 주입 |
| **Hook** | 공통 `MetricsHook` + `EventHook` + Logging/Tracing/Guardrail (플랫폼 전역) |
| 그 외 | Model Gateway(노드별 alias) · Tool(권한·검증) · Skill(툴+지침) · Long-Term Memory · Guardrail · Observability · Event · Evaluation · Registry |

---

## 개발

```bash
python -m pytest -q      # 186 tests
```

## 모듈 맵

```
klafi/
├─ core/          BaseGraph · ExecutionContext · AgentSpec · Exception · Hook
├─ runtime/       Execution Engine · Policy · State machine
├─ context/       Checkpoint · Long-Term Memory · Context Manager
├─ observability/ OpenTelemetry Tracing
├─ model/         Model Gateway (Alias/Token/Cost/Fallback)
├─ tool/          Tool Framework · Skill(툴+지침) · Registry
├─ guardrail/     Input/Output Guardrail (fail-close)
├─ evaluation/    Evaluator · Offline 실행 · Version 비교
├─ hitl/          Human-in-the-Loop 승인
├─ templates/     Simple · RAG · Supervisor
├─ server/        Agent Server + FastAPI 어댑터
├─ registry/      Agent Registry (Control Plane)
├─ events/        Event Bus · EventHook
└─ config/        5계층 Config 병합
```
