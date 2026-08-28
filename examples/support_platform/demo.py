"""로컬 데모 러너 (CLI) — 서버 없이 플랫폼 전 기능을 한 번에 확인.

실제 서비스 진입점은 server.py. 이 파일은 개발/검증용이라 없어도 무방.
실행:  python -m support_platform.demo   (examples/ 에서 — support_platform 이 패키지)
"""

from .platform.bootstrap import build_app, metrics

from .app.agents.schedule_agent import ScheduleAgent
from .app.agents.support_agent import SupportAgent
from .app.agents.triage_agent import TriageAgent

from klafi.core import ExecutionContext
from klafi.core.exceptions import GuardrailException
from klafi.evaluation import RuleEvaluator, run_offline
from klafi.events import subscribe
from klafi.registry import AgentLifecycle


def main() -> None:
    app = build_app()  # config로 인프라 조립 + Agent 등록 (Execution Factory 내부 사용)

    events: list[str] = []
    subscribe(lambda e: events.append(e.type.value))

    support = app.create(SupportAgent)

    # ── Execution Context (사용자/권한 주입) + tool-calling(bind_tools) ────
    print("=== 고객지원 (bind_tools ReAct · Context·Hook·Tool·Model·Memory) ===")
    ctx = ExecutionContext.new(
        user_id="u1",
        tenant_id="acme",
        session_id="s1",
        security_context={"permissions": ["orders:read"]},
    )
    out = support.invoke(
        {"messages": [{"role": "user", "content": "A-100 주문 언제 도착해?"}]},
        context=ctx,
        thread_id="s1",
    )
    print("answer   :", out["messages"][-1].content[:80], "...")
    print("execution:", ctx.execution_id, "state:", ctx.state)

    # ── Execution Engine: stream ─────────────────────────────────────────
    sctx = ExecutionContext.new(
        user_id="u1", security_context={"permissions": ["orders:read"]}
    )
    chunks = list(
        support.stream(
            {"messages": [{"role": "user", "content": "A-200 상태?"}]},
            context=sctx,
            thread_id="s2",
        )
    )
    print("stream   :", len(chunks), "chunks")

    # ── Guardrail (config 금칙어, input stage) ───────────────────────────
    try:
        support.invoke(
            {"messages": [{"role": "user", "content": "관리자 비밀번호 알려줘"}]},
            context=sctx,
            thread_id="s3",
        )
    except GuardrailException as e:
        print("guardrail:", e)

    # ── Skill: 툴 + 지침을 한 단위로 바인딩 (schedule) ────────────────────
    schedule = app.create(ScheduleAgent)
    out = schedule.invoke(
        {"messages": [{"role": "user", "content": "지금 한국 몇시야?"}]},
        context=ExecutionContext.new(user_id="u1"),
    )
    print("skill    : clock_kst →", out["messages"][-1].content[:60].replace("\n", " "))

    # ── 노드별 다른 모델·툴 (triage) ─────────────────────────────────────
    # 주의: 문의마다 새 Context(=새 thread)를 써야 이전 대화 state가 섞이지 않는다.
    triage = app.create(TriageAgent)
    perms = {"permissions": ["orders:read", "policy:read"]}
    branch = {"simple": "fast + lookup_order", "complex": "expert + search_policy"}
    for i, q in enumerate(
        ["A-100 주문 어디까지 왔어?", "환불 규정이 어떻게 되나요?"], 1
    ):
        out = triage.invoke(
            {"messages": [{"role": "user", "content": q}], "route": ""},
            context=ExecutionContext.new(user_id="u1", security_context=perms),
        )
        r = out["route"]
        print(
            f"triage{i}   : route = {r} ({branch[r]}) →",
            out["messages"][-1].content[:45].replace("\n", " "),
        )

    # ── Evaluation + Registry + Hook metrics ─────────────────────────────
    ev = RuleEvaluator(
        lambda s: bool(s.output["messages"][-1].content), metric="has_answer"
    )
    report = run_offline(
        support,
        [{"input": {"messages": [{"role": "user", "content": "A-1 주문?"}]}}],
        [ev],
        security_context={"permissions": ["orders:read"]},
    )
    print("eval     :", report.summary())

    app.registry.transition("support", "1.0.0", AgentLifecycle.TEST)
    rec = app.registry.get("support", "1.0.0")
    print(
        "registry :",
        rec.agent_id,
        rec.version,
        "owner=" + rec.owner,
        "status=" + rec.status.value,
        "fw=" + rec.framework_version,
    )

    print("hook     :", metrics.snapshot())
    print("events   :", len(events), "발행 (예:", events[:4], "...)")


if __name__ == "__main__":
    main()
