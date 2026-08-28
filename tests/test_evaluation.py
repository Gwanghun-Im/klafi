"""Evaluation Framework 검증 (요구사항 §17, F12 / EVA-01~10).

완료조건(WS6): 동일 Agent의 Version별 품질 비교가 가능한 평가결과 구조.
"""

import pytest

from klafi import (
    AgentSpec,
    CustomEvaluator,
    LLMJudgeEvaluator,
    RuleEvaluator,
    SimpleAgent,
    run_offline,
)
from klafi.evaluation import EvalSample, EvaluationResult, InMemoryEvaluationStore


def _agent(model, version="1.0.0"):
    return SimpleAgent(model=model, spec=AgentSpec(id="qa", name="QA", version=version))


# ── Evaluator 종류 ──────────────────────────────────────────────────────
def test_rule_evaluator():
    ev = RuleEvaluator(lambda s: s.reference in s.output["answer"], metric="Task Success")
    sample = EvalSample(input={"q": 1}, output={"answer": "정답은 42"}, reference="42", agent_version="1.0.0")
    r = ev.evaluate(sample)
    assert r.passed and r.score == 1.0 and r.metric == "Task Success"
    assert r.agent_version == "1.0.0"  # EVA-10 correlation


def test_llm_judge_parses_score():
    ev = LLMJudgeEvaluator(model=lambda p: "점수: 0.9", criteria="정확성", threshold=0.5)
    r = ev.evaluate(EvalSample(input="q", output="a"))
    assert r.score == pytest.approx(0.9) and r.passed


def test_custom_evaluator_accepts_bool_float_result():
    assert CustomEvaluator(lambda s: True).evaluate(EvalSample(input=1)).score == 1.0
    assert CustomEvaluator(lambda s: 0.3).evaluate(EvalSample(input=1)).passed is False
    made = EvaluationResult("m", 0.5, True)
    assert CustomEvaluator(lambda s: made).evaluate(EvalSample(input=1)) is made


# ── Offline 실행 + 저장 + Trace 연결 ────────────────────────────────────
def test_offline_run_populates_correlation_and_store():
    agent = _agent(model=lambda p: "hello")
    store = InMemoryEvaluationStore()
    dataset = [{"input": {"question": "a"}}, {"input": {"question": "b"}}]
    ev = RuleEvaluator(lambda s: "hello" in s.output["answer"], metric="Response Quality")

    report = run_offline(agent, dataset, [ev], store=store)
    assert len(report.results) == 2
    assert all(r.passed for r in report.results)
    # EVA-09: 각 결과에 execution_id가 실려 Trace와 연결됨
    assert all(len(r.execution_id) == 32 for r in report.results)
    # EVA-08: Store 저장
    assert len(store.results) == 2


def test_summary_aggregates():
    agent = _agent(model=lambda p: "x")
    ev_pass = RuleEvaluator(lambda s: True, metric="A")
    ev_half = CustomEvaluator(lambda s: 0.5 if "1" in str(s.input) else 1.0, metric="B")
    report = run_offline(agent, [{"input": {"question": "1"}}, {"input": {"question": "2"}}], [ev_pass, ev_half])
    summ = report.summary()
    assert summ["A"]["pass_rate"] == 1.0
    assert summ["B"]["avg_score"] == pytest.approx(0.75)  # (0.5+1.0)/2


# ── Version 비교 (EVA-10, 완료조건) ─────────────────────────────────────
def test_compare_versions():
    dataset = [{"input": {"question": "hi"}}]
    ev = RuleEvaluator(lambda s: "good" in s.output["answer"], metric="quality")

    report = run_offline(_agent(lambda p: "bad", version="1.0.0"), dataset, [ev])
    for r in run_offline(_agent(lambda p: "good", version="2.0.0"), dataset, [ev]).results:
        report.add(r)

    cmp = report.compare_versions("quality")
    assert cmp["1.0.0"] == 0.0 and cmp["2.0.0"] == 1.0  # v2가 개선됨을 비교 가능
