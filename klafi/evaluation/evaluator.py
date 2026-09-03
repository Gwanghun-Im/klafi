"""Evaluation Framework (요구사항 §17, F12 / EVA-01~10).

Langfuse/LangSmith에 완전히 의존하지 않는다. KLAFI가 Evaluator Interface를 정의하고
실제 Evaluator는 Rule / LLM Judge / Custom 또는 프로젝트 구현으로 연결한다.

Execution Result → Evaluator → EvaluationResult → Store(→ Langfuse/LangSmith/DB)

- 결과에 execution_id/agent_version을 실어 Trace 연결(EVA-09)과 Version 비교(EVA-10)를 가능케 한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Protocol

from klafi.core.context import ExecutionContext
from klafi.observability.tracing import span


@dataclass
class EvalSample:
    input: Any
    output: Any = None
    reference: Any = None  # 정답/기대값 (있으면 Rule 평가에 사용)
    execution_id: str | None = None
    agent_id: str | None = None
    agent_version: str | None = None
    latency_ms: float | None = None
    tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    metric: str  # Task Success / Groundedness / Response Quality ...
    score: float  # 0.0 ~ 1.0
    passed: bool
    reason: str | None = None
    execution_id: str | None = None  # EVA-09: Trace 연결
    agent_id: str | None = None
    agent_version: str | None = None  # EVA-10: Version 비교


class Evaluator:
    metric: str = "eval"

    def evaluate(self, sample: EvalSample) -> EvaluationResult:  # pragma: no cover
        raise NotImplementedError

    def _result(self, sample: EvalSample, score: float, passed: bool, reason: str | None = None) -> EvaluationResult:
        return EvaluationResult(
            metric=self.metric,
            score=score,
            passed=passed,
            reason=reason,
            execution_id=sample.execution_id,
            agent_id=sample.agent_id,
            agent_version=sample.agent_version,
        )


class RuleEvaluator(Evaluator):
    """규칙(predicate) 기반 평가 (EVA-02). 예: 출력이 reference를 포함하는가."""

    def __init__(self, rule: Callable[[EvalSample], bool], metric: str = "rule") -> None:
        self.metric = metric
        self._rule = rule

    def evaluate(self, sample: EvalSample) -> EvaluationResult:
        ok = bool(self._rule(sample))
        return self._result(sample, 1.0 if ok else 0.0, ok)


def _parse_score(text: str) -> float:
    """심판 응답에서 점수 하나를 뽑는다. JSON {"score": x} > 분수 a/b > 마지막 숫자(점수는 보통 말미).
    1<x<=10 이면 10점 척도로 보고 /10. 첫 숫자를 취하던 이전 구현은 '2 errors… score 0.1' 을 1.0 으로 읽었다."""
    s = str(text)
    m = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', s)
    if m:
        val = float(m.group(1))
    else:
        frac = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", s)
        if frac and float(frac.group(2)) > 0:
            return max(0.0, min(1.0, float(frac.group(1)) / float(frac.group(2))))
        nums = re.findall(r"\d+(?:\.\d+)?", s)
        if not nums:
            return 0.0
        val = float(nums[-1])
    if 1.0 < val <= 10.0:
        val /= 10.0
    return max(0.0, min(1.0, val))


class LLMJudgeEvaluator(Evaluator):
    """LLM을 심판으로 사용 (EVA-03). model은 Model Gateway의 (prompt)->str 콜러블 재사용."""

    def __init__(self, model: Callable[[str], str], criteria: str, threshold: float = 0.5, metric: str = "llm_judge") -> None:
        self.metric = metric
        self._model = model
        self._criteria = criteria
        self._threshold = threshold

    def evaluate(self, sample: EvalSample) -> EvaluationResult:
        prompt = (
            f"평가 기준: {self._criteria}\n"
            f"입력: {sample.input}\n출력: {sample.output}\n"
            f"참고: {sample.reference}\n"
            "0.0~1.0 사이 점수만 숫자로 답하라."
        )
        raw = self._model(prompt)
        score = _parse_score(raw)
        return self._result(sample, score, score >= self._threshold, reason=str(raw))


class CustomEvaluator(Evaluator):
    """임의 함수 래핑 (EVA-04). fn은 bool/float/EvaluationResult 중 무엇이든 반환 가능."""

    def __init__(self, fn: Callable[[EvalSample], Any], metric: str = "custom") -> None:
        self.metric = metric
        self._fn = fn

    def evaluate(self, sample: EvalSample) -> EvaluationResult:
        out = self._fn(sample)
        if isinstance(out, EvaluationResult):
            return out
        if isinstance(out, bool):
            return self._result(sample, 1.0 if out else 0.0, out)
        score = max(0.0, min(1.0, float(out)))
        return self._result(sample, score, score >= 0.5)


# ── 결과 저장 (EVA-08) ──────────────────────────────────────────────────
class EvaluationStore(Protocol):
    def save(self, result: EvaluationResult) -> None: ...


class InMemoryEvaluationStore:
    def __init__(self) -> None:
        self.results: list[EvaluationResult] = []

    def save(self, result: EvaluationResult) -> None:
        self.results.append(result)


# ── 집계/비교 (EVA-10) ──────────────────────────────────────────────────
class EvaluationReport:
    def __init__(self) -> None:
        self.results: list[EvaluationResult] = []

    def add(self, result: EvaluationResult) -> None:
        self.results.append(result)

    def summary(self) -> dict[str, dict[str, float]]:
        """metric별 평균 score와 pass율."""
        out: dict[str, dict[str, float]] = {}
        for metric in {r.metric for r in self.results}:
            rs = [r for r in self.results if r.metric == metric]
            out[metric] = {
                "avg_score": sum(r.score for r in rs) / len(rs),
                "pass_rate": sum(r.passed for r in rs) / len(rs),
            }
        return out

    def compare_versions(self, metric: str) -> dict[str, float]:
        """동일 metric에 대해 agent_version별 평균 score (EVA-10)."""
        out: dict[str, float] = {}
        for version in {r.agent_version for r in self.results if r.metric == metric}:
            rs = [r for r in self.results if r.metric == metric and r.agent_version == version]
            out[str(version)] = sum(r.score for r in rs) / len(rs)
        return out


# ── Offline 실행 (EVA-05) ───────────────────────────────────────────────
def run_offline(
    agent: Any,
    dataset: list[dict[str, Any]],
    evaluators: list[Evaluator],
    store: EvaluationStore | None = None,
    security_context: dict[str, Any] | None = None,
) -> EvaluationReport:
    """dataset의 각 input으로 agent를 실행하고 evaluators로 평가한다.

    dataset item: {"input": ..., "reference": (선택)}
    security_context: 실행에 필요한 권한 등(예: Tool 권한)을 각 실행 Context에 주입.
    """
    report = EvaluationReport()
    for item in dataset:
        ctx = ExecutionContext.new(
            agent_id=agent.spec.id,
            agent_version=agent.spec.version,
            security_context=security_context or {},
        )
        t0 = perf_counter()
        output = agent.invoke(item["input"], context=ctx)
        latency = (perf_counter() - t0) * 1000

        sample = EvalSample(
            input=item["input"],
            output=output,
            reference=item.get("reference"),
            execution_id=ctx.execution_id,
            agent_id=ctx.agent_id,
            agent_version=ctx.agent_version,
            latency_ms=latency,
        )
        for ev in evaluators:
            res = ev.evaluate(sample)
            # EVA-09: 평가를 Trace와 연결 (execution_id를 span에 실어 관측계와 상관)
            with span(f"eval.{res.metric}") as sp:
                sp.set_attribute("klafi.execution_id", res.execution_id or "-")
                sp.set_attribute("klafi.eval_score", res.score)
                sp.set_attribute("klafi.eval_passed", res.passed)
            report.add(res)
            if store is not None:
                store.save(res)
    return report
