"""WS4 Hook 엔진 검증 (요구사항 §11, F06).

핵심 DoD: 개발자가 Node에 로깅 코드를 안 써도 Node 실행 로그가 자동 생성된다.
추가: Before/After/Error/Finally 순서, onion 순서, fail-open/close.
"""

import logging

import pytest
from langgraph.graph import END, START, StateGraph
from typing import TypedDict

from klafi import AgentSpec, BaseGraph, Hook, LoggingHook, clear_hooks, register_hook
from klafi.core.exceptions import GuardrailException


class State(TypedDict):
    text: str


class OneNode(BaseGraph):
    def build(self) -> StateGraph:
        g = StateGraph(State)
        g.add_node("work", lambda s: {"text": s["text"] + "!"})
        g.add_edge(START, "work")
        g.add_edge("work", END)
        return g


@pytest.fixture(autouse=True)
def _reset():
    clear_hooks()
    yield
    clear_hooks()


class Recorder(Hook):
    def __init__(self, tag, priority=100):
        self.tag = tag
        self.priority = priority
        self.events = []

    def before_node(self, node, state, ctx):
        self.events.append(f"before:{self.tag}")

    def after_node(self, node, state, result, ctx):
        self.events.append(f"after:{self.tag}")

    def finally_node(self, node, state, ctx):
        self.events.append(f"finally:{self.tag}")

    def on_node_error(self, node, state, exc, ctx):
        self.events.append(f"error:{self.tag}")


def _agent(hooks=None):
    return OneNode(AgentSpec(id="a", name="A"), hooks=hooks)


def test_auto_logging_without_node_code(caplog):
    # Node 코드에는 로깅이 전혀 없다. LoggingHook만 붙인다.
    with caplog.at_level(logging.INFO, logger="klafi.node"):
        out = _agent(hooks=[LoggingHook()]).invoke({"text": "hi"})
    assert out["text"] == "hi!"
    msgs = [r.message for r in caplog.records]
    assert any("node.start" in m and "node=work" in m for m in msgs)
    assert any("node.end" in m and "node=work" in m for m in msgs)
    # execution_id가 로그에 correlation으로 찍힌다
    assert all("execution_id=" in m for m in msgs)


def test_onion_order():
    r = Recorder("x")
    _agent(hooks=[r]).invoke({"text": "z"})
    assert r.events == ["before:x", "after:x", "finally:x"]


def test_priority_before_ascending_after_reversed():
    r = Recorder("shared")
    a = Recorder("lo", priority=10)
    b = Recorder("hi", priority=90)
    a.events = b.events = shared = []
    _agent(hooks=[b, a]).invoke({"text": "z"})  # 등록 순서 무관, priority로 정렬
    # before는 priority 오름차순(lo→hi), after/finally는 역순(hi→lo)
    assert shared == ["before:lo", "before:hi", "after:hi", "after:lo", "finally:hi", "finally:lo"]


def test_global_hook_applies():
    r = Recorder("g")
    register_hook(r)
    _agent().invoke({"text": "z"})  # Agent에 직접 안 붙여도 전역 Hook 발화
    assert "before:g" in r.events


def test_error_hook_fires_and_reraises():
    class Boom(BaseGraph):
        def build(self):
            g = StateGraph(State)
            def bad(s):
                raise ValueError("boom")
            g.add_node("bad", bad)
            g.add_edge(START, "bad")
            g.add_edge("bad", END)
            return g

    r = Recorder("e")
    with pytest.raises(ValueError):
        Boom(AgentSpec(id="b", name="B"), hooks=[r]).invoke({"text": "z"})
    assert r.events == ["before:e", "error:e", "finally:e"]  # after 없음


def test_fail_open_hook_does_not_break_agent():
    class NoisyLogger(Hook):
        fail_open = True
        def before_node(self, node, state, ctx):
            raise RuntimeError("logging backend down")

    out = _agent(hooks=[NoisyLogger()]).invoke({"text": "hi"})
    assert out["text"] == "hi!"  # 로깅 Hook 장애가 업무를 막지 않음


def test_fail_close_guardrail_blocks():
    class Guard(Hook):
        fail_open = False
        def before_node(self, node, state, ctx):
            raise GuardrailException("blocked input")

    with pytest.raises(GuardrailException):
        _agent(hooks=[Guard()]).invoke({"text": "hi"})


def test_disabled_hook_skipped():
    r = Recorder("d")
    r.enabled = False
    _agent(hooks=[r]).invoke({"text": "z"})
    assert r.events == []
