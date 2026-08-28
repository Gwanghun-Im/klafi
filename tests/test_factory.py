"""ExecutionFactory 검증 (요구사항 §7, F02) — KlafiGraph 클래스로 실행환경 조립·주입."""

from typing import TypedDict

from langgraph.graph import END, START

from klafi import AgentSpec, ExecutionFactory, ExecutionPolicy, KlafiGraph, ModelGateway, klafi_node
from klafi.model import FunctionProvider


class State(TypedDict):
    q: str
    a: str


class MyAgent(KlafiGraph):
    spec = AgentSpec(id="a", name="A", model="m")
    state_schema = State

    def define(self):
        @klafi_node("n")
        def n(s):
            return {"a": self.model(s["q"]) if self.model else "no-model"}

        self.add_node("n", n)
        self.add_edge(START, "n")
        self.add_edge("n", END)


def test_factory_injects_model_checkpoint_policy():
    gw = ModelGateway()
    gw.register("m", FunctionProvider(lambda p: f"r:{p}"))
    factory = ExecutionFactory(
        gateway=gw, checkpointer="memory", store="memory", policy=ExecutionPolicy(timeout=15, max_retries=1)
    )
    agent = factory.create(MyAgent)

    assert agent.checkpointer is not None  # FAC-02
    assert agent.store is not None  # FAC-04
    assert agent.policy.timeout == 15  # FAC-06
    out = agent.invoke({"q": "hi", "a": ""}, thread_id="t1")
    assert out["a"] == "r:hi"  # FAC-03 model 주입


def test_factory_without_model_passes_none():
    class NoModel(KlafiGraph):
        spec = AgentSpec(id="a", name="A")  # model alias 없음
        state_schema = State

        def define(self):
            @klafi_node("n")
            def n(s):
                return {"a": "no-model" if self.model is None else "x"}

            self.add_node("n", n)
            self.add_edge(START, "n")
            self.add_edge("n", END)

    agent = ExecutionFactory().create(NoModel)
    assert agent.invoke({"q": "x", "a": ""})["a"] == "no-model"


def test_factory_base_hooks_applied():
    from klafi import Hook

    seen = []

    class H(Hook):
        def before_agent(self, i, c):
            seen.append("h")

    ExecutionFactory(base_hooks=[H()]).create(MyAgent).invoke({"q": "x", "a": ""})
    assert "h" in seen
