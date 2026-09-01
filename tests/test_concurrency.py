"""서버 동시 실행 상한(concurrency) — 초과 시 429 (백프레셔)."""

import threading
import time
from typing import TypedDict

from fastapi.testclient import TestClient
from langgraph.graph import END, START

from klafi.core import AgentSpec, KlafiGraph, klafi_node
from klafi.runtime import ExecutionPolicy
from klafi.server import AgentServer, create_app


class State(TypedDict):
    n: int


def _server(policy=None):
    class Slow(KlafiGraph):
        spec = AgentSpec(id="slow", name="Slow")
        state_schema = State
        observability = False

        def define(self):
            @klafi_node("wait")
            def wait(s):
                time.sleep(0.4)  # 슬롯 점유 유지
                return {"n": s["n"] + 1}

            self.add_node("wait", wait)
            self.add_edge(START, "wait")
            self.add_edge("wait", END)

    srv = AgentServer()
    srv.register(Slow(policy=policy), agent_id="slow")
    return srv


def test_exceeding_concurrency_returns_429():
    app = create_app(_server(), max_concurrency=2)
    client = TestClient(app)
    codes: list[int] = []

    def hit():
        codes.append(client.post("/agents/slow/invoke", json={"input": {"n": 0}}).status_code)

    threads = [threading.Thread(target=hit) for _ in range(5)]  # 5개 동시 → 2 통과, 나머지 429
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert codes.count(200) == 2  # 슬롯 2개만 실행
    assert codes.count(429) == 3  # 초과 3개는 즉시 거절
    assert all(c in (200, 429) for c in codes)


def test_slot_released_after_request():
    app = create_app(_server(), max_concurrency=1)
    client = TestClient(app)
    # 순차 호출은 슬롯이 반납되므로 모두 200
    assert client.post("/agents/slow/invoke", json={"input": {"n": 0}}).status_code == 200
    assert client.post("/agents/slow/invoke", json={"input": {"n": 0}}).status_code == 200


def test_no_limit_when_unset():
    app = create_app(_server(), max_concurrency=None)  # 무제한
    client = TestClient(app)
    codes: list[int] = []

    def hit():
        codes.append(client.post("/agents/slow/invoke", json={"input": {"n": 0}}).status_code)

    threads = [threading.Thread(target=hit) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert codes.count(200) == 4  # 상한 없으면 전부 통과


def test_policy_from_config_carries_concurrency():
    p = ExecutionPolicy.from_config({"timeout": 30, "concurrency": 5})
    assert p.concurrency == 5


# ── 스트리밍도 슬롯을 점유한다 (미들웨어 리팩터 회귀 방지) ───────────────────
def test_stream_holds_slot_until_finished():
    """스트림이 진행 중인 동안 슬롯을 잡고 있어야 한다 → 초과 요청은 429.
    (엔드포인트별 세마포어를 ASGI 미들웨어로 옮기며 가장 깨지기 쉬운 지점.)"""
    from fastapi.testclient import TestClient

    client = TestClient(create_app(_server(), max_concurrency=1))
    during: dict[str, int] = {}

    def consume_stream():
        with client.stream("POST", "/agents/slow/stream", json={"input": {"n": 0}}) as r:
            for _ in r.iter_lines():
                pass

    def invoke_mid():
        time.sleep(0.2)  # 스트림이 아직 도는 시점
        during["code"] = client.post("/agents/slow/invoke", json={"input": {"n": 0}}).status_code

    t1 = threading.Thread(target=consume_stream)
    t2 = threading.Thread(target=invoke_mid)
    t1.start(); t2.start(); t1.join(); t2.join()

    assert during["code"] == 429  # 스트림이 유일 슬롯을 점유 중
    # 스트림이 끝났으면 슬롯이 반납되어 다시 200
    time.sleep(0.1)
    assert client.post("/agents/slow/invoke", json={"input": {"n": 0}}).status_code == 200


def test_readonly_endpoints_not_throttled():
    """health·목록 조회는 슬롯을 세지 않는다 (백프레셔 대상은 실행 계열만)."""
    from fastapi.testclient import TestClient

    client = TestClient(create_app(_server(), max_concurrency=1))
    # 상한이 1이어도 조회는 항상 통과
    for _ in range(5):
        assert client.get("/health").status_code == 200
        assert client.get("/agents").status_code == 200


# ── per-agent concurrency (2단계: 전역 + 에이전트별) ────────────────────────
def _slow_agent(aid, policy=None):
    class Slow(KlafiGraph):
        spec = AgentSpec(id=aid, name=aid)
        state_schema = State
        observability = False

        def define(self):
            @klafi_node("wait")
            def wait(s):
                time.sleep(0.4)
                return {"n": s["n"] + 1}

            self.add_node("wait", wait)
            self.add_edge(START, "wait")
            self.add_edge("wait", END)

    return Slow(policy=policy)


def _multi_server(*ids):
    srv = AgentServer()
    for aid in ids:
        srv.register(_slow_agent(aid), agent_id=aid)
    return srv


def _fire(client, calls):
    """calls: [(agent_id, key), ...] 동시 실행. {key: status_code} 반환."""
    out: dict = {}
    lock = threading.Lock()

    def hit(aid, key):
        code = client.post(f"/agents/{aid}/invoke", json={"input": {"n": 0}}).status_code
        with lock:
            out[key] = code

    threads = [threading.Thread(target=hit, args=(aid, key)) for aid, key in calls]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def test_per_agent_concurrency_isolates():
    app = create_app(_multi_server("a", "b"), per_agent_concurrency={"a": 1, "b": 3})
    client = TestClient(app)
    r = _fire(client, [("a", "a1"), ("a", "a2"), ("b", "b1")])
    assert sorted([r["a1"], r["a2"]]) == [200, 429]  # a 는 캡 1 → 하나만
    assert r["b1"] == 200  # b 는 격리 — a 포화와 무관


def test_two_level_agent_cap_binds_before_global():
    # 전역 2 여유가 있어도 에이전트 캡 1 이 먼저 막는다.
    app = create_app(_multi_server("a", "b"), max_concurrency=2, per_agent_concurrency={"a": 1})
    client = TestClient(app)
    r = _fire(client, [("a", "a0"), ("a", "a1")])
    assert sorted([r["a0"], r["a1"]]) == [200, 429]


def test_two_level_global_cap_applies_to_agent_without_own_limit():
    # b 는 per-agent 캡 없음 → 전역 총량 2 만 적용.
    app = create_app(_multi_server("a", "b"), max_concurrency=2, per_agent_concurrency={"a": 1})
    client = TestClient(app)
    r = _fire(client, [("b", "b0"), ("b", "b1"), ("b", "b2")])
    codes = [r["b0"], r["b1"], r["b2"]]
    assert codes.count(200) == 2 and codes.count(429) == 1  # 전역 캡 2


def test_policy_merge_overrides_only_given_keys():
    base = ExecutionPolicy(timeout=30, max_retries=1, concurrency=4, backoff_base=0.5)
    m = base.merge({"concurrency": 1, "timeout": 120})
    assert m.concurrency == 1 and m.timeout == 120
    assert m.max_retries == 1 and m.backoff_base == 0.5  # 명시 안 한 키는 상속
    assert base.concurrency == 4  # 원본 불변
    assert base.merge(None) is base  # 빈 override → self
