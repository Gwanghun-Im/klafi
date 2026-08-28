"""Config Framework 검증 (요구사항 §22).

5계층 우선순위: Framework → Environment → Project → Agent → Runtime Override.
"""

import pytest

from klafi import ExecutionPolicy, LayeredConfig, deep_merge


# ── deep_merge ──────────────────────────────────────────────────────────
def test_deep_merge_recurses_dicts_and_replaces_scalars():
    base = {"policy": {"timeout": 60, "max_retries": 1}, "model": {"alias": "a"}}
    over = {"policy": {"timeout": 5}, "model": {"alias": "b"}}
    merged = deep_merge(base, over)
    assert merged == {"policy": {"timeout": 5, "max_retries": 1}, "model": {"alias": "b"}}
    assert base["policy"]["timeout"] == 60  # 원본 불변


# ── 계층 우선순위 (핵심) ────────────────────────────────────────────────
def test_layer_priority_order():
    cfg = LayeredConfig(
        framework={"policy": {"timeout": 300, "max_retries": 0}},
        environment={"policy": {"timeout": 60}},
        project={"policy": {"max_retries": 2}},
        agent={"policy": {"timeout": 30}},
    )
    cfg.override({"policy": {"max_retries": 5}})  # Runtime 최우선

    pol = cfg.get("policy")
    assert pol["timeout"] == 30  # agent가 env/framework 이김
    assert pol["max_retries"] == 5  # runtime이 project 이김


def test_get_dotted_and_default():
    cfg = LayeredConfig(framework={"observability": {"trace": {"backend": "otel"}}})
    assert cfg.get("observability.trace.backend") == "otel"
    assert cfg.get("observability.trace.missing", "none") == "none"
    assert cfg.get("nope.path") is None


# ── 실제 정책 해석과 연결 ───────────────────────────────────────────────
def test_resolved_config_feeds_execution_policy():
    cfg = LayeredConfig(framework={"policy": {"timeout": 300}}, agent={"policy": {"timeout": 10, "max_retries": 3}})
    pol = ExecutionPolicy.from_config(cfg.get("policy"))
    assert pol.timeout == 10 and pol.max_retries == 3


# ── 디렉터리 로딩 (§22 구조) ────────────────────────────────────────────
def test_from_dir_loads_domain_files_and_layers(tmp_path):
    (tmp_path / "framework.yaml").write_text("service: klafi\n", encoding="utf-8")
    (tmp_path / "policy.yaml").write_text("timeout: 300\nmax_retries: 0\n", encoding="utf-8")
    (tmp_path / "model.yaml").write_text("default_alias: quality-high\n", encoding="utf-8")
    (tmp_path / "environments").mkdir()
    (tmp_path / "environments" / "prod.yaml").write_text("policy:\n  timeout: 60\n", encoding="utf-8")
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "qa.yaml").write_text("policy:\n  max_retries: 3\n", encoding="utf-8")

    cfg = LayeredConfig.from_dir(tmp_path, environment="prod", agent_id="qa")

    assert cfg.get("service") == "klafi"
    assert cfg.get("model.default_alias") == "quality-high"  # model.yaml → {model: ...}
    # policy.yaml(timeout=300) < prod env(timeout=60), agent(max_retries=3) 병합
    assert cfg.get("policy.timeout") == 60
    assert cfg.get("policy.max_retries") == 3


def test_from_dir_missing_files_ok(tmp_path):
    # 파일이 하나도 없어도 빈 설정으로 안전
    cfg = LayeredConfig.from_dir(tmp_path)
    assert cfg.resolve() == {}


# ── 환경변수 치환 (SEC-05 Secret 외부화) ────────────────────────────────
def test_env_expansion_in_config(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_DSN", "postgresql://real:pw@db/x")
    (tmp_path / "framework.yaml").write_text(
        "checkpoint:\n  type: postgres\n  conn_string: ${TEST_DSN}\n", encoding="utf-8"
    )
    cfg = LayeredConfig.from_dir(tmp_path)
    assert cfg.get("checkpoint.conn_string") == "postgresql://real:pw@db/x"


def test_env_expansion_default_value(tmp_path):
    (tmp_path / "framework.yaml").write_text("checkpoint: ${NO_SUCH_VAR:memory}\n", encoding="utf-8")
    assert LayeredConfig.from_dir(tmp_path).get("checkpoint") == "memory"


def test_env_expansion_missing_raises(tmp_path):
    from klafi.core.exceptions import ConfigNotFoundError

    (tmp_path / "framework.yaml").write_text("checkpoint: ${KLAFI_ABSENT_VAR}\n", encoding="utf-8")
    with pytest.raises(ConfigNotFoundError, match="환경변수"):
        LayeredConfig.from_dir(tmp_path)
