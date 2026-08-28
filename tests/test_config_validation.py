"""Config fail-fast 검증 — 잘못된 설정은 기동(from_config) 시점에 에러."""

import pytest

from klafi import KlafiApp
from klafi.core.exceptions import (
    CheckpointException,
    ConfigException,
    KlafiException,
    ModelException,
)

BASE = {
    "framework.yaml": "service: t\ncheckpoint: memory\n",
    "model.yaml": "providers:\n  main:\n    type: echo\n",
}


def _dir(tmp_path, files):
    for name, text in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return str(tmp_path)


def test_missing_config_dir_raises():
    with pytest.raises(ConfigException, match="config 디렉터리"):
        KlafiApp.from_config("/tmp/klafi-does-not-exist-xyz")


def test_missing_environment_raises(tmp_path):
    with pytest.raises(ConfigException, match="environment"):
        KlafiApp.from_config(_dir(tmp_path, BASE), environment="staging")


def test_unknown_policy_key_raises(tmp_path):
    # 오타(timeoutt)가 조용히 무시되면 정책이 안 걸린 채 운영된다
    with pytest.raises(ConfigException, match="timeoutt"):
        KlafiApp.from_config(_dir(tmp_path, {**BASE, "policy.yaml": "timeoutt: 30\n"}))


def test_unknown_hooks_top_level_raises(tmp_path):
    with pytest.raises(ConfigException, match="최상위"):
        KlafiApp.from_config(_dir(tmp_path, {**BASE, "hooks.yaml": "alll:\n  hooks: [event]\n"}))


def test_guardrails_key_in_yaml_rejected(tmp_path):
    # 가드레일은 코드(@klafi_node/@klafi_graph)로만 적용한다. YAML에 두면 기동 시 fail-fast.
    files = {**BASE, "hooks.yaml": "all:\n  guardrails:\n    input: [pii]\n"}
    with pytest.raises(ConfigException, match="guardrails"):
        KlafiApp.from_config(_dir(tmp_path, files))


def test_unregistered_hook_name_raises(tmp_path):
    files = {**BASE, "hooks.yaml": "all:\n  hooks: [nope_hook]\n"}
    with pytest.raises(KlafiException, match="미등록"):
        KlafiApp.from_config(_dir(tmp_path, files))


def test_unknown_provider_type_raises(tmp_path):
    files = {**BASE, "model.yaml": "providers:\n  main:\n    type: gpt99\n"}
    with pytest.raises(ModelException, match="provider type"):
        KlafiApp.from_config(_dir(tmp_path, files))


def test_unknown_checkpoint_type_raises(tmp_path):
    files = {**BASE, "framework.yaml": "checkpoint: mysql\n"}
    with pytest.raises(CheckpointException, match="알 수 없는"):
        KlafiApp.from_config(_dir(tmp_path, files))


def test_agent_level_hooks_yaml_validated(tmp_path):
    files = {**BASE, "hooks.yaml": "agents:\n  qa:\n    hookz: [audit]\n"}  # hooks 오타
    with pytest.raises(ConfigException, match="agents.qa"):
        KlafiApp.from_config(_dir(tmp_path, files))


def test_valid_config_passes(tmp_path):
    files = {**BASE, "policy.yaml": "timeout: 30\n", "hooks.yaml": "all:\n  hooks: []\n"}
    app = KlafiApp.from_config(_dir(tmp_path, files))
    assert app.policy.timeout == 30
