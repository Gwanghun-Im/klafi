"""OTLP exporter 결정 — Intelligence(ON) > 로컬 config > 표준 env > 없음, 전 분기 fail-open.

실제 네트워크·exporter 없이 검증: urlopen 은 가짜로, _make_exporter 는 conf 를 캡처하는
스텁으로 패치해 '어느 소스가 이겼는지'를 확인한다.
"""

import io
import json

import pytest

import klafi.observability.tracing as tr


class _Cfg:
    def __init__(self, otlp=None):
        self._otlp = otlp

    def get(self, key, default=None):
        return self._otlp if key == "observability.otlp" else default


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(tr, "_otlp_attached", False)  # 프로세스당 1회 가드 리셋
    for k in ("INTELLIGENCE_MODE", "INTELLIGENCE_ENDPOINT", "INTELLIGENCE_TOKEN",
              "OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        monkeypatch.delenv(k, raising=False)


def _capture_maker(monkeypatch, seen):
    monkeypatch.setattr(tr, "_make_exporter", lambda conf: seen.append(conf) or object())


def _fake_urlopen(monkeypatch, payload=None, exc=None):
    def urlopen(req, timeout=None):
        if exc:
            raise exc

        class R(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return R(json.dumps(payload).encode())

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)


def test_intelligence_wins_when_on_and_healthy(monkeypatch):
    seen = []
    _capture_maker(monkeypatch, seen)
    monkeypatch.setenv("INTELLIGENCE_MODE", "ON")
    monkeypatch.setenv("INTELLIGENCE_ENDPOINT", "https://intel.internal")
    _fake_urlopen(monkeypatch, {"endpoint": "https://col:4318/v1/traces", "headers": {"authorization": "Bearer x"}})
    assert tr.resolve_otlp_exporter(_Cfg({"endpoint": "https://local:4318/v1/traces"})) is not None
    assert seen[0]["source"] == "intelligence" and seen[0]["endpoint"].startswith("https://col")


def test_on_but_down_falls_back_to_local_config(monkeypatch):
    seen = []
    _capture_maker(monkeypatch, seen)
    monkeypatch.setenv("INTELLIGENCE_MODE", "ON")
    monkeypatch.setenv("INTELLIGENCE_ENDPOINT", "https://intel.internal")
    _fake_urlopen(monkeypatch, exc=TimeoutError("down"))
    tr.resolve_otlp_exporter(_Cfg({"endpoint": "https://local:4318/v1/traces"}))
    assert seen[0]["source"] == "local-config"  # ON 이 로컬을 무효화하지 않는다


def test_bad_endpoint_url_still_falls_back(monkeypatch):
    """스킴 없는 URL 은 Request 생성자에서 터진다 — 그래도 로컬 폴백이 살아야 한다."""
    seen = []
    _capture_maker(monkeypatch, seen)
    monkeypatch.setenv("INTELLIGENCE_MODE", "ON")
    monkeypatch.setenv("INTELLIGENCE_ENDPOINT", "intel.internal")  # 스킴 없음 → ValueError
    tr.resolve_otlp_exporter(_Cfg({"endpoint": "https://local:4318/v1/traces"}))
    assert seen and seen[0]["source"] == "local-config"


def test_off_uses_local_then_otel_env(monkeypatch):
    seen = []
    _capture_maker(monkeypatch, seen)
    tr.resolve_otlp_exporter(_Cfg({"endpoint": "https://local:4318/v1/traces"}))
    assert seen[0]["source"] == "local-config"
    monkeypatch.setattr(tr, "_otlp_attached", False)
    seen.clear()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "https://col:4318/v1/traces")
    tr.resolve_otlp_exporter(_Cfg(None))
    assert seen[0]["source"] == "otel-env" and seen[0]["endpoint"] is None  # SDK 표준 처리에 위임


def test_nothing_configured_is_quiet_none(monkeypatch):
    assert tr.resolve_otlp_exporter(_Cfg(None)) is None  # 예외 0건 — 계측만


def test_attached_guard_prevents_duplicate_processor(monkeypatch):
    seen = []
    _capture_maker(monkeypatch, seen)
    cfg = _Cfg({"endpoint": "https://local:4318/v1/traces"})
    assert tr.resolve_otlp_exporter(cfg) is not None
    assert tr.resolve_otlp_exporter(cfg) is None  # 재호출(테스트·멀티앱) → 중복 부착 금지
    assert len(seen) == 1


def test_empty_header_values_are_dropped(monkeypatch):
    """${VAR:} 빈 기본값 헤더는 제외 — 빈 'Bearer ' 헤더 송출 방지."""
    seen = []
    _capture_maker(monkeypatch, seen)
    tr.resolve_otlp_exporter(_Cfg({"endpoint": "https://l:4318/v1/traces", "headers": {"authorization": ""}}))
    assert seen[0]["headers"] == {}


def test_logs_never_contain_secrets(monkeypatch, caplog):
    """endpoint 의 userinfo·query 토큰, 헤더 값이 로그에 남지 않는다."""
    import logging

    monkeypatch.setenv("INTELLIGENCE_MODE", "ON")
    monkeypatch.setenv("INTELLIGENCE_ENDPOINT", "https://intel.internal")
    _fake_urlopen(monkeypatch, {"endpoint": "https://pk:sk@col:4318/v1/traces?api-key=SECRET",
                                "headers": {"authorization": "Bearer TOPSECRET"}})
    seen = []
    _capture_maker(monkeypatch, seen)
    with caplog.at_level(logging.INFO, logger="klafi.observability"):
        tr.resolve_otlp_exporter(_Cfg(None))
    text = caplog.text
    assert "TOPSECRET" not in text and "SECRET" not in text and "pk:sk" not in text
    assert "source=intelligence" in text  # 어느 소스가 이겼는지는 남는다
