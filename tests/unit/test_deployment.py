"""Unit tests for deployment readiness: Docker/FastAPI health, readiness, and auth."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from vaak.agent import VoiceHandler, attach_app_endpoints, create_app
from vaak.config import AgentConfig


class FakeSTT:
    def transcribe(self, audio):
        return "hello"


class FakeLLM:
    def generate(self, messages, **kwargs):
        return "world"


class FakeTTS:
    def synthesize(self, text, **kwargs):
        yield (16000, [0.0])


@pytest.fixture
def fake_handler():
    cfg = AgentConfig()
    return VoiceHandler(
        cfg,
        _stt=FakeSTT(),  # type: ignore[arg-type]
        _llm=FakeLLM(),  # type: ignore[arg-type]
        _tts=FakeTTS(),  # type: ignore[arg-type]
    )


def test_health_endpoint_returns_200(fake_handler):
    app = create_app(fake_handler)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_attach_app_endpoints_directly(fake_handler):
    from fastapi import FastAPI

    app = FastAPI()
    attach_app_endpoints(app, fake_handler)
    client = TestClient(app)
    assert client.get("/health").status_code == 200


def test_ready_endpoint_success(fake_handler):
    app = create_app(fake_handler)
    client = TestClient(app)

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}


def test_ready_endpoint_ollama_unreachable(fake_handler):
    app = create_app(fake_handler)
    client = TestClient(app)

    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        response = client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not ready"
        assert "connection refused" in data["error"]


def test_auth_disabled_by_default(fake_handler):
    app = create_app(fake_handler)
    client = TestClient(app)

    # Health and public endpoints return 200
    assert client.get("/health").status_code == 200


def test_auth_token_enforcement():
    cfg = AgentConfig(auth_token="secret-token-123")
    handler = VoiceHandler(
        cfg,
        _stt=FakeSTT(),  # type: ignore[arg-type]
        _llm=FakeLLM(),  # type: ignore[arg-type]
        _tts=FakeTTS(),  # type: ignore[arg-type]
    )
    app = create_app(handler)

    # Add a dummy protected route to test auth
    @app.get("/protected")
    def protected():
        return {"data": "secure"}

    client = TestClient(app)

    # Public routes bypass auth
    assert client.get("/health").status_code == 200

    # Missing auth on protected route -> 401
    resp_no_auth = client.get("/protected")
    assert resp_no_auth.status_code == 401
    assert resp_no_auth.json() == {"error": "unauthorized"}

    # Wrong token -> 401
    resp_wrong = client.get("/protected", headers={"Authorization": "Bearer bad-token"})
    assert resp_wrong.status_code == 401
    assert resp_wrong.json() == {"error": "unauthorized"}

    # Valid token -> 200
    resp_ok = client.get("/protected", headers={"Authorization": "Bearer secret-token-123"})
    assert resp_ok.status_code == 200
    assert resp_ok.json() == {"data": "secure"}


def test_audit_log_defaults_to_false():
    cfg = AgentConfig()
    assert cfg.audit_log is False


def test_auth_and_audit_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("VAAK_AUTH_TOKEN", "env-secret")
    monkeypatch.setenv("VAAK_AUDIT_LOG", "true")

    from vaak.config import load_config

    cfg = load_config(config_path=tmp_path / "none.yaml")
    assert cfg.auth_token == "env-secret"
    assert cfg.audit_log is True
