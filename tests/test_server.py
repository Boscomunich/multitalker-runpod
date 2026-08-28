import pytest
from fastapi.testclient import TestClient

pytest.importorskip("multipart")


def test_health_without_model(monkeypatch):
    monkeypatch.setenv("MULTITALKER_API_KEY", "test")
    import server
    server.service.ready = False
    server.service.load = lambda: None
    with TestClient(server.app, raise_server_exceptions=False) as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["models_loaded"] is False


def test_transcribe_requires_auth(monkeypatch):
    monkeypatch.setenv("MULTITALKER_API_KEY", "test")
    import server
    with TestClient(server.app, raise_server_exceptions=False) as client:
        response = client.post("/transcribe", files={"file": ("test.wav", b"RIFF")})
    assert response.status_code == 401