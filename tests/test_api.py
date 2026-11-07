import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from deployment.app import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))      # aucun modele disponible
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_models_empty(client):
    assert client.get("/models").json() == []


def test_predict_unknown_model_is_404(client):
    r = client.post("/predict?model=cnn", files={"file": ("a.png", b"x", "image/png")})
    assert r.status_code == 404
