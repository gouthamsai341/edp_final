"""
tests/test_api.py

Covers: health endpoint, disease library endpoint, history endpoint,
and the predict endpoint's handling of a missing/invalid image
(exercised without needing a trained model on disk).
"""

import io


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "model_ready" in data


def test_diseases_endpoint_returns_library(client):
    res = client.get("/api/diseases")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert isinstance(data["diseases"], dict)
    assert len(data["diseases"]) > 0


def test_disease_detail_not_found(client):
    res = client.get("/api/diseases/Not_A_Real_Class")
    assert res.status_code == 404


def test_history_endpoint_empty_initially(client):
    res = client.get("/api/history")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["history"] == []


def test_predict_rejects_missing_image(client):
    res = client.post("/api/predict", data={}, content_type="multipart/form-data")
    data = res.get_json()
    assert res.status_code == 400
    assert data["success"] is False


def test_predict_rejects_unsupported_extension(client):
    data = {"image": (io.BytesIO(b"not an image"), "malware.exe")}
    res = client.post("/api/predict", data=data, content_type="multipart/form-data")
    body = res.get_json()
    assert res.status_code == 400
    assert body["success"] is False


def test_home_and_pages_render(client):
    for path in ["/", "/detect", "/history", "/library", "/about"]:
        res = client.get(path)
        assert res.status_code == 200, f"{path} failed"
