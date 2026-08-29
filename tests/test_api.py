import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

def test_predict_endpoint():
    resp = client.post("/api/predict", json={
        "session_id": "2024_Bahrain_FP2",
        "driver_code": "VER",
        "compound": "MEDIUM",
        "stint_length": 20
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["driver_code"] == "VER"
    assert data["compound"] == "MEDIUM"
    assert len(data["curve"]) == 20

def test_strategy_endpoint():
    resp = client.post("/api/strategy", json={
        "session_id": "2024_Bahrain_FP2",
        "driver_code": "VER",
        "current_lap": 12,
        "current_compound": "SOFT",
        "tyre_age": 12,
        "target_race_laps": 57
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "recommended_strategy" in data
