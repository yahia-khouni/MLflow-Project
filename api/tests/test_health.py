"""
test_health.py — Health endpoint tests
"""
from tests.conftest import VALID_CUSTOMER


def test_health_returns_200(client, mock_model_service):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_contains_model_version(client, mock_model_service):
    response = client.get("/health")
    data = response.json()
    assert "model_version" in data
    assert data["model_version"] == "1"


def test_health_contains_uptime(client, mock_model_service):
    response = client.get("/health")
    data = response.json()
    assert "uptime_seconds" in data
    assert isinstance(data["uptime_seconds"], float)
    assert data["uptime_seconds"] >= 0


def test_health_status_healthy_when_model_loaded(client, mock_model_service):
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["mlflow_connected"] is True


def test_health_contains_timestamp(client, mock_model_service):
    response = client.get("/health")
    data = response.json()
    assert "timestamp" in data
