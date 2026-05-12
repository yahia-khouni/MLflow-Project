"""
test_model_info.py — Model info endpoint tests
"""


def test_model_info_returns_200(client, mock_model_service):
    response = client.get("/model/info")
    assert response.status_code == 200


def test_model_info_contains_version(client, mock_model_service):
    response = client.get("/model/info")
    data = response.json()
    assert "model_version" in data
    assert data["model_version"] == "1"


def test_model_info_contains_training_metrics(client, mock_model_service):
    response = client.get("/model/info")
    data = response.json()
    assert "training_metrics" in data
    assert "auc_roc" in data["training_metrics"]
    assert "accuracy" in data["training_metrics"]


def test_model_info_contains_algorithm(client, mock_model_service):
    response = client.get("/model/info")
    data = response.json()
    assert data["algorithm"] == "Random Forest"


def test_model_info_contains_load_timestamp(client, mock_model_service):
    response = client.get("/model/info")
    data = response.json()
    assert "load_timestamp" in data
