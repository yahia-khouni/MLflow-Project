"""
test_batch.py — Batch prediction endpoint tests
"""
import copy
from tests.conftest import VALID_CUSTOMER


def _make_batch(n: int) -> dict:
    """Helper: create a batch request with n identical customers."""
    return {"customers": [copy.deepcopy(VALID_CUSTOMER) for _ in range(n)]}


def test_batch_predict_10_records_returns_200(client, mock_model_service):
    response = client.post("/predict/batch", json=_make_batch(10))
    assert response.status_code == 200


def test_batch_predict_1000_records_returns_200(client, mock_model_service):
    response = client.post("/predict/batch", json=_make_batch(1000))
    assert response.status_code == 200


def test_batch_predict_1001_records_returns_422(client, mock_model_service):
    # Exceeds max_length=1000 on the customers list
    response = client.post("/predict/batch", json=_make_batch(1001))
    assert response.status_code == 422


def test_batch_predict_empty_list_returns_422(client, mock_model_service):
    # min_length=1 on the customers list
    response = client.post("/predict/batch", json={"customers": []})
    assert response.status_code == 422


def test_batch_response_length_matches_input(client, mock_model_service):
    n = 25
    response = client.post("/predict/batch", json=_make_batch(n))
    data = response.json()
    assert data["batch_size"] == n
    assert len(data["predictions"]) == n


def test_batch_response_contains_processing_time(client, mock_model_service):
    response = client.post("/predict/batch", json=_make_batch(5))
    data = response.json()
    assert "processing_time_ms" in data
    assert data["processing_time_ms"] >= 0


def test_batch_each_prediction_has_unique_id(client, mock_model_service):
    response = client.post("/predict/batch", json=_make_batch(10))
    data = response.json()
    ids = [p["prediction_id"] for p in data["predictions"]]
    assert len(set(ids)) == 10  # all unique
