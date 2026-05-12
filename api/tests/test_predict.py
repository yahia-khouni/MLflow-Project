"""
test_predict.py — Single prediction endpoint tests
"""
import copy
from tests.conftest import VALID_CUSTOMER


def test_predict_valid_customer_returns_200(client, mock_model_service):
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200


def test_predict_returns_probability_between_0_and_1(client, mock_model_service):
    response = client.post("/predict", json=VALID_CUSTOMER)
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_missing_field_returns_422(client, mock_model_service):
    # Remove a required field
    incomplete = copy.deepcopy(VALID_CUSTOMER)
    del incomplete["gender"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_invalid_enum_value_returns_422(client, mock_model_service):
    invalid = copy.deepcopy(VALID_CUSTOMER)
    invalid["gender"] = "Other"  # not in Literal["Male", "Female"]
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_negative_tenure_returns_422(client, mock_model_service):
    invalid = copy.deepcopy(VALID_CUSTOMER)
    invalid["tenure"] = -5  # ge=0 constraint violated
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_response_contains_prediction_id(client, mock_model_service):
    response = client.post("/predict", json=VALID_CUSTOMER)
    data = response.json()
    assert "prediction_id" in data
    assert len(data["prediction_id"]) > 0  # UUID string


def test_predict_response_contains_confidence(client, mock_model_service):
    response = client.post("/predict", json=VALID_CUSTOMER)
    data = response.json()
    assert data["confidence"] in ["high", "medium", "low"]


def test_predict_response_contains_model_version(client, mock_model_service):
    response = client.post("/predict", json=VALID_CUSTOMER)
    data = response.json()
    assert data["model_version"] == "1"


def test_predict_has_traceability_header(client, mock_model_service):
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert "x-prediction-id" in response.headers


def test_predict_invalid_payment_method_returns_422(client, mock_model_service):
    invalid = copy.deepcopy(VALID_CUSTOMER)
    invalid["PaymentMethod"] = "Bitcoin"  # not a valid option
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_zero_monthly_charges_returns_422(client, mock_model_service):
    invalid = copy.deepcopy(VALID_CUSTOMER)
    invalid["MonthlyCharges"] = 0  # gt=0 constraint violated
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422
