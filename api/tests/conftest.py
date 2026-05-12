"""
conftest.py — Shared test fixtures
====================================
In pytest, conftest.py is like a shared setup file. Fixtures
defined here are available to ALL test files automatically.

This creates a test client that talks to our FastAPI app
without needing a running server — like supertest in Node.js.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock, patch

from main import app


# A valid customer payload for reuse across tests
VALID_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 1397.475,
}


@pytest.fixture
def client():
    """Synchronous test client (for simple GET/POST tests)."""
    from fastapi.testclient import TestClient
    return TestClient(app)


@pytest.fixture
def mock_model_service():
    """
    Mock the model service so tests don't need a running MLflow.

    This patches model_service to return fake predictions.
    Like mocking a database in unit tests.
    """
    with patch("routers.predict.model_service") as mock_predict, \
         patch("routers.health.model_service") as mock_health:

        # Configure the mock to act like a loaded model
        for mock in [mock_predict, mock_health]:
            mock.is_loaded = True
            mock.model_version = "1"
            mock.model_name = "churn-predictor"
            mock.model_stage = "Production"
            mock.mlflow_connected = True
            mock.load_timestamp = "2024-01-01T00:00:00Z"
            mock.training_metrics = {
                "auc_roc": 0.87,
                "accuracy": 0.82,
                "f1_score": 0.79,
            }
            mock.model_info = {
                "algorithm": "Random Forest",
                "registration_date": "2024-01-01",
                "description": "Test model",
                "tags": {"team": "test"},
            }

        # Mock single prediction
        mock_predict.predict.return_value = {
            "churn_prediction": True,
            "churn_probability": 0.85,
        }

        # Mock batch prediction
        def batch_predict(features_list):
            return [
                {"churn_prediction": i % 2 == 0, "churn_probability": 0.5 + (i % 5) * 0.1}
                for i in range(len(features_list))
            ]
        mock_predict.predict_batch.side_effect = batch_predict

        yield mock_predict, mock_health
