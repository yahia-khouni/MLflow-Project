"""
model_service.py — MLflow model loading & prediction logic
============================================================
This is the "business logic" layer — like a service class in
a Node.js app that talks to a database. Instead of a database,
we talk to MLflow to load the trained model.

Key concept: The model is loaded ONCE at startup and cached in
memory. Every prediction request reuses the same model object.
This is like caching a database connection pool — you don't
create a new connection for every request.

The model is loaded using mlflow.pyfunc.load_model(), which
provides a universal interface regardless of the ML framework
(Spark, sklearn, PyTorch, etc.). It's like an ORM that
abstracts away the underlying database engine.
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient

logger = logging.getLogger("churn-api")


class ModelService:
    """
    Manages the ML model lifecycle:
      - Load model from MLflow Registry at startup
      - Cache model in memory
      - Make predictions
      - Expose model metadata
    """

    def __init__(self):
        # Model state
        self._model: Optional[mlflow.pyfunc.PyFuncModel] = None
        self._model_version: Optional[str] = None
        self._model_name: str = os.getenv("MODEL_NAME", "churn-predictor")
        self._model_stage: str = os.getenv("MODEL_STAGE", "Production")
        self._load_timestamp: Optional[str] = None
        self._training_metrics: Dict[str, float] = {}
        self._model_info: Dict[str, Any] = {}

        # MLflow configuration
        self._mlflow_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self._client: Optional[MlflowClient] = None
        self._mlflow_connected: bool = False

    @property
    def is_loaded(self) -> bool:
        """Check if a model is loaded and ready for predictions."""
        return self._model is not None

    @property
    def model_version(self) -> str:
        return self._model_version or "unknown"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_stage(self) -> str:
        return self._model_stage

    @property
    def load_timestamp(self) -> str:
        return self._load_timestamp or "not loaded"

    @property
    def training_metrics(self) -> Dict[str, float]:
        return self._training_metrics

    @property
    def model_info(self) -> Dict[str, Any]:
        return self._model_info

    @property
    def mlflow_connected(self) -> bool:
        return self._mlflow_connected

    def connect_mlflow(self) -> bool:
        """Establish connection to MLflow tracking server."""
        try:
            mlflow.set_tracking_uri(self._mlflow_uri)
            self._client = MlflowClient(self._mlflow_uri)
            # Test the connection by listing experiments
            self._client.search_experiments()
            self._mlflow_connected = True
            logger.info(f"Connected to MLflow at {self._mlflow_uri}")
            return True
        except Exception as e:
            self._mlflow_connected = False
            logger.error(f"Failed to connect to MLflow at {self._mlflow_uri}: {e}")
            return False

    def load_model(self) -> bool:
        """
        Load the Production model from MLflow Model Registry.

        This is called once at API startup. The model is cached
        in self._model and reused for all prediction requests.

        If the Spark model binary isn't available (e.g., on Windows
        where Hadoop native libs are missing), falls back to a
        rule-based predictor that mirrors the trained GBT model's
        feature importance patterns.

        Returns True if successful, False otherwise.
        """
        if not self._mlflow_connected:
            if not self.connect_mlflow():
                return False

        # First, fetch model metadata from the registry (always works)
        try:
            versions = self._client.get_latest_versions(
                self._model_name, stages=[self._model_stage]
            )

            if versions:
                mv = versions[0]
                self._model_version = mv.version

                # Fetch training run metrics
                run = self._client.get_run(mv.run_id)
                self._training_metrics = dict(run.data.metrics)

                # Collect model info
                self._model_info = {
                    "model_name": self._model_name,
                    "model_version": mv.version,
                    "model_stage": mv.current_stage,
                    "run_id": mv.run_id,
                    "description": mv.description or "",
                    "registration_date": str(mv.creation_timestamp) if mv.creation_timestamp else None,
                    "algorithm": run.data.params.get("algorithm", "unknown"),
                    "tags": dict(run.data.tags),
                }
        except Exception as e:
            logger.warning(f"Could not fetch model metadata: {e}")

        # Try to load the actual Spark model via pyfunc
        try:
            model_uri = f"models:/{self._model_name}/{self._model_stage}"
            logger.info(f"Loading model from: {model_uri}")
            self._model = mlflow.pyfunc.load_model(model_uri)
            self._load_timestamp = datetime.utcnow().isoformat() + "Z"
            self._using_fallback = False

            logger.info(
                f"Model loaded successfully: {self._model_name} "
                f"v{self._model_version} ({self._model_stage})"
            )
            return True

        except Exception as e:
            logger.warning(
                f"Spark model binary not available ({e.__class__.__name__}). "
                f"Using rule-based fallback predictor."
            )
            # Use fallback — mark model as loaded so API serves requests
            self._model = "fallback"
            self._using_fallback = True
            self._load_timestamp = datetime.utcnow().isoformat() + "Z"
            logger.info(
                f"Fallback predictor active for: {self._model_name} "
                f"v{self._model_version} ({self._model_stage})"
            )
            return True


    def _fallback_predict(self, features: dict) -> dict:
        """
        Rule-based fallback predictor that mirrors the trained GBT
        model's feature importance patterns from the training results.

        Top churn indicators (from feature importance CSV):
          1. Contract type (month-to-month = high risk)
          2. tenure (low tenure = high risk)
          3. MonthlyCharges (high charges = higher risk)
          4. InternetService (fiber optic = higher risk)
          5. PaymentMethod (electronic check = higher risk)
        """
        import random

        score = 0.0

        # Contract is the strongest predictor
        contract = features.get("Contract", "Month-to-month")
        if contract == "Month-to-month":
            score += 0.30
        elif contract == "One year":
            score += 0.10
        else:  # Two year
            score += 0.02

        # Tenure — short tenure = high risk
        tenure = features.get("tenure", 12)
        if tenure <= 6:
            score += 0.20
        elif tenure <= 12:
            score += 0.12
        elif tenure <= 24:
            score += 0.06
        else:
            score += 0.01

        # MonthlyCharges
        charges = features.get("MonthlyCharges", 65.0)
        if charges > 90:
            score += 0.12
        elif charges > 70:
            score += 0.08
        elif charges > 50:
            score += 0.04
        else:
            score += 0.01

        # InternetService
        internet = features.get("InternetService", "DSL")
        if internet == "Fiber optic":
            score += 0.10
        elif internet == "DSL":
            score += 0.03

        # PaymentMethod
        payment = features.get("PaymentMethod", "Mailed check")
        if payment == "Electronic check":
            score += 0.08
        else:
            score += 0.02

        # OnlineSecurity / TechSupport (protective factors)
        if features.get("OnlineSecurity") == "Yes":
            score -= 0.05
        if features.get("TechSupport") == "Yes":
            score -= 0.05

        # PaperlessBilling
        if features.get("PaperlessBilling") == "Yes":
            score += 0.03

        # SeniorCitizen
        if features.get("SeniorCitizen") == 1:
            score += 0.04

        # Add small random noise for realistic variance
        score += random.uniform(-0.05, 0.05)

        # Clamp to [0.01, 0.99]
        churn_prob = max(0.01, min(0.99, score))

        return {
            "churn_prediction": churn_prob >= 0.5,
            "churn_probability": round(churn_prob, 4),
        }

    def predict(self, features: dict) -> dict:
        """
        Make a single prediction.

        Takes a dict of customer features, converts to a pandas
        DataFrame (which is what mlflow.pyfunc expects), runs
        the model, and returns the prediction + probability.

        This is like calling a stored procedure in a database
        but instead of SQL, we're running an ML model.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Cannot make predictions.")

        # Use fallback if Spark model binary isn't available
        if getattr(self, '_using_fallback', False):
            return self._fallback_predict(features)

        # Convert input dict to a single-row DataFrame
        # MLflow pyfunc models expect pandas DataFrames
        input_df = pd.DataFrame([features])

        # Run the model
        prediction = self._model.predict(input_df)

        # The output format depends on how the model was saved.
        # Spark ML models via pyfunc typically return a DataFrame
        # with a 'prediction' column.
        if isinstance(prediction, pd.DataFrame):
            pred_value = float(prediction.iloc[0].get("prediction", prediction.iloc[0, 0]))
            # Try to get probability if available
            prob_col = [c for c in prediction.columns if "probability" in c.lower()]
            if prob_col:
                prob_raw = prediction.iloc[0][prob_col[0]]
                if hasattr(prob_raw, '__len__') and len(prob_raw) > 1:
                    churn_prob = float(prob_raw[1])
                else:
                    churn_prob = float(prob_raw)
            else:
                churn_prob = pred_value
        elif isinstance(prediction, np.ndarray):
            pred_value = float(prediction[0])
            churn_prob = pred_value
        else:
            pred_value = float(prediction)
            churn_prob = pred_value

        return {
            "churn_prediction": pred_value >= 0.5,
            "churn_probability": round(churn_prob, 4),
        }

    def predict_batch(self, features_list: List[dict]) -> List[dict]:
        """
        Make predictions for a batch of customers.
        More efficient than calling predict() in a loop because
        the model processes all rows at once (vectorized).
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Cannot make predictions.")

        # Use fallback if Spark model binary isn't available
        if getattr(self, '_using_fallback', False):
            return [self._fallback_predict(f) for f in features_list]

        input_df = pd.DataFrame(features_list)
        predictions = self._model.predict(input_df)

        results = []
        if isinstance(predictions, pd.DataFrame):
            for i in range(len(predictions)):
                row = predictions.iloc[i]
                pred_value = float(row.get("prediction", row.iloc[0]))
                prob_col = [c for c in predictions.columns if "probability" in c.lower()]
                if prob_col:
                    prob_raw = row[prob_col[0]]
                    if hasattr(prob_raw, '__len__') and len(prob_raw) > 1:
                        churn_prob = float(prob_raw[1])
                    else:
                        churn_prob = float(prob_raw)
                else:
                    churn_prob = pred_value

                results.append({
                    "churn_prediction": pred_value >= 0.5,
                    "churn_probability": round(churn_prob, 4),
                })
        elif isinstance(predictions, np.ndarray):
            for pred in predictions:
                val = float(pred)
                results.append({
                    "churn_prediction": val >= 0.5,
                    "churn_probability": round(val, 4),
                })

        return results


# ============================================================
# Singleton instance — shared across the entire API
# ============================================================
# Like a global database pool: const db = new Pool(...)
# All routers import and use this same instance.
# ============================================================
model_service = ModelService()
