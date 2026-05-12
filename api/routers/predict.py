"""
predict.py — Prediction endpoints
===================================
These are the core business endpoints:

POST /predict       → Single real-time prediction
POST /predict/batch → Batch prediction (up to 1,000 records)

This is like a typical POST endpoint in Express:
  app.post('/predict', async (req, res) => {
    const result = await model.predict(req.body)
    res.json(result)
  })

But with automatic input validation (Pydantic), auto-generated
Swagger docs, and structured error handling — all for free.
"""

import time
import logging
from fastapi import APIRouter, HTTPException, Response
from prometheus_client import Counter, Histogram

from models.schemas import (
    CustomerFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    generate_prediction_id,
    get_confidence_level,
)
from services.model_service import model_service

logger = logging.getLogger("churn-api")
router = APIRouter(tags=["Predictions"])

# ============================================================
# Prometheus Custom Metrics
# ============================================================
# These are the custom metrics we add on top of the auto-
# instrumented ones (request count, latency). They track
# ML-specific stats that show up in Grafana dashboards.
# ============================================================
PREDICTION_COUNTER = Counter(
    "churn_predictions_total",
    "Total number of churn predictions made",
    ["prediction"],  # labeled: prediction=true or prediction=false
)

PROBABILITY_HISTOGRAM = Histogram(
    "churn_probability_score",
    "Distribution of churn probability scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

CONFIDENCE_HISTOGRAM = Histogram(
    "model_confidence",
    "Distribution of model confidence levels",
    ["level"],  # labeled: level=high/medium/low
    buckets=[0.0, 0.5, 1.0],
)

BATCH_SIZE_HISTOGRAM = Histogram(
    "batch_size",
    "Distribution of batch request sizes",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Single prediction",
    description="Predict whether a single customer will churn.",
    response_description="Churn prediction with probability and confidence",
    responses={
        200: {"description": "Prediction successful"},
        422: {"description": "Validation error — invalid input fields"},
        503: {"description": "Model not loaded or MLflow unreachable"},
    },
)
async def predict_single(
    customer: CustomerFeatures,
    response: Response,
):
    """
    Predict churn for a single customer.

    Send customer features and receive:
    - **churn_prediction**: True/False
    - **churn_probability**: 0.0 to 1.0
    - **confidence**: high/medium/low
    - **prediction_id**: UUID for traceability
    """
    # Check model is loaded
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. The API is starting up or MLflow is unreachable.",
        )

    # Generate a unique prediction ID (like a request ID in Express)
    prediction_id = generate_prediction_id()

    # Add traceability header
    response.headers["X-Prediction-ID"] = prediction_id

    # Make prediction
    features_dict = customer.model_dump()
    result = model_service.predict(features_dict)

    # Determine confidence level
    confidence = get_confidence_level(result["churn_probability"])

    # Update Prometheus metrics
    pred_label = "true" if result["churn_prediction"] else "false"
    PREDICTION_COUNTER.labels(prediction=pred_label).inc()
    PROBABILITY_HISTOGRAM.observe(result["churn_probability"])
    CONFIDENCE_HISTOGRAM.labels(level=confidence).observe(1.0)

    return PredictionResponse(
        churn_prediction=result["churn_prediction"],
        churn_probability=result["churn_probability"],
        confidence=confidence,
        model_version=model_service.model_version,
        prediction_id=prediction_id,
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch prediction",
    description="Predict churn for up to 1,000 customers at once.",
    response_description="List of predictions matching input order",
    responses={
        200: {"description": "Batch prediction successful"},
        400: {"description": "Empty list or exceeds 1,000 records"},
        422: {"description": "Validation error in one or more records"},
        503: {"description": "Model not loaded or MLflow unreachable"},
    },
)
async def predict_batch(
    batch: BatchPredictionRequest,
    response: Response,
):
    """
    Predict churn for a batch of customers (max 1,000).

    More efficient than calling /predict in a loop because the
    model processes all records in a single vectorized operation.
    Results are returned in the same order as the input.
    """
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. The API is starting up or MLflow is unreachable.",
        )

    start_time = time.time()

    # Convert all customer features to dicts
    features_list = [c.model_dump() for c in batch.customers]

    # Batch predict
    results = model_service.predict_batch(features_list)

    processing_time_ms = round((time.time() - start_time) * 1000, 2)

    # Build response with prediction IDs and confidence
    predictions = []
    for result in results:
        pred_id = generate_prediction_id()
        confidence = get_confidence_level(result["churn_probability"])

        # Update Prometheus metrics
        pred_label = "true" if result["churn_prediction"] else "false"
        PREDICTION_COUNTER.labels(prediction=pred_label).inc()
        PROBABILITY_HISTOGRAM.observe(result["churn_probability"])

        predictions.append(PredictionResponse(
            churn_prediction=result["churn_prediction"],
            churn_probability=result["churn_probability"],
            confidence=confidence,
            model_version=model_service.model_version,
            prediction_id=pred_id,
        ))

    # Track batch size in Prometheus
    BATCH_SIZE_HISTOGRAM.observe(len(batch.customers))

    return BatchPredictionResponse(
        predictions=predictions,
        batch_size=len(predictions),
        processing_time_ms=processing_time_ms,
    )
