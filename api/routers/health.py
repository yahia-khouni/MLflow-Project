"""
health.py — Health check & model info endpoints
=================================================
These are operational endpoints — not for business logic,
but for monitoring and debugging. Like a /status or /health
endpoint in any Node.js API.

GET /health     → Is the API alive? Is the model loaded?
GET /model/info → What model version is running? What metrics?
"""

from fastapi import APIRouter
from datetime import datetime
import time

from models.schemas import HealthResponse, ModelInfoResponse
from services.model_service import model_service

router = APIRouter(tags=["Health & Info"])

# Track startup time for uptime calculation
START_TIME = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API health check",
    description="Returns API status, model state, uptime, and MLflow connectivity.",
    response_description="Health status of the API",
)
async def health_check():
    """
    Health check endpoint.

    Returns:
    - **status**: healthy/degraded/unhealthy
    - **model_loaded**: whether the ML model is in memory
    - **mlflow_connected**: whether MLflow server is reachable
    - **uptime_seconds**: how long the API has been running
    """
    # Determine overall status
    if model_service.is_loaded and model_service.mlflow_connected:
        status = "healthy"
    elif model_service.is_loaded:
        status = "degraded"  # model works but MLflow is down
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        uptime_seconds=round(time.time() - START_TIME, 2),
        model_loaded=model_service.is_loaded,
        model_name=model_service.model_name if model_service.is_loaded else None,
        model_version=model_service.model_version if model_service.is_loaded else None,
        mlflow_connected=model_service.mlflow_connected,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model metadata",
    description="Returns detailed info about the currently loaded model.",
    response_description="Model version, algorithm, training metrics, and tags",
)
async def model_info():
    """
    Model information endpoint.

    Returns metadata from the MLflow Model Registry:
    - Model name, version, and lifecycle stage
    - Algorithm used for training
    - Training metrics (AUC-ROC, accuracy, etc.)
    - Registration date and description
    """
    info = model_service.model_info

    return ModelInfoResponse(
        model_name=model_service.model_name,
        model_version=model_service.model_version,
        model_stage=model_service.model_stage,
        algorithm=info.get("algorithm"),
        training_metrics=model_service.training_metrics,
        registration_date=info.get("registration_date"),
        description=info.get("description"),
        tags=info.get("tags", {}),
        load_timestamp=model_service.load_timestamp,
    )
