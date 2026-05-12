"""
schemas.py — Pydantic models for request/response validation
=============================================================
In Express.js, you'd use Joi or Zod for input validation.
In FastAPI, we use Pydantic — it's built-in and auto-generates
Swagger docs from these schemas. Every field with a type hint
is automatically validated on incoming requests.

Key concepts:
  BaseModel    = like a TypeScript interface with runtime validation
  Field(...)   = like Joi.number().min(0).max(72)
  Literal[...] = like TypeScript's "Male" | "Female" union type
"""

from pydantic import BaseModel, Field
from typing import Literal, List, Optional
from datetime import datetime
import uuid


# ============================================================
# Input: Single Customer Features
# ============================================================
# This matches the exact columns from the Telco Churn dataset.
# FastAPI will auto-validate every field — if someone sends
# gender="Other", they'll get a 422 error with a clear message.
# ============================================================
class CustomerFeatures(BaseModel):
    """All features needed to predict churn for one customer."""

    gender: Literal["Male", "Female"] = Field(
        ..., description="Customer gender", examples=["Male"]
    )
    SeniorCitizen: Literal[0, 1] = Field(
        ..., description="1 if senior citizen, 0 otherwise", examples=[0]
    )
    Partner: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer has a partner", examples=["Yes"]
    )
    Dependents: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer has dependents", examples=["No"]
    )
    tenure: int = Field(
        ..., ge=0, le=72,
        description="Number of months the customer has been with the company",
        examples=[12]
    )
    PhoneService: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer has phone service", examples=["Yes"]
    )
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ..., description="Whether the customer has multiple phone lines",
        examples=["No"]
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Customer's internet service type", examples=["Fiber optic"]
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has online security add-on",
        examples=["No"]
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has online backup add-on",
        examples=["Yes"]
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has device protection add-on",
        examples=["No"]
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has tech support add-on",
        examples=["No"]
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has streaming TV add-on",
        examples=["No"]
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has streaming movies add-on",
        examples=["No"]
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Customer's contract type", examples=["Month-to-month"]
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer uses paperless billing",
        examples=["Yes"]
    )
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ] = Field(
        ..., description="Customer's payment method", examples=["Electronic check"]
    )
    MonthlyCharges: float = Field(
        ..., gt=0,
        description="Monthly charge amount in dollars", examples=[70.35]
    )
    TotalCharges: float = Field(
        ..., ge=0,
        description="Total charges since signup in dollars", examples=[1397.475]
    )


# ============================================================
# Output: Single Prediction Response
# ============================================================
class PredictionResponse(BaseModel):
    """Prediction result for one customer."""

    churn_prediction: bool = Field(
        ..., description="True if the model predicts the customer will churn"
    )
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability of churning (0.0 = definitely stays, 1.0 = definitely churns)"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="Confidence level based on probability distance from 0.5"
    )
    model_version: str = Field(
        ..., description="MLflow model version used for this prediction"
    )
    prediction_id: str = Field(
        ..., description="Unique UUID for traceability and debugging"
    )


# ============================================================
# Batch: Request & Response
# ============================================================
class BatchPredictionRequest(BaseModel):
    """Batch of customers to predict (max 1,000 records)."""

    customers: List[CustomerFeatures] = Field(
        ..., min_length=1, max_length=1000,
        description="List of customer feature sets (1–1,000 records)"
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction results."""

    predictions: List[PredictionResponse] = Field(
        ..., description="One prediction per input customer"
    )
    batch_size: int = Field(
        ..., description="Number of records processed"
    )
    processing_time_ms: float = Field(
        ..., description="Total processing time in milliseconds"
    )


# ============================================================
# Health & Model Info
# ============================================================
class HealthResponse(BaseModel):
    """API health check response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    uptime_seconds: float
    model_loaded: bool
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    mlflow_connected: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Detailed model metadata from MLflow registry."""

    model_name: str
    model_version: str
    model_stage: str
    algorithm: Optional[str] = None
    training_metrics: dict = Field(default_factory=dict)
    registration_date: Optional[str] = None
    description: Optional[str] = None
    tags: dict = Field(default_factory=dict)
    load_timestamp: str


# ============================================================
# Error Response
# ============================================================
class ErrorResponse(BaseModel):
    """Structured error response for all error cases."""

    error: str
    detail: str
    timestamp: str


# ============================================================
# Helper: Generate prediction ID
# ============================================================
def generate_prediction_id() -> str:
    """Generate a unique prediction ID (UUID v4)."""
    return str(uuid.uuid4())


# ============================================================
# Helper: Determine confidence level
# ============================================================
def get_confidence_level(probability: float) -> str:
    """
    Map prediction probability to a confidence level.
    The further from 0.5 (the decision boundary), the more
    confident the model is.

    - High:   probability < 0.2 or > 0.8  (very sure)
    - Medium: probability < 0.35 or > 0.65
    - Low:    near 0.5 (model is basically guessing)
    """
    distance_from_boundary = abs(probability - 0.5)
    if distance_from_boundary >= 0.3:
        return "high"
    elif distance_from_boundary >= 0.15:
        return "medium"
    else:
        return "low"
