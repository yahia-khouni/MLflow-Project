"""
main.py — FastAPI Application Entry Point
============================================
This is the "app.ts" or "server.js" of our prediction service.
It wires together routers, middleware, startup logic, and
error handlers into a single FastAPI application.

Key responsibilities:
  1. Create the FastAPI app with metadata for Swagger docs
  2. Register middleware (CORS, request logging, prediction ID)
  3. Register routers (health, predict)
  4. Load the ML model at startup (from MLflow Registry)
  5. Expose Prometheus metrics at /metrics

Run with:
  uvicorn main:app --host 0.0.0.0 --port 8000
  OR: python main.py
"""

import os
import time
import uuid
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from prometheus_fastapi_instrumentator import Instrumentator

from routers import health, predict
from services.model_service import model_service

# ============================================================
# Logging Setup
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("churn-api")


# ============================================================
# Application Lifespan (startup + shutdown)
# ============================================================
# In Express, you'd do setup in app.listen() callback.
# In FastAPI, the "lifespan" context manager runs code
# at startup (before the yield) and shutdown (after yield).
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: Load the ML model from MLflow Registry.
    Shutdown: Cleanup resources.
    """
    logger.info("=" * 50)
    logger.info("Starting Churn Prediction API...")
    logger.info("=" * 50)

    # Try to connect to MLflow and load the model
    if model_service.connect_mlflow():
        if model_service.load_model():
            logger.info(
                f"Model loaded: {model_service.model_name} "
                f"v{model_service.model_version}"
            )
        else:
            logger.warning(
                "Failed to load model from MLflow. "
                "API will start but /predict will return 503."
            )
    else:
        logger.warning(
            "MLflow is unreachable. API will start in degraded mode. "
            "Health endpoint will report 'unhealthy'."
        )

    logger.info("API is ready to serve requests!")
    logger.info(f"  Swagger docs: http://localhost:{os.getenv('API_PORT', '8000')}/docs")

    yield  # --- App runs here ---

    logger.info("Shutting down Churn Prediction API...")


# ============================================================
# Create FastAPI Application
# ============================================================
app = FastAPI(
    title="Churn Prediction API",
    description=(
        "## ML Production Pipeline — Customer Churn Prediction\n\n"
        "This API serves predictions from a Spark MLlib model "
        "registered in MLflow's Model Registry.\n\n"
        "### Features\n"
        "- **Single prediction**: POST `/predict`\n"
        "- **Batch prediction**: POST `/predict/batch` (up to 1,000 records)\n"
        "- **Health monitoring**: GET `/health`\n"
        "- **Model info**: GET `/model/info`\n"
        "- **Prometheus metrics**: GET `/metrics`\n\n"
        "### How it works\n"
        "1. At startup, the API loads the Production model from MLflow\n"
        "2. For each request, input is validated by Pydantic schemas\n"
        "3. The model runs inference and returns a probability + confidence\n"
        "4. Every prediction gets a unique ID for traceability\n"
        "5. Prometheus scrapes metrics every 10 seconds for Grafana dashboards"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ============================================================
# Middleware Stack
# ============================================================
# Middleware runs on EVERY request, like Express app.use().
# Order matters — they execute top to bottom for requests,
# bottom to top for responses.
# ============================================================

# --- 1. CORS Middleware ---
# Allow all origins during development.
# In production, you'd restrict this to your frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 2. Request Logging Middleware ---
# Logs every request: method, path, status code, duration.
# Like morgan('combined') in Express.
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]

    # Process the request
    response = await call_next(request)

    # Calculate duration
    duration_ms = round((time.time() - start_time) * 1000, 2)

    # Log the request
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"→ {response.status_code} ({duration_ms}ms)"
    )

    # Add traceability header to every response
    response.headers["X-Request-ID"] = request_id

    return response


# ============================================================
# Prometheus Instrumentation
# ============================================================
# Auto-exposes GET /metrics with HTTP request stats:
#   - http_requests_total (counter by method, endpoint, status)
#   - http_request_duration_seconds (latency histogram)
# ============================================================
Instrumentator().instrument(app).expose(app)


# ============================================================
# Register Routers
# ============================================================
# Like app.use('/api', userRouter) in Express.
# FastAPI routers group related endpoints together.
# ============================================================
app.include_router(health.router)
app.include_router(predict.router)


# ============================================================
# Global Error Handlers
# ============================================================
# These catch errors that slip through individual routes.
# Like Express error middleware: app.use((err, req, res, next) => {...})
# ============================================================

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """
    Handle 422 Validation Errors with clear field-level messages.
    This fires when Pydantic rejects input (wrong type, missing field, etc.)
    """
    errors = []
    for error in exc.errors():
        field = " → ".join(str(loc) for loc in error["loc"])
        errors.append(f"{field}: {error['msg']}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": "; ".join(errors),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    """Catch-all for unexpected errors."""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    )


# ============================================================
# Root Redirect
# ============================================================
@app.get("/", include_in_schema=False)
async def root():
    """Redirect to Swagger docs for convenience."""
    return {
        "message": "Churn Prediction API v1.0.0",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model/info",
    }


# ============================================================
# Direct Run
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,  # auto-restart on code changes (dev only)
    )
