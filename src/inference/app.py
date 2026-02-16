"""
FastAPI inference service entrypoint.

This file defines:
- /health : for liveness checks (required by rubric)
- /predict : placeholder for prediction endpoint (we implement fully after model training)
"""

from fastapi import FastAPI  # FastAPI framework
from pydantic import BaseModel  # Request/response validation using Pydantic


# Create the FastAPI application object (ASGI app)
app = FastAPI(
    title="Cats vs Dogs Inference API",  # Shown in Swagger UI
    version="0.1.0",  # App version for visibility
)


class HealthResponse(BaseModel):
    """
    Typed schema for the /health response.
    Using a schema makes responses consistent and testable.
    """
    status: str  # e.g., "ok"


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Health endpoint used for smoke tests and deployment checks.
    Returns a simple OK response when service is up.
    """
    return HealthResponse(status="ok")


class PredictResponse(BaseModel):
    """
    Placeholder schema for /predict.
    We'll later return probabilities + label.
    """
    message: str  # temporary field until model inference is wired


@app.post("/predict", response_model=PredictResponse)
def predict_placeholder() -> PredictResponse:
    """
    Placeholder /predict endpoint.
    We keep it minimal now so tests + CI can run immediately.
    """
    return PredictResponse(message="Model inference not wired yet")
