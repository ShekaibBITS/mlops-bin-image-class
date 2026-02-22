# src/inference/app.py

"""
FastAPI inference service.

Endpoints:
- GET /health : liveness check (rubric requirement)
- POST /predict : image classification (rubric requirement)

Notes:
- Loads model bundle at startup for performance.
- Logs basic request/latency info (useful for M5 monitoring/logging).
"""

import logging  # Logging to stdout
import time  # Latency timing
from io import BytesIO  # Convert uploaded bytes to file-like stream

from fastapi import FastAPI, File, UploadFile  # FastAPI primitives
from PIL import Image  # Decode image bytes

from src.inference.predict import load_model_bundle, predict  # Inference utilities
from src.inference.schema import HealthResponse, PredictResponse  # Response schemas


# Configure basic logging (stdout)
logging.basicConfig(level=logging.INFO)  # Set log level
logger = logging.getLogger("cats-dogs-api")  # Named logger


# Create FastAPI app
app = FastAPI(
    title="Cats vs Dogs Inference API",
    version="0.2.0",
)

# Load model at process startup (global bundle)
MODEL_BUNDLE = load_model_bundle(models_dir="models")  # Load weights + metadata once


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Health check endpoint.
    Used for smoke tests and deployment readiness checks.
    """
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(file: UploadFile = File(...)) -> PredictResponse:
    """
    Predict endpoint:
    - Accepts image upload as multipart/form-data
    - Returns class probabilities and predicted label
    """
    req_t0 = time.time()  # Request start time

    # Basic request logging (do NOT log raw image bytes)
    logger.info("predict_request_received filename=%s content_type=%s", file.filename, file.content_type)

    # Read uploaded file bytes
    content = await file.read()  # Read entire uploaded file into memory
    if not content:  # Empty upload check
        raise ValueError("Empty file uploaded")  # Fast failure

    # Decode image bytes into PIL image
    image = Image.open(BytesIO(content))  # PIL image from bytes

    # Run model inference
    result = predict(image=image, bundle=MODEL_BUNDLE)  # Dict with label, probabilities, latency_ms

    req_latency_ms = (time.time() - req_t0) * 1000.0  # End-to-end request latency
    logger.info(
        "predict_request_completed label=%s model_latency_ms=%.2f request_latency_ms=%.2f",
        result["label"],
        result["latency_ms"],
        req_latency_ms,
    )

    # Return typed response
    return PredictResponse(**result)
