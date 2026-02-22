# src/inference/schema.py

"""
Pydantic models for API responses.
Keeping schema separate makes testing and evolution easier.
"""

from pydantic import BaseModel  # Base schema class
from typing import Dict  # Type hints


class HealthResponse(BaseModel):
    """Schema for health endpoint."""
    status: str  # "ok"


class PredictResponse(BaseModel):
    """Schema for predict endpoint response."""
    label: str  # "cat" or "dog"
    probabilities: Dict[str, float]  # {"cat": p0, "dog": p1}
    latency_ms: float  # inference latency in milliseconds
    image_size: int  # model expected image size
    device: str  # device used ("mps" or "cpu")
