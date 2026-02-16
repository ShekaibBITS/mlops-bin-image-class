"""
Unit tests for the FastAPI service.

We start with /health because it is:
- required by the rubric
- a standard smoke-test target for CI/CD
"""

from fastapi.testclient import TestClient  # Test client for FastAPI apps
from src.inference.app import app  # Import the FastAPI app


def test_health_check_returns_ok() -> None:
    """
    Ensure /health returns HTTP 200 and expected JSON payload.
    """
    client = TestClient(app)  # Create a test client bound to our app
    response = client.get("/health")  # Call the endpoint

    assert response.status_code == 200  # Must be reachable
    assert response.json() == {"status": "ok"}  # Must match schema output
