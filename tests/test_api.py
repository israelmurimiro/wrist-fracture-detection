"""
API Tests - Unit tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_analyze_no_file():
    """Test analyze endpoint without file."""
    response = client.post("/analyze")
    assert response.status_code == 422


def test_diagnose_no_file():
    """Test diagnose endpoint without file."""
    response = client.post("/diagnose")
    assert response.status_code == 422


def test_explain_no_file():
    """Test explain endpoint without file."""
    response = client.post("/explain")
    assert response.status_code == 422


def test_treatment_no_file():
    """Test treatment endpoint without file."""
    response = client.post("/treatment")
    assert response.status_code == 422


def test_root():
    """Test root endpoint returns HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
