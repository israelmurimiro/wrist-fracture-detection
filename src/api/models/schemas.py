"""
Pydantic Schemas for API Request/Response Validation
"""

from pydantic import BaseModel
from typing import List, Optional, Dict


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class AnalysisRequest(BaseModel):
    """Request schema for analysis endpoint."""
    file: bytes


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    message: str


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class DiagnosisResponse(BaseModel):
    """Response schema for quick diagnosis."""
    prediction: str
    confidence: float
    probabilities: Dict[str, float]


class AnalysisResponse(BaseModel):
    """Response schema for full analysis."""
    prediction: str
    confidence: float
    detected_classes: List[str]
    caption: str
    treatment: str
    heatmap: str  # base64 encoded image


class ExplainResponse(BaseModel):
    """Response schema for explanation endpoint."""
    prediction: str
    confidence: float
    heatmap: str  # base64 encoded image
    probabilities: Dict[str, float]


class TreatmentResponse(BaseModel):
    """Response schema for treatment endpoint."""
    prediction: str
    confidence: float
    caption: str
    treatment: str


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    error: str


# ============================================================
# HEALTH CHECK
# ============================================================

class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    version: str
    device: str
    model_loaded: bool
