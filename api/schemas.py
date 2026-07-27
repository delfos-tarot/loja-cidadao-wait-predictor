"""Pydantic response models for the wait-time prediction API."""

from __future__ import annotations

from pydantic import BaseModel


class RecommendedAlternative(BaseModel):
    branch_id: str
    branch_name: str
    distance_km: float
    predicted_wait_minutes: float
    time_saved_minutes: float


class PredictionResponse(BaseModel):
    branch_id: str
    desk_service_id: str
    datetime: str
    predicted_wait_minutes: float
    surge_level: str
    confidence_score: float
    recommended_alternative_store: RecommendedAlternative | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_trained_at: str | None = None
