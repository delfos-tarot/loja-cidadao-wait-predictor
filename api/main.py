"""FastAPI service for Loja do Cidadao wait-time predictions.

The trained XGBoost model artifact is loaded once at startup. Every
/predict request is then a pure in-process model inference plus a couple of
indexed SQLite lookups and an in-memory dict read for weather — no LLM calls
and no synchronous network I/O in the request path, keeping latency in the
sub-50ms range.

Run with:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, Query

from api.schemas import HealthResponse, PredictionResponse, RecommendedAlternative
from api.service import DemandBaselineCache, PredictionService, WeatherCache, classify_surge_level
from config import BRANCHES_BY_ID, DEFAULT_DB_PATH, DEFAULT_MODEL_PATH, DESK_SERVICES, SURGE_CLOSED_LABEL
from pipeline.feature_engineering import WeatherClient

logger = logging.getLogger(__name__)

WEATHER_REFRESH_INTERVAL_SECONDS = 30 * 60


async def _weather_refresh_loop(weather_cache: WeatherCache) -> None:
    while True:
        await asyncio.sleep(WEATHER_REFRESH_INTERVAL_SECONDS)
        try:
            await asyncio.to_thread(weather_cache.refresh)
        except Exception:
            logger.exception("Weather cache refresh failed; keeping previous cached values")


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_artifact: dict[str, Any] | None
    try:
        model_artifact = joblib.load(DEFAULT_MODEL_PATH)
    except FileNotFoundError:
        logger.error("Model artifact not found at %s; run `python -m pipeline.train` first", DEFAULT_MODEL_PATH)
        model_artifact = None

    weather_cache = WeatherCache(WeatherClient())
    try:
        weather_cache.refresh()
    except Exception:
        logger.exception("Initial weather cache refresh failed; starting with baseline fallback")

    demand_baseline_cache = DemandBaselineCache(DEFAULT_DB_PATH)

    app.state.model_artifact = model_artifact
    app.state.prediction_service = (
        PredictionService(
            model_artifact,
            db_path=DEFAULT_DB_PATH,
            weather_cache=weather_cache,
            demand_baseline_cache=demand_baseline_cache,
        )
        if model_artifact is not None
        else None
    )

    refresh_task = asyncio.create_task(_weather_refresh_loop(weather_cache))
    try:
        yield
    finally:
        refresh_task.cancel()


app = FastAPI(title="Loja do Cidadao Wait Time Predictor", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model_artifact = app.state.model_artifact
    return HealthResponse(
        status="ok" if model_artifact is not None else "degraded",
        model_loaded=model_artifact is not None,
        model_trained_at=model_artifact.get("trained_at") if model_artifact else None,
    )


@app.get("/predict", response_model=PredictionResponse)
def predict(
    branch_id: str = Query(..., description="Target branch identifier"),
    desk_service_id: str = Query(..., description="Target desk/service identifier"),
    target_datetime: str | None = Query(None, alias="datetime", description="ISO8601 datetime; defaults to now"),
) -> PredictionResponse:
    if branch_id not in BRANCHES_BY_ID:
        raise HTTPException(status_code=404, detail=f"Unknown branch_id '{branch_id}'")
    if desk_service_id not in DESK_SERVICES:
        raise HTTPException(status_code=404, detail=f"Unknown desk_service_id '{desk_service_id}'")

    if target_datetime is None:
        parsed_datetime = datetime.now(timezone.utc)
    else:
        try:
            parsed_datetime = datetime.fromisoformat(target_datetime)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid datetime '{target_datetime}'; expected ISO8601")
        if parsed_datetime.tzinfo is None:
            parsed_datetime = parsed_datetime.replace(tzinfo=timezone.utc)

    prediction_service: PredictionService | None = app.state.prediction_service
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded; run `python -m pipeline.train` first")

    predicted_wait_minutes, confidence_score = prediction_service.predict_wait_minutes(
        branch_id, desk_service_id, parsed_datetime
    )
    # surge_level carries the closed state instead of a new response field,
    # so the documented four-field /predict contract never changes shape.
    is_open, _ = prediction_service.get_is_open(branch_id, desk_service_id, parsed_datetime)
    surge_level = SURGE_CLOSED_LABEL if not is_open else classify_surge_level(predicted_wait_minutes)
    alternative = prediction_service.find_smart_reroute(branch_id, desk_service_id, parsed_datetime, predicted_wait_minutes)

    return PredictionResponse(
        branch_id=branch_id,
        desk_service_id=desk_service_id,
        datetime=parsed_datetime.isoformat(),
        predicted_wait_minutes=predicted_wait_minutes,
        surge_level=surge_level,
        confidence_score=confidence_score,
        recommended_alternative_store=RecommendedAlternative(**alternative) if alternative else None,
    )
