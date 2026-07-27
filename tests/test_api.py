"""API-level tests: response schema and graceful-degradation fallback paths."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from config import BRANCHES, DESK_SERVICES

REQUIRED_PREDICTION_FIELDS = {
    "predicted_wait_minutes",
    "surge_level",
    "confidence_score",
    "recommended_alternative_store",
}


def test_health_reports_model_loaded() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_200_and_valid_schema() -> None:
    with TestClient(app) as client:
        response = client.get("/predict", params={"branch_id": BRANCHES[0].branch_id, "desk_service_id": DESK_SERVICES[0]})

    assert response.status_code == 200
    body = response.json()

    assert REQUIRED_PREDICTION_FIELDS.issubset(body.keys())
    assert isinstance(body["predicted_wait_minutes"], (int, float))
    assert body["predicted_wait_minutes"] >= 0
    # "closed" is a real, valid state (config.SURGE_CLOSED_LABEL) — whatever
    # time this test happens to run, the requested branch may legitimately
    # be outside business hours.
    assert body["surge_level"] in {"low", "moderate", "high", "critical", "closed"}
    assert 0.0 <= body["confidence_score"] <= 1.0
    assert body["branch_id"] == BRANCHES[0].branch_id
    assert body["desk_service_id"] == DESK_SERVICES[0]


def test_predict_accepts_explicit_datetime() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/predict",
            params={
                "branch_id": BRANCHES[0].branch_id,
                "desk_service_id": DESK_SERVICES[0],
                "datetime": "2026-12-24T09:30:00",
            },
        )
    assert response.status_code == 200
    assert response.json()["datetime"].startswith("2026-12-24T09:30:00")


def test_predict_unknown_branch_returns_404() -> None:
    with TestClient(app) as client:
        response = client.get("/predict", params={"branch_id": "not_a_real_branch", "desk_service_id": DESK_SERVICES[0]})
    assert response.status_code == 404


def test_predict_unknown_desk_service_returns_404() -> None:
    with TestClient(app) as client:
        response = client.get("/predict", params={"branch_id": BRANCHES[0].branch_id, "desk_service_id": "not_a_real_service"})
    assert response.status_code == 404


def test_predict_invalid_datetime_returns_400() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/predict",
            params={"branch_id": BRANCHES[0].branch_id, "desk_service_id": DESK_SERVICES[0], "datetime": "not-a-date"},
        )
    assert response.status_code == 400


def test_predict_survives_missing_weather_cache(monkeypatch) -> None:
    """Simulates the Open-Meteo cache never having been populated (e.g. the
    API has been unreachable since startup): /predict must still succeed via
    the baseline rain_mm fallback, per the graceful-degradation rule."""
    with TestClient(app) as client:
        prediction_service = app.state.prediction_service
        assert prediction_service is not None
        monkeypatch.setattr(prediction_service.weather_cache, "_rain_mm_by_branch", {})

        response = client.get("/predict", params={"branch_id": BRANCHES[0].branch_id, "desk_service_id": DESK_SERVICES[0]})

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_wait_minutes"] >= 0
    assert 0.0 <= body["confidence_score"] <= 1.0
