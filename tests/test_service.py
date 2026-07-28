"""Unit tests for prediction/reroute business logic (api/service.py), isolated
from the FastAPI app so fallback behavior can be verified precisely with
stub models and a stub weather client (no network, no trained artifact
required).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from api.service import (
    PredictionService,
    WeatherCache,
    classify_surge_level,
    estimate_people_waiting,
    haversine_km,
    minutes_to_nearest_snapshot,
)
from config import (
    BASELINE_ATTENDANCES,
    BASELINE_RAIN_MM,
    BRANCHES_BY_ID,
    DIURNAL_SNAPSHOTS,
    MAX_DERIVED_WAIT_MINUTES,
    SNAPSHOT_DISTANCE_CONFIDENCE_PENALTY,
)
from pipeline.db import get_connection, insert_queue_samples, upsert_branch
from pipeline.feature_engineering import FEATURE_COLUMNS
from schemas import QueueReading

# Two real, distinct branches ~2.7km apart (both in Lisboa) that share the
# "Atendimento Geral" service, verified against the real branch registry —
# used by the smart-reroute tests below.
_NEARBY_BRANCH_A = "loja_de_cidadao_das_laranjeiras"
_NEARBY_BRANCH_B = "loja_de_cidadao_do_saldanha"
_SHARED_SERVICE = "Atendimento Geral"


class _ConstantModel:
    """Ignores input features and always predicts the same wait time."""

    def __init__(self, constant: float) -> None:
        self.constant = constant

    def predict(self, X):
        return np.full(len(X), self.constant)


class _PerBranchModel:
    """Returns a distinct constant wait time per branch_id."""

    def __init__(self, wait_by_branch: dict[str, float], default: float = 20.0) -> None:
        self.wait_by_branch = wait_by_branch
        self.default = default

    def predict(self, X):
        return np.array([self.wait_by_branch.get(branch_id, self.default) for branch_id in X["branch_id"]])


class _OverCapModel:
    """Simulates a gradient-boosted model extrapolating past the training
    label range — real XGBoost outputs aren't bounded by label range even
    though every label was capped at MAX_DERIVED_WAIT_MINUTES."""

    def predict(self, X):
        return np.full(len(X), MAX_DERIVED_WAIT_MINUTES * 2)


class _AlwaysFailingWeatherClient:
    def get_hourly_rain(self, latitude, longitude, start_date, end_date):
        return None


def _make_service(db_path: str, model, weather_cache=None) -> PredictionService:
    return PredictionService({"model": model, "feature_columns": FEATURE_COLUMNS}, db_path=db_path, weather_cache=weather_cache)


def test_estimate_people_waiting_scales_with_avg_attendances() -> None:
    anchor_hour, anchor_minute, _ = DIURNAL_SNAPSHOTS[0]
    ts = datetime(2026, 1, 5, anchor_hour, anchor_minute, tzinfo=timezone.utc)

    low_volume = estimate_people_waiting(80.0, ts)
    high_volume = estimate_people_waiting(918.0, ts)

    # Neither must be the old hardcoded 0 -- that's exactly the bug this
    # replaced (a busy branch's fallback must not look like an idle one).
    assert low_volume > 0
    assert high_volume > low_volume


def test_estimate_people_waiting_scales_with_time_of_day_volume_factor() -> None:
    peak_hour, peak_minute, _ = max(DIURNAL_SNAPSHOTS, key=lambda s: s[2])
    trough_hour, trough_minute, _ = min(DIURNAL_SNAPSHOTS, key=lambda s: s[2])
    avg_attendances = 500.0

    at_peak = estimate_people_waiting(avg_attendances, datetime(2026, 1, 5, peak_hour, peak_minute, tzinfo=timezone.utc))
    at_trough = estimate_people_waiting(avg_attendances, datetime(2026, 1, 5, trough_hour, trough_minute, tzinfo=timezone.utc))

    assert at_peak > at_trough


def test_classify_surge_level_boundaries() -> None:
    assert classify_surge_level(0) == "low"
    assert classify_surge_level(9.9) == "low"
    assert classify_surge_level(10) == "moderate"
    assert classify_surge_level(24.9) == "moderate"
    assert classify_surge_level(25) == "high"
    assert classify_surge_level(44.9) == "high"
    assert classify_surge_level(45) == "critical"
    assert classify_surge_level(200) == "critical"


def test_haversine_known_distance() -> None:
    # Lisbon <-> Porto is roughly 275km as the crow flies.
    distance = haversine_km(38.7452, -9.1662, 41.1496, -8.6109)
    assert 260 <= distance <= 290


def test_weather_cache_falls_back_when_client_always_fails() -> None:
    cache = WeatherCache(_AlwaysFailingWeatherClient())
    cache.refresh()  # must not raise even though every branch fetch fails
    rain_mm, was_live = cache.get(_NEARBY_BRANCH_A)
    assert rain_mm == BASELINE_RAIN_MM
    assert was_live is False


def _mark_open(db_path: str, branch_id: str, desk_service_id: str) -> None:
    """Inserts a live is_open=True reading so tests aren't gated by whatever
    the real business-hours heuristic says at the moment they happen to run
    — these tests are about model/fallback/reroute behavior, not is_open."""
    with get_connection(db_path) as connection:
        upsert_branch(connection, branch_id, branch_id, "Lisboa", 38.7, -9.1)
        insert_queue_samples(
            connection,
            [
                QueueReading(
                    branch_id=branch_id,
                    desk_service_id=desk_service_id,
                    sampled_at=datetime.now(timezone.utc),
                    people_waiting=None,
                    last_ticket_called=None,
                    estimated_wait_minutes=None,
                    source="siga_live",
                    is_open=True,
                )
            ],
        )


def test_predict_falls_back_when_no_live_data_available(tmp_path) -> None:
    db_path = str(tmp_path / "empty_queue_history.db")
    _mark_open(db_path, _NEARBY_BRANCH_A, _SHARED_SERVICE)
    service = _make_service(db_path, _ConstantModel(15.0), weather_cache=WeatherCache(_AlwaysFailingWeatherClient()))

    predicted_wait, confidence = service.predict_wait_minutes(_NEARBY_BRANCH_A, _SHARED_SERVICE, datetime.now(timezone.utc))

    # Stub model ignores features and always returns this constant.
    assert predicted_wait == 15.0
    # No live people_waiting, no rolling stats, no live weather, no demand baseline => heavy discount.
    assert confidence < 0.5


def test_find_smart_reroute_recommends_faster_nearby_branch(tmp_path) -> None:
    db_path = str(tmp_path / "empty_queue_history.db")
    assert _NEARBY_BRANCH_B in BRANCHES_BY_ID  # sanity check the fixture branch still exists in the registry
    _mark_open(db_path, _NEARBY_BRANCH_A, _SHARED_SERVICE)
    _mark_open(db_path, _NEARBY_BRANCH_B, _SHARED_SERVICE)
    model = _PerBranchModel({_NEARBY_BRANCH_A: 60.0, _NEARBY_BRANCH_B: 10.0})
    service = _make_service(db_path, model, weather_cache=WeatherCache(_AlwaysFailingWeatherClient()))

    alternative = service.find_smart_reroute(
        _NEARBY_BRANCH_A, _SHARED_SERVICE, datetime.now(timezone.utc), current_wait_minutes=60.0
    )

    assert alternative is not None
    assert alternative["branch_id"] == _NEARBY_BRANCH_B
    assert alternative["time_saved_minutes"] > 0


def test_find_smart_reroute_returns_none_when_no_better_option(tmp_path) -> None:
    db_path = str(tmp_path / "empty_queue_history.db")
    _mark_open(db_path, _NEARBY_BRANCH_A, _SHARED_SERVICE)
    _mark_open(db_path, _NEARBY_BRANCH_B, _SHARED_SERVICE)
    # Every branch predicts the same wait, so nothing is a worthwhile reroute.
    model = _ConstantModel(20.0)
    service = _make_service(db_path, model, weather_cache=WeatherCache(_AlwaysFailingWeatherClient()))

    alternative = service.find_smart_reroute(
        _NEARBY_BRANCH_A, _SHARED_SERVICE, datetime.now(timezone.utc), current_wait_minutes=20.0
    )

    assert alternative is None


def test_predict_clamps_output_to_max_derived_wait_minutes(tmp_path) -> None:
    db_path = str(tmp_path / "empty_queue_history.db")
    _mark_open(db_path, _NEARBY_BRANCH_A, _SHARED_SERVICE)
    service = _make_service(db_path, _OverCapModel(), weather_cache=WeatherCache(_AlwaysFailingWeatherClient()))

    predicted_wait, _ = service.predict_wait_minutes(_NEARBY_BRANCH_A, _SHARED_SERVICE, datetime.now(timezone.utc))

    assert predicted_wait == MAX_DERIVED_WAIT_MINUTES


def test_ignores_stale_people_waiting_for_a_datetime_far_from_now(tmp_path) -> None:
    """A live reading exists, but the requested prediction datetime is far
    from wall-clock now (a different hour a month out) — reusing that
    reading would pair a stale queue-state value with an unrelated
    hour_of_day/calendar context, so it must not be used (config.NEAR_NOW_WINDOW_MINUTES).
    """
    db_path = str(tmp_path / "queue_history.db")
    with get_connection(db_path) as connection:
        upsert_branch(connection, _NEARBY_BRANCH_A, "Test Branch", "Lisboa", 38.7, -9.1)
        insert_queue_samples(
            connection,
            [
                QueueReading(
                    branch_id=_NEARBY_BRANCH_A,
                    desk_service_id=_SHARED_SERVICE,
                    sampled_at=datetime.now(timezone.utc),
                    people_waiting=999,
                    last_ticket_called=None,
                    estimated_wait_minutes=50.0,
                    source="siga_live",
                )
            ],
        )

    service = _make_service(db_path, _ConstantModel(10.0))
    far_future = datetime.now(timezone.utc) + timedelta(days=30)

    build_result = service._build_features(_NEARBY_BRANCH_A, _SHARED_SERVICE, far_future)

    assert build_result.used_live_people_waiting is False
    # Falls back to a demand-consistent estimate (config.BASELINE_ATTENDANCES,
    # since no demand_baseline_cache was supplied here), NOT a hardcoded 0 --
    # a hardcoded 0 paired with a busy branch's real historical_avg_attendances
    # was found 2026-07-27 to create an out-of-distribution feature
    # combination that made the model extrapolate erratically (negative raw
    # predictions on some hours).
    assert build_result.features["people_waiting"].iloc[0] == estimate_people_waiting(BASELINE_ATTENDANCES, far_future)


def test_uses_live_people_waiting_for_a_near_now_request(tmp_path) -> None:
    db_path = str(tmp_path / "queue_history.db")
    with get_connection(db_path) as connection:
        upsert_branch(connection, _NEARBY_BRANCH_A, "Test Branch", "Lisboa", 38.7, -9.1)
        insert_queue_samples(
            connection,
            [
                QueueReading(
                    branch_id=_NEARBY_BRANCH_A,
                    desk_service_id=_SHARED_SERVICE,
                    sampled_at=datetime.now(timezone.utc),
                    people_waiting=42,
                    last_ticket_called=None,
                    estimated_wait_minutes=30.0,
                    source="siga_live",
                )
            ],
        )

    service = _make_service(db_path, _ConstantModel(10.0))
    build_result = service._build_features(_NEARBY_BRANCH_A, _SHARED_SERVICE, datetime.now(timezone.utc))

    assert build_result.used_live_people_waiting is True
    assert build_result.features["people_waiting"].iloc[0] == 42


def _far_future_matching_weekday(target_weekday: int, hour: int) -> datetime:
    """A datetime far enough out to never hit the near-now live-data gate,
    landing on a specific weekday — for exercising the is_open heuristic
    fallback deterministically regardless of when the test suite runs."""
    base = datetime.now(timezone.utc) + timedelta(days=3650)
    days_ahead = (target_weekday - base.weekday()) % 7
    return (base + timedelta(days=days_ahead)).replace(hour=hour, minute=0, second=0, microsecond=0)


def test_minutes_to_nearest_snapshot_at_an_anchor_is_zero() -> None:
    anchor_hour, anchor_minute, _ = DIURNAL_SNAPSHOTS[0]
    assert minutes_to_nearest_snapshot(datetime(2026, 1, 5, anchor_hour, anchor_minute, tzinfo=timezone.utc)) == 0.0


def test_minutes_to_nearest_snapshot_picks_closest_anchor() -> None:
    # A small, fixed offset from a real anchor -- robust to DIURNAL_SNAPSHOTS'
    # exact spacing changing later, as long as anchors stay > 10 min apart.
    anchor_hour, anchor_minute, _ = DIURNAL_SNAPSHOTS[0]
    just_after_anchor = datetime(2026, 1, 5, anchor_hour, anchor_minute, tzinfo=timezone.utc) + timedelta(minutes=5)
    assert minutes_to_nearest_snapshot(just_after_anchor) == 5.0


def test_confidence_discounted_far_from_snapshot_anchor_hour(tmp_path) -> None:
    db_path = str(tmp_path / "empty_queue_history.db")
    _mark_open(db_path, _NEARBY_BRANCH_A, _SHARED_SERVICE)
    service = _make_service(db_path, _ConstantModel(10.0), weather_cache=WeatherCache(_AlwaysFailingWeatherClient()))

    # Same weekday, same everything else (empty DB, no live weather) -- the
    # only difference is proximity to a trained diurnal snapshot anchor.
    # Uses the midpoint between two real adjacent anchors (always > the
    # proximity threshold, by construction) rather than a hardcoded hour.
    base = _far_future_matching_weekday(0, 0)
    anchor_hour, anchor_minute, _ = DIURNAL_SNAPSHOTS[0]
    next_hour, next_minute, _ = DIURNAL_SNAPSHOTS[1]
    at_anchor = base.replace(hour=anchor_hour, minute=anchor_minute)
    at_next_anchor = base.replace(hour=next_hour, minute=next_minute)
    midpoint = at_anchor + (at_next_anchor - at_anchor) / 2

    _, confidence_at_anchor = service.predict_wait_minutes(_NEARBY_BRANCH_A, _SHARED_SERVICE, at_anchor)
    _, confidence_far_from_anchor = service.predict_wait_minutes(_NEARBY_BRANCH_A, _SHARED_SERVICE, midpoint)

    assert confidence_at_anchor - confidence_far_from_anchor == pytest.approx(SNAPSHOT_DISTANCE_CONFIDENCE_PENALTY)


def test_is_open_heuristic_true_on_weekday_business_hours(tmp_path) -> None:
    db_path = str(tmp_path / "empty_queue_history.db")
    service = _make_service(db_path, _ConstantModel(10.0))

    monday_11am = _far_future_matching_weekday(0, 11)
    is_open, used_live = service.get_is_open(_NEARBY_BRANCH_A, _SHARED_SERVICE, monday_11am)

    assert is_open is True
    assert used_live is False


def test_is_open_heuristic_false_on_weekend(tmp_path) -> None:
    db_path = str(tmp_path / "empty_queue_history.db")
    service = _make_service(db_path, _ConstantModel(10.0))

    saturday_11am = _far_future_matching_weekday(5, 11)
    is_open, used_live = service.get_is_open(_NEARBY_BRANCH_A, _SHARED_SERVICE, saturday_11am)

    assert is_open is False
    assert used_live is False


def test_predict_short_circuits_to_zero_when_closed(tmp_path) -> None:
    db_path = str(tmp_path / "empty_queue_history.db")
    # A model that would predict something big if actually invoked.
    service = _make_service(db_path, _OverCapModel())

    saturday_11am = _far_future_matching_weekday(5, 11)
    predicted_wait, confidence = service.predict_wait_minutes(_NEARBY_BRANCH_A, _SHARED_SERVICE, saturday_11am)

    assert predicted_wait == 0.0
    assert confidence > 0.5


def test_live_is_open_false_overrides_heuristic_for_near_now(tmp_path) -> None:
    """Even on an otherwise-open weekday/hour, a real live 'closed' reading
    for a near-now request must win over the fixed schedule heuristic."""
    db_path = str(tmp_path / "queue_history.db")
    with get_connection(db_path) as connection:
        upsert_branch(connection, _NEARBY_BRANCH_A, "Test Branch", "Lisboa", 38.7, -9.1)
        insert_queue_samples(
            connection,
            [
                QueueReading(
                    branch_id=_NEARBY_BRANCH_A,
                    desk_service_id=_SHARED_SERVICE,
                    sampled_at=datetime.now(timezone.utc),
                    people_waiting=None,
                    last_ticket_called=None,
                    estimated_wait_minutes=None,
                    source="siga_live",
                    is_open=False,
                )
            ],
        )

    service = _make_service(db_path, _ConstantModel(10.0))
    is_open, used_live = service.get_is_open(_NEARBY_BRANCH_A, _SHARED_SERVICE, datetime.now(timezone.utc))

    assert is_open is False
    assert used_live is True


def test_find_smart_reroute_skips_closed_candidate(tmp_path) -> None:
    """A closed candidate must never be recommended, even though a closed
    branch's own predict_wait_minutes short-circuits to an unbeatable 0.0."""
    db_path = str(tmp_path / "queue_history.db")
    with get_connection(db_path) as connection:
        upsert_branch(connection, _NEARBY_BRANCH_B, "Test Branch B", "Lisboa", 38.7, -9.1)
        insert_queue_samples(
            connection,
            [
                QueueReading(
                    branch_id=_NEARBY_BRANCH_B,
                    desk_service_id=_SHARED_SERVICE,
                    sampled_at=datetime.now(timezone.utc),
                    people_waiting=None,
                    last_ticket_called=None,
                    estimated_wait_minutes=None,
                    source="siga_live",
                    is_open=False,
                )
            ],
        )

    model = _PerBranchModel({_NEARBY_BRANCH_A: 30.0, _NEARBY_BRANCH_B: 5.0})
    service = _make_service(db_path, model, weather_cache=WeatherCache(_AlwaysFailingWeatherClient()))

    alternative = service.find_smart_reroute(
        _NEARBY_BRANCH_A, _SHARED_SERVICE, datetime.now(timezone.utc), current_wait_minutes=30.0
    )

    assert alternative is None or alternative["branch_id"] != _NEARBY_BRANCH_B
