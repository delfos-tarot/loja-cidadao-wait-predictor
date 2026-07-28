"""Prediction and smart-reroute business logic.

Kept separate from FastAPI route handlers (api/main.py) so it can be unit
tested directly, without spinning up the app. All required "wait, this
external signal might be missing" fallbacks (live people_waiting, recent
rolling stats, live weather) live here, one level below the API surface, per
the project's graceful-degradation rule.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from config import (
    BASELINE_ATTENDANCES,
    BASELINE_RAIN_MM,
    BASELINE_WAIT_MINUTES,
    BRANCHES,
    BRANCHES_BY_ID,
    DEFAULT_DB_PATH,
    DIURNAL_SNAPSHOTS,
    MAX_DERIVED_WAIT_MINUTES,
    NEAR_NOW_WINDOW_MINUTES,
    OPERATING_HOURS_PER_DAY,
    REROUTE_RADIUS_KM,
    SNAPSHOT_DISTANCE_CONFIDENCE_PENALTY,
    SNAPSHOT_PROXIMITY_MINUTES,
    SURGE_CLOSED_LABEL,
)
from pipeline.db import get_latest_open_status, get_latest_people_waiting, get_rolling_wait_stats
from pipeline.feature_engineering import (
    WeatherClient,
    apply_categorical_dtypes,
    add_calendar_features,
    add_tax_deadline_features,
    estimate_is_open_heuristic,
)

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0

# Upper bound (exclusive) of predicted wait minutes for each surge label;
# anything at/above the last threshold is "critical". Placeholder values —
# tune against observed wait-time distributions once live data accumulates.
# A separate "closed" label (config.SURGE_CLOSED_LABEL) is used instead of
# these when the branch is known/assumed closed — see PredictionService.get_is_open.
SURGE_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (10.0, "low"),
    (25.0, "moderate"),
    (45.0, "high"),
)
SURGE_CRITICAL_LABEL = "critical"


def classify_surge_level(predicted_wait_minutes: float) -> str:
    for threshold, label in SURGE_THRESHOLDS:
        if predicted_wait_minutes < threshold:
            return label
    return SURGE_CRITICAL_LABEL


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))


def minutes_to_nearest_snapshot(target_datetime: datetime) -> float:
    """Minutes between target_datetime's time-of-day and the nearest
    DIURNAL_SNAPSHOTS anchor -- the hours the bulk of training labels
    actually exist at. Used to discount confidence for requests the model
    has comparatively little real support for."""
    target_minutes = target_datetime.hour * 60 + target_datetime.minute
    distances = [abs(target_minutes - (hour * 60 + minute)) for hour, minute, _ in DIURNAL_SNAPSHOTS]
    return float(min(distances))


def estimate_people_waiting(avg_attendances: float, target_datetime: datetime) -> int:
    """A people_waiting estimate consistent with historical_avg_attendances
    and time-of-day, for use when no live reading is available.

    Found 2026-07-27: the previous fallback (a hardcoded 0, regardless of
    avg_attendances) paired "zero people waiting" with a high
    historical_avg_attendances at a busy hour -- a combination that never
    occurs in training (a real high-volume midday row always has a
    correspondingly high people_waiting, since both are derived from the
    same attendance count in pipeline/demand_baseline.py). That
    out-of-distribution combination made the model extrapolate erratically,
    including negative raw predictions the API then silently clamped to
    0.0. This mirrors demand_baseline.py's own
    avg_hourly_attendance * volume_factor calculation, using the nearest
    snapshot's volume_factor for target_datetime's time-of-day, so the
    fallback feature vector looks like something the model actually saw.
    """
    nearest = min(DIURNAL_SNAPSHOTS, key=lambda s: abs((target_datetime.hour * 60 + target_datetime.minute) - (s[0] * 60 + s[1])))
    volume_factor = nearest[2]
    avg_hourly_attendance = avg_attendances / OPERATING_HOURS_PER_DAY
    return round(avg_hourly_attendance * volume_factor)


class WeatherCache:
    """In-memory rain_mm cache, refreshed periodically in the background so
    `/predict` never makes a live network call in the request path. Falls
    back to `config.BASELINE_RAIN_MM` for any branch not yet cached or if a
    refresh cycle fails outright.
    """

    def __init__(self, weather_client: WeatherClient) -> None:
        self._client = weather_client
        self._rain_mm_by_branch: dict[str, float] = {}

    def refresh(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        current_hour = pd.Timestamp(datetime.now(timezone.utc)).floor("h")
        for branch in BRANCHES:
            rain_frame = self._client.get_hourly_rain(branch.latitude, branch.longitude, today, today)
            if rain_frame is None or rain_frame.empty:
                continue
            match = rain_frame.loc[rain_frame["sampled_hour"] == current_hour, "rain_mm"]
            if not match.empty:
                self._rain_mm_by_branch[branch.branch_id] = float(match.iloc[0])

    def get(self, branch_id: str) -> tuple[float, bool]:
        """Returns (rain_mm, was_live_reading)."""
        if branch_id in self._rain_mm_by_branch:
            return self._rain_mm_by_branch[branch_id], True
        return BASELINE_RAIN_MM, False


class DemandBaselineCache:
    """In-memory copy of the `historical_demand_baseline` table (real
    dados.gov.pt attendance averages by branch/service/day-of-week). It's
    small and only changes when pipeline/demand_baseline.py is re-run
    offline, so it's loaded once at startup rather than queried per request.
    Falls back to config.BASELINE_ATTENDANCES for any combination not present
    (a new branch/service, or if the table hasn't been built yet).
    """

    def __init__(self, db_path: str) -> None:
        self._lookup: dict[tuple[str, str, int], float] = {}
        self.load(db_path)

    def load(self, db_path: str) -> None:
        try:
            with sqlite3.connect(db_path) as connection:
                frame = pd.read_sql_query(
                    "SELECT branch_id, desk_service_id, day_of_week, avg_attendances FROM historical_demand_baseline",
                    connection,
                )
        except Exception:
            logger.warning("historical_demand_baseline table unavailable; using baseline fallback", exc_info=True)
            return
        self._lookup = {
            (row.branch_id, row.desk_service_id, int(row.day_of_week)): float(row.avg_attendances)
            for row in frame.itertuples(index=False)
        }

    def get(self, branch_id: str, desk_service_id: str, day_of_week: int) -> tuple[float, bool]:
        """Returns (avg_attendances, was_real_reading)."""
        key = (branch_id, desk_service_id, day_of_week)
        if key in self._lookup:
            return self._lookup[key], True
        return BASELINE_ATTENDANCES, False


@dataclass
class _FeatureBuildResult:
    features: pd.DataFrame
    used_live_people_waiting: bool
    used_live_rolling_stats: bool
    used_live_weather: bool
    used_real_demand_baseline: bool
    is_open: bool
    used_live_is_open: bool


class PredictionService:
    def __init__(
        self,
        model_artifact: dict[str, Any],
        db_path: str = DEFAULT_DB_PATH,
        weather_cache: WeatherCache | None = None,
        demand_baseline_cache: DemandBaselineCache | None = None,
    ) -> None:
        self.model = model_artifact["model"]
        self.feature_columns: list[str] = model_artifact["feature_columns"]
        self.trained_at = model_artifact.get("trained_at")
        self.metrics: dict[str, float] = model_artifact.get("metrics", {})
        self.db_path = db_path
        self.weather_cache = weather_cache
        self.demand_baseline_cache = demand_baseline_cache

    def get_is_open(self, branch_id: str, desk_service_id: str, target_datetime: datetime) -> tuple[bool, bool]:
        """Returns (is_open, was_live_reading).

        Single source of truth for open/closed status, used by feature
        building, predict_wait_minutes (to short-circuit inference for a
        closed branch), and find_smart_reroute (to exclude closed
        candidates) — all three must agree, or a closed candidate could
        look like an attractive "0 minute wait" alternative.
        """
        is_near_now = abs((datetime.now(timezone.utc) - target_datetime).total_seconds()) <= NEAR_NOW_WINDOW_MINUTES * 60
        if is_near_now:
            live_is_open = get_latest_open_status(self.db_path, branch_id, desk_service_id)
            if live_is_open is not None:
                return live_is_open, True
        return estimate_is_open_heuristic(target_datetime), False

    def _build_features(self, branch_id: str, desk_service_id: str, target_datetime: datetime) -> _FeatureBuildResult:
        is_open, used_live_is_open = self.get_is_open(branch_id, desk_service_id, target_datetime)

        row = pd.DataFrame([{"branch_id": branch_id, "desk_service_id": desk_service_id, "sampled_at": target_datetime}])
        row = add_calendar_features(row)
        row["is_open"] = int(is_open)
        row = add_tax_deadline_features(row)

        if self.weather_cache is not None:
            rain_mm, used_live_weather = self.weather_cache.get(branch_id)
        else:
            rain_mm, used_live_weather = BASELINE_RAIN_MM, False
        row["rain_mm"] = rain_mm

        if self.demand_baseline_cache is not None:
            day_of_week = target_datetime.weekday()
            avg_attendances, used_real_demand_baseline = self.demand_baseline_cache.get(
                branch_id, desk_service_id, day_of_week
            )
        else:
            avg_attendances, used_real_demand_baseline = BASELINE_ATTENDANCES, False
        row["historical_avg_attendances"] = avg_attendances

        # "Latest known people_waiting" only reflects reality for a request
        # close to actual wall-clock now — reusing it for a different hour
        # or a future/past date pairs a stale queue-state value with an
        # hour_of_day/calendar context the model never saw together in
        # training. See config.NEAR_NOW_WINDOW_MINUTES for why.
        is_near_now = abs((datetime.now(timezone.utc) - target_datetime).total_seconds()) <= NEAR_NOW_WINDOW_MINUTES * 60
        people_waiting = get_latest_people_waiting(self.db_path, branch_id, desk_service_id)[0] if is_near_now else None
        used_live_people_waiting = people_waiting is not None
        row["people_waiting"] = (
            people_waiting if people_waiting is not None else estimate_people_waiting(avg_attendances, target_datetime)
        )

        avg_15min, avg_1h = get_rolling_wait_stats(self.db_path, branch_id, desk_service_id, target_datetime)
        used_live_rolling_stats = avg_15min is not None and avg_1h is not None
        row["rolling_15min_avg_wait"] = avg_15min if avg_15min is not None else BASELINE_WAIT_MINUTES
        row["rolling_1h_avg_wait"] = avg_1h if avg_1h is not None else BASELINE_WAIT_MINUTES

        row = apply_categorical_dtypes(row)
        return _FeatureBuildResult(
            features=row[self.feature_columns],
            used_live_people_waiting=used_live_people_waiting,
            used_live_rolling_stats=used_live_rolling_stats,
            used_live_weather=used_live_weather,
            used_real_demand_baseline=used_real_demand_baseline,
            is_open=is_open,
            used_live_is_open=used_live_is_open,
        )

    def predict_wait_minutes(self, branch_id: str, desk_service_id: str, target_datetime: datetime) -> tuple[float, float]:
        """Returns (predicted_wait_minutes, confidence_score).

        confidence_score is a deterministic heuristic (not a statistical
        prediction interval): it starts at 1.0 and is discounted for each
        external/live signal that had to fall back to a baseline, since a
        model built from more, fresher live signal is more trustworthy. It
        is also discounted when target_datetime falls far from the diurnal
        snapshot hours (9:30/12:30/15:30) the bulk of training labels are
        anchored to — see config.SNAPSHOT_PROXIMITY_MINUTES and
        minutes_to_nearest_snapshot's docstring for why: the model has
        materially less real support away from those hours.

        A closed branch short-circuits to (0.0, high confidence) without
        running the model at all — predicting a queue wait at a closed
        branch is meaningless, and the training data never modeled it.
        """
        is_open, used_live_is_open = self.get_is_open(branch_id, desk_service_id, target_datetime)
        if not is_open:
            # Deterministic (either a real live reading or a fixed schedule
            # assumption), not a modeled guess — high confidence either way.
            return 0.0, 0.95 if used_live_is_open else 0.85

        build_result = self._build_features(branch_id, desk_service_id, target_datetime)
        try:
            # Gradient-boosted trees aren't bounded by the training label
            # range even though every label was capped at
            # MAX_DERIVED_WAIT_MINUTES — clamp the output to match, so an
            # out-of-distribution feature combination can't surface a
            # prediction the training data never justified.
            raw_prediction = float(self.model.predict(build_result.features)[0])
            prediction = max(0.0, min(raw_prediction, MAX_DERIVED_WAIT_MINUTES))
        except Exception:
            logger.exception("Model inference failed for %s/%s; falling back to baseline", branch_id, desk_service_id)
            return BASELINE_WAIT_MINUTES, 0.2

        confidence = 1.0
        if not build_result.used_live_people_waiting:
            confidence -= 0.30
        if not build_result.used_live_rolling_stats:
            confidence -= 0.20
        if not build_result.used_live_weather:
            confidence -= 0.10
        if not build_result.used_real_demand_baseline:
            confidence -= 0.10
        if minutes_to_nearest_snapshot(target_datetime) > SNAPSHOT_PROXIMITY_MINUTES:
            confidence -= SNAPSHOT_DISTANCE_CONFIDENCE_PENALTY
        confidence = round(min(0.99, max(0.05, confidence)), 2)

        return round(prediction, 1), confidence

    def find_smart_reroute(
        self,
        branch_id: str,
        desk_service_id: str,
        target_datetime: datetime,
        current_wait_minutes: float,
    ) -> dict[str, Any] | None:
        """Evaluates every branch within REROUTE_RADIUS_KM offering the same
        desk service and returns the one with the largest predicted time
        savings, or None if nothing within range beats the current branch.
        """
        origin = BRANCHES_BY_ID.get(branch_id)
        if origin is None:
            return None

        best_alternative: dict[str, Any] | None = None
        best_time_saved = 0.0
        for candidate in BRANCHES:
            if candidate.branch_id == branch_id or desk_service_id not in candidate.desk_service_ids:
                continue

            distance_km = haversine_km(origin.latitude, origin.longitude, candidate.latitude, candidate.longitude)
            if distance_km > REROUTE_RADIUS_KM:
                continue

            # A closed candidate would otherwise look like an unbeatable
            # "0 minute wait" — predict_wait_minutes short-circuits closed
            # branches to 0.0, which is correct for *that* branch's own
            # response but must never win a reroute comparison against it.
            candidate_is_open, _ = self.get_is_open(candidate.branch_id, desk_service_id, target_datetime)
            if not candidate_is_open:
                continue

            candidate_wait, _ = self.predict_wait_minutes(candidate.branch_id, desk_service_id, target_datetime)
            time_saved = current_wait_minutes - candidate_wait
            if time_saved > best_time_saved:
                best_time_saved = time_saved
                best_alternative = {
                    "branch_id": candidate.branch_id,
                    "branch_name": candidate.name,
                    "distance_km": round(distance_km, 2),
                    "predicted_wait_minutes": candidate_wait,
                    "time_saved_minutes": round(time_saved, 1),
                }

        return best_alternative
