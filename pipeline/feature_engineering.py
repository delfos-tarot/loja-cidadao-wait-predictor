"""Feature engineering: calendar, tax-deadline, weather, and lag/rolling signals.

Converts raw queue_samples rows into the feature matrix consumed by the
XGBoost wait-time model. The same `QueueFeatureTransformer` is used both by
`pipeline/train.py` (offline, batched) and `api/main.py` (online, single-row)
so training and inference never drift apart on feature definitions.

Weather enrichment degrades gracefully: if the Open-Meteo API is unreachable,
rate-limited, or returns no data for a given point, `rain_mm` falls back to
`config.BASELINE_RAIN_MM` rather than failing the pipeline (architectural
rule: missing external signals must never break a prediction).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config import (
    ASSUMED_BUSINESS_DAYS,
    ASSUMED_BUSINESS_HOUR_END,
    ASSUMED_BUSINESS_HOUR_START,
    BASELINE_ATTENDANCES,
    BASELINE_RAIN_MM,
    BASELINE_WAIT_MINUTES,
    BRANCHES_BY_ID,
    DEFAULT_DB_PATH,
    DESK_SERVICES,
    IMI_DEADLINE_MONTHS_DAYS,
    IRS_DEADLINE_MONTH_DAY,
)

logger = logging.getLogger(__name__)

# branch_id / desk_service_id are cast to a fixed categorical dtype (see
# `apply_categorical_dtypes`) so XGBoost can learn each branch/service's own
# baseline demand and per-ticket service time, instead of only ever seeing an
# undifferentiated people_waiting count.
CATEGORICAL_COLUMNS: list[str] = ["branch_id", "desk_service_id"]

FEATURE_COLUMNS: list[str] = [
    *CATEGORICAL_COLUMNS,
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "is_payday_week",
    "days_until_irs_deadline",
    "days_until_imi_deadline",
    "rain_mm",
    "people_waiting",
    "historical_avg_attendances",
    "is_open",
    "rolling_15min_avg_wait",
    "rolling_1h_avg_wait",
]

TARGET_COLUMN = "wait_time_minutes"


# --------------------------------------------------------------------------
# Calendar signals
# --------------------------------------------------------------------------

def add_calendar_features(frame: pd.DataFrame, timestamp_column: str = "sampled_at") -> pd.DataFrame:
    frame = frame.copy()
    timestamps = pd.to_datetime(frame[timestamp_column], utc=True)
    frame["hour_of_day"] = timestamps.dt.hour
    frame["day_of_week"] = timestamps.dt.dayofweek
    frame["month"] = timestamps.dt.month
    frame["is_weekend"] = (timestamps.dt.dayofweek >= 5).astype(int)
    frame["is_payday_week"] = (timestamps.dt.day >= 25).astype(int)
    return frame


def estimate_is_open_heuristic(timestamp) -> bool:
    """Scalar counterpart to add_is_open_feature's vectorized heuristic, used
    by the API for single-datetime lookups. Keep the two in sync."""
    return (
        timestamp.weekday() in ASSUMED_BUSINESS_DAYS
        and ASSUMED_BUSINESS_HOUR_START <= timestamp.hour < ASSUMED_BUSINESS_HOUR_END
    )


def add_is_open_feature(frame: pd.DataFrame, timestamp_column: str = "sampled_at") -> pd.DataFrame:
    """Uses a real observed is_open value when one exists on the row (from
    live siga_live polling); falls back to a fixed Mon-Fri/9-17 heuristic
    otherwise (config.ASSUMED_BUSINESS_*), since no real per-branch schedule
    data exists yet for historical_derived_proxy/synthetic_bootstrap rows.
    """
    frame = frame.copy()
    timestamps = pd.to_datetime(frame[timestamp_column], utc=True)
    heuristic = (
        timestamps.dt.dayofweek.isin(ASSUMED_BUSINESS_DAYS)
        & (timestamps.dt.hour >= ASSUMED_BUSINESS_HOUR_START)
        & (timestamps.dt.hour < ASSUMED_BUSINESS_HOUR_END)
    ).astype(int)

    if "is_open" in frame.columns:
        real_is_open = pd.to_numeric(frame["is_open"], errors="coerce")
        frame["is_open"] = real_is_open.fillna(heuristic).astype(int)
    else:
        frame["is_open"] = heuristic
    return frame


# --------------------------------------------------------------------------
# Tax / administrative deadline signals
# --------------------------------------------------------------------------

def _days_until_next_occurrence(reference: pd.Timestamp, month: int, day: int) -> int:
    candidate = reference.replace(month=month, day=day, hour=0, minute=0, second=0, microsecond=0)
    if candidate < reference:
        candidate = candidate.replace(year=candidate.year + 1)
    return (candidate - reference).days


def _days_until_irs_deadline(reference: pd.Timestamp) -> int:
    month, day = IRS_DEADLINE_MONTH_DAY
    return _days_until_next_occurrence(reference, month, day)


def _days_until_imi_deadline(reference: pd.Timestamp) -> int:
    return min(_days_until_next_occurrence(reference, month, day) for month, day in IMI_DEADLINE_MONTHS_DAYS)


def add_tax_deadline_features(frame: pd.DataFrame, timestamp_column: str = "sampled_at") -> pd.DataFrame:
    frame = frame.copy()
    timestamps = pd.to_datetime(frame[timestamp_column], utc=True)
    frame["days_until_irs_deadline"] = timestamps.apply(_days_until_irs_deadline)
    frame["days_until_imi_deadline"] = timestamps.apply(_days_until_imi_deadline)
    return frame


# --------------------------------------------------------------------------
# Historical demand-baseline signal (real dados.gov.pt attendance data)
# --------------------------------------------------------------------------

def add_demand_baseline_feature(
    frame: pd.DataFrame,
    db_path: str = DEFAULT_DB_PATH,
    timestamp_column: str = "sampled_at",
) -> pd.DataFrame:
    """Merges in `historical_avg_attendances`, the real average daily
    attendance count for this (branch, service, day_of_week) computed by
    pipeline/demand_baseline.py from dados.gov.pt data. Falls back to
    `config.BASELINE_ATTENDANCES` if the lookup table doesn't exist yet or has
    no row for a given combination (architectural rule: missing historical
    context must never break a prediction).
    """
    frame = frame.copy()
    timestamps = pd.to_datetime(frame[timestamp_column], utc=True)
    frame["_day_of_week"] = timestamps.dt.dayofweek

    try:
        with sqlite3.connect(db_path) as connection:
            baseline = pd.read_sql_query(
                "SELECT branch_id, desk_service_id, day_of_week, avg_attendances FROM historical_demand_baseline",
                connection,
            )
    except Exception:
        logger.warning("historical_demand_baseline table unavailable; using baseline fallback", exc_info=True)
        baseline = pd.DataFrame(columns=["branch_id", "desk_service_id", "day_of_week", "avg_attendances"])

    merged = frame.merge(
        baseline,
        left_on=["branch_id", "desk_service_id", "_day_of_week"],
        right_on=["branch_id", "desk_service_id", "day_of_week"],
        how="left",
    )
    fallback = merged["avg_attendances"].mean()
    if pd.isna(fallback):
        fallback = BASELINE_ATTENDANCES
    frame["historical_avg_attendances"] = merged["avg_attendances"].fillna(fallback).to_numpy()
    frame = frame.drop(columns=["_day_of_week"])
    return frame


# --------------------------------------------------------------------------
# Weather signals (Open-Meteo)
# --------------------------------------------------------------------------

class WeatherClient:
    """Thin wrapper around the Open-Meteo Forecast API (openmeteo-requests client).

    A single endpoint (`start_date`/`end_date` on the forecast API) serves
    both recent-past and forecasted rainfall, so training-time and
    inference-time weather lookups stay consistent. All network failures are
    caught and logged; callers get `None` back and must apply the baseline
    fallback themselves.
    """

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, cache_path: str = "data/.weather_cache", cache_expire_seconds: int = 3600) -> None:
        self._client = None
        self._cache_path = cache_path
        self._cache_expire_seconds = cache_expire_seconds

    def _get_client(self):
        if self._client is None:
            import openmeteo_requests
            import requests_cache
            from retry_requests import retry

            Path(self._cache_path).parent.mkdir(parents=True, exist_ok=True)
            cache_session = requests_cache.CachedSession(self._cache_path, expire_after=self._cache_expire_seconds)
            retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
            self._client = openmeteo_requests.Client(session=retry_session)
        return self._client

    def get_hourly_rain(self, latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame | None:
        """Returns a DataFrame with columns [sampled_hour, rain_mm], or None on failure."""
        try:
            client = self._get_client()
            responses = client.weather_api(
                self.BASE_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "hourly": "rain",
                    "start_date": start_date,
                    "end_date": end_date,
                    "timezone": "UTC",
                },
            )
            hourly = responses[0].Hourly()
            rain = hourly.Variables(0).ValuesAsNumpy()
            timestamps = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )
            return pd.DataFrame({"sampled_hour": timestamps, "rain_mm": rain})
        except Exception:
            logger.warning(
                "Weather fetch failed for (%s, %s) [%s..%s]; falling back to baseline",
                latitude, longitude, start_date, end_date, exc_info=True,
            )
            return None


def add_weather_features(
    frame: pd.DataFrame,
    weather_client: WeatherClient | None = None,
    timestamp_column: str = "sampled_at",
) -> pd.DataFrame:
    frame = frame.copy()
    if weather_client is None:
        weather_client = WeatherClient()

    timestamps = pd.to_datetime(frame[timestamp_column], utc=True)
    sampled_hour = timestamps.dt.floor("h")
    rain_mm = pd.Series(np.nan, index=frame.index, dtype=float)

    for branch_id, branch_index in frame.groupby("branch_id").groups.items():
        branch = BRANCHES_BY_ID.get(branch_id)
        if branch is None:
            continue
        branch_hours = sampled_hour.loc[branch_index]
        start_date = branch_hours.min().strftime("%Y-%m-%d")
        end_date = branch_hours.max().strftime("%Y-%m-%d")
        rain_frame = weather_client.get_hourly_rain(branch.latitude, branch.longitude, start_date, end_date)
        if rain_frame is None or rain_frame.empty:
            continue
        rain_lookup = rain_frame.set_index("sampled_hour")["rain_mm"]
        rain_mm.loc[branch_index] = branch_hours.map(rain_lookup)

    frame["rain_mm"] = rain_mm.fillna(BASELINE_RAIN_MM)
    return frame


# --------------------------------------------------------------------------
# Lag / rolling wait-time signals
# --------------------------------------------------------------------------

def add_lag_rolling_features(
    frame: pd.DataFrame,
    timestamp_column: str = "sampled_at",
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    frame = frame.sort_values(["branch_id", "desk_service_id", timestamp_column]).copy()

    def _rolling_for_group(group: pd.DataFrame) -> pd.DataFrame:
        indexed = group.set_index(timestamp_column)
        # Shift by one observation before rolling so the window only ever
        # looks at *past* wait times, never the current (unknown) one.
        shifted = indexed[target_column].shift(1)
        group = group.assign(
            rolling_15min_avg_wait=shifted.rolling("15min").mean().to_numpy(),
            rolling_1h_avg_wait=shifted.rolling("1h").mean().to_numpy(),
        )
        return group

    frame = frame.groupby(["branch_id", "desk_service_id"], group_keys=False).apply(
        _rolling_for_group, include_groups=False
    ).join(frame[["branch_id", "desk_service_id"]])
    # groupby(...).apply(...) reorders rows by group key; restore the caller's
    # original row order so positional alignment with other columns is safe.
    frame = frame.sort_index()

    fallback = frame[target_column].mean()
    if pd.isna(fallback):
        fallback = BASELINE_WAIT_MINUTES
    frame["rolling_15min_avg_wait"] = frame["rolling_15min_avg_wait"].fillna(fallback)
    frame["rolling_1h_avg_wait"] = frame["rolling_1h_avg_wait"].fillna(fallback)
    return frame


# --------------------------------------------------------------------------
# Combined transformer
# --------------------------------------------------------------------------

def apply_categorical_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Casts branch_id/desk_service_id to a fixed categorical dtype.

    Using a fixed category set (from config, not whatever happens to appear
    in a given batch) guarantees training and single-row inference encode
    these columns identically even when a batch doesn't include every branch
    or service.
    """
    frame = frame.copy()
    frame["branch_id"] = pd.Categorical(frame["branch_id"], categories=list(BRANCHES_BY_ID.keys()))
    frame["desk_service_id"] = pd.Categorical(frame["desk_service_id"], categories=list(DESK_SERVICES))
    return frame


def build_feature_matrix(
    frame: pd.DataFrame,
    weather_client: WeatherClient | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Runs the full feature pipeline over raw queue_samples-shaped rows.

    Expects columns: branch_id, desk_service_id, sampled_at, people_waiting,
    and (for training) wait_time_minutes. Returns the original columns plus
    every column in FEATURE_COLUMNS.
    """
    frame = add_calendar_features(frame)
    frame = add_is_open_feature(frame)
    frame = add_tax_deadline_features(frame)
    frame = add_demand_baseline_feature(frame, db_path=db_path)
    frame = add_weather_features(frame, weather_client=weather_client)
    if TARGET_COLUMN in frame.columns:
        frame = add_lag_rolling_features(frame)
    else:
        frame["rolling_15min_avg_wait"] = BASELINE_WAIT_MINUTES
        frame["rolling_1h_avg_wait"] = BASELINE_WAIT_MINUTES

    if "people_waiting" not in frame.columns:
        frame["people_waiting"] = np.nan
    # .astype(float) guards against an all-NULL SQL-sourced column coming back
    # as dtype=object (fillna alone won't upcast it), which XGBoost rejects.
    frame["people_waiting"] = pd.to_numeric(frame["people_waiting"], errors="coerce")
    frame["people_waiting"] = frame["people_waiting"].fillna(frame["people_waiting"].mean()).fillna(0).astype(float)

    frame = apply_categorical_dtypes(frame)
    return frame


class QueueFeatureTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer wrapping `build_feature_matrix`.

    Kept stateless (fit is a no-op) so the exact same transformer instance can
    be reused for both offline training and online single-row inference
    without any risk of train/serve feature skew.
    """

    def __init__(self, weather_client: WeatherClient | None = None, db_path: str = DEFAULT_DB_PATH) -> None:
        self.weather_client = weather_client
        self.db_path = db_path

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "QueueFeatureTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        enriched = build_feature_matrix(X, weather_client=self.weather_client, db_path=self.db_path)
        return enriched[FEATURE_COLUMNS]
