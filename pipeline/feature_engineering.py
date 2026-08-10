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
from datetime import datetime
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
    DIURNAL_SNAPSHOTS,
    IMI_DEADLINE_MONTHS_DAYS,
    IRS_DEADLINE_MONTH_DAY,
    OPERATING_HOURS_PER_DAY,
)
from pipeline.holidays_pt import add_holiday_features, holiday_closure_mask, is_closed_for_holiday

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
    # Holiday-adjacent days, not holidays themselves — a holiday row does not
    # exist in this corpus, so `is_holiday` would be constant-0 and
    # unlearnable. Holidays act on `is_open` instead. Both flags below carry a
    # large measured effect (~+25% wait each); see pipeline/holidays_pt.py.
    "is_bridge_day",
    "is_post_holiday",
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


def estimate_is_open_heuristic(timestamp, branch_id: str | None = None) -> bool:
    """Scalar counterpart to add_is_open_feature's vectorized heuristic, used
    by the API for single-datetime lookups. Keep the two in sync.

    `branch_id` is optional so existing callers keep working, but passing it
    is what makes the holiday check municipality-aware — without it only
    national holidays are caught, and a Porto branch still reports as open on
    São João.
    """
    if is_closed_for_holiday(branch_id, timestamp.date()):
        return False
    return (
        timestamp.weekday() in ASSUMED_BUSINESS_DAYS
        and ASSUMED_BUSINESS_HOUR_START <= timestamp.hour < ASSUMED_BUSINESS_HOUR_END
    )


def add_is_open_feature(frame: pd.DataFrame, timestamp_column: str = "sampled_at") -> pd.DataFrame:
    """Uses a real observed is_open value when one exists on the row (from
    live siga_live polling); falls back to a fixed Mon-Fri/9-17 heuristic
    otherwise (config.ASSUMED_BUSINESS_*), since no real per-branch schedule
    data exists yet for historical_derived_proxy/synthetic_bootstrap rows.

    National and municipal holidays close the branch in the *heuristic* only.
    A real observed value still wins: if live SIGA polling reports a desk open
    on a holiday, that is a measurement and this is a calendar guess, so the
    measurement is kept. Ordering matters here — the holiday mask is folded
    into `heuristic` before `fillna`, never applied to the merged result.
    """
    frame = frame.copy()
    timestamps = pd.to_datetime(frame[timestamp_column], utc=True)
    heuristic = (
        timestamps.dt.dayofweek.isin(ASSUMED_BUSINESS_DAYS)
        & (timestamps.dt.hour >= ASSUMED_BUSINESS_HOUR_START)
        & (timestamps.dt.hour < ASSUMED_BUSINESS_HOUR_END)
        & ~holiday_closure_mask(
            frame.assign(_holiday_date=timestamps.dt.date),
            date_column="_holiday_date",
        )
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

    Lives here rather than in api/service.py (where it was originally
    written) so training and serving share one definition -- see
    estimate_people_waiting_series below for why that matters.
    """
    nearest = min(DIURNAL_SNAPSHOTS, key=lambda s: abs((target_datetime.hour * 60 + target_datetime.minute) - (s[0] * 60 + s[1])))
    volume_factor = nearest[2]
    avg_hourly_attendance = avg_attendances / OPERATING_HOURS_PER_DAY
    return round(avg_hourly_attendance * volume_factor)


def estimate_people_waiting_series(avg_attendances: pd.Series, timestamps: pd.Series) -> pd.Series:
    """Vectorized `estimate_people_waiting`, for filling the column across
    millions of training rows without a per-row Python call.

    Exists because training previously filled a missing `people_waiting`
    with the *frame mean* while serving used `estimate_people_waiting` --
    a train/serve skew found 2026-07-31 affecting 100% of the ~862k
    `historical_real_daily_avg` rows (IALC-M carries no queue-length
    field, so every one of them is null). That frame mean was also
    computed over train+test together, making it a leak. Both paths now
    call the same formula; `tests/test_feature_engineering.py` asserts
    this matches the scalar version exactly.
    """
    minutes_of_day = timestamps.dt.hour * 60 + timestamps.dt.minute
    snapshot_minutes = np.array([s[0] * 60 + s[1] for s in DIURNAL_SNAPSHOTS])
    volume_factors = np.array([s[2] for s in DIURNAL_SNAPSHOTS])
    nearest_index = np.abs(minutes_of_day.to_numpy()[:, None] - snapshot_minutes[None, :]).argmin(axis=1)
    factors = volume_factors[nearest_index]
    return ((avg_attendances.to_numpy() / OPERATING_HOURS_PER_DAY) * factors).round()


def add_lag_rolling_features(
    frame: pd.DataFrame,
    timestamp_column: str = "sampled_at",
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    frame = frame.sort_values(["branch_id", "desk_service_id", timestamp_column]).copy()

    def _rolling_for_group(group: pd.DataFrame) -> pd.DataFrame:
        indexed = group.set_index(timestamp_column)
        # closed="left" makes each window [t - span, t) -- every strictly
        # earlier observation inside the span, and never the current
        # (unknown) one. Replaced a `shift(1).rolling(...)` on 2026-07-31,
        # which was subtly wrong: shift(1) moves the previous observation's
        # VALUE onto the current row's timestamp, so it always landed inside
        # the window no matter how old it really was. A combo last seen three
        # days ago still got that stale reading reported as its "average wait
        # over the last hour". That was both a mislabeled feature and a third
        # train/serve skew -- pipeline/db.py's get_rolling_wait_stats does a
        # genuine time-bounded query at inference and correctly returns
        # nothing when the window is empty, so training saw stale values
        # where serving sees the fallback.
        series = indexed[target_column]
        group = group.assign(
            rolling_15min_avg_wait=series.rolling("15min", closed="left").mean().to_numpy(),
            rolling_1h_avg_wait=series.rolling("1h", closed="left").mean().to_numpy(),
        )
        return group

    frame = frame.groupby(["branch_id", "desk_service_id"], group_keys=False).apply(
        _rolling_for_group, include_groups=False
    ).join(frame[["branch_id", "desk_service_id"]])
    # groupby(...).apply(...) reorders rows by group key; restore the caller's
    # original row order so positional alignment with other columns is safe.
    frame = frame.sort_index()

    # config.BASELINE_WAIT_MINUTES, NOT the observed mean of the target.
    # Two bugs fixed here on 2026-07-31, both from the old
    # `fallback = frame[target_column].mean()`:
    #   1. Train/serve skew. api/service.py fills these same two features
    #      with BASELINE_WAIT_MINUTES (20.0) when no recent reading exists,
    #      while training filled them with the dataset's target mean
    #      (~4.985) -- so the model learned "~5 means no recent data" and
    #      was then handed 20.0 at inference, which it reads as a real
    #      recent 20-minute wait. This affected 11.2% of training rows,
    #      30.6% of siga_live rows, and *every* future-dated prediction
    #      (rolling stats always fall back there). Same class of bug as the
    #      people_waiting fallback documented in CLAUDE.md, in a different
    #      feature.
    #   2. Target leakage. That mean was computed over the whole frame,
    #      train and test together, so a feature's value was derived from
    #      held-out labels -- weak (one global constant, not per-row
    #      information) but enough to make any metric optimistic by an
    #      unknown amount.
    # A fixed config constant has neither problem: identical in training
    # and serving, and independent of the data.
    frame["rolling_15min_avg_wait"] = frame["rolling_15min_avg_wait"].fillna(BASELINE_WAIT_MINUTES)
    frame["rolling_1h_avg_wait"] = frame["rolling_1h_avg_wait"].fillna(BASELINE_WAIT_MINUTES)
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
    frame = add_holiday_features(frame)
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
    # Left as NaN when genuinely unknown -- XGBoost handles missing values
    # natively and learns its own default split direction, which is the
    # honest encoding of "no queue-length reading exists".
    #
    # Two rejected alternatives, both tried and measured on 2026-07-31:
    #   - The original `fillna(frame["people_waiting"].mean())`: a train+test
    #     leak, and a train/serve skew (serving derives an estimate instead).
    #   - Filling via estimate_people_waiting_series, to mirror serving: much
    #     worse (historical_real_daily_avg R^2 0.469 -> -1.22). That tier is
    #     ~862k rows -- 100% of IALC-M, which carries no queue-length field --
    #     and its label is a DAILY AVERAGE, deliberately stamped with an
    #     arbitrary rotated hour (see pipeline/ingest_real_wait_times.py's
    #     date.toordinal() % N). Deriving people_waiting from that arbitrary
    #     hour's volume_factor injects a strongly hour-varying feature against
    #     a label with no hour dependence -- noise, and shaped by the
    #     DIURNAL_SNAPSHOTS curve that config.py flags as inverted.
    # No skew is reintroduced: api/service.py always supplies either a live
    # reading or an estimate, so NaN never occurs at serving time -- these
    # rows teach the wait-level relationship, not the people_waiting one.
    frame["people_waiting"] = frame["people_waiting"].astype(float)

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
