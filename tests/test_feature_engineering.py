"""Unit tests for feature fallbacks -- the train/serve skew and leakage
surface found 2026-07-31."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from config import BASELINE_WAIT_MINUTES
from pipeline.feature_engineering import (
    add_lag_rolling_features,
    estimate_people_waiting,
    estimate_people_waiting_series,
)


def test_rolling_fallback_is_the_serving_constant_not_the_target_mean() -> None:
    # The bug: fallback used to be frame[target].mean(), which (a) differs
    # from what api/service.py fills in at inference and (b) is computed over
    # train+test together, leaking held-out labels into a feature.
    now = datetime.now(timezone.utc)
    frame = pd.DataFrame(
        {
            "branch_id": ["b1", "b1"],
            "desk_service_id": ["s1", "s1"],
            # Far apart, so neither row has a prior observation within 1h.
            "sampled_at": [now - timedelta(days=3), now],
            "wait_time_minutes": [900.0, 900.0],
        }
    )

    out = add_lag_rolling_features(frame)

    # Target mean here is 900.0 -- if that leaked in, these would be 900.0.
    assert out["rolling_15min_avg_wait"].tolist() == [BASELINE_WAIT_MINUTES] * 2
    assert out["rolling_1h_avg_wait"].tolist() == [BASELINE_WAIT_MINUTES] * 2


def test_rolling_fallback_does_not_depend_on_other_rows_targets() -> None:
    # Directly asserts the leak is gone: changing held-out target values must
    # not change any feature value.
    now = datetime.now(timezone.utc)

    def build(target_values: list[float]) -> pd.DataFrame:
        return add_lag_rolling_features(
            pd.DataFrame(
                {
                    "branch_id": ["b1", "b1"],
                    "desk_service_id": ["s1", "s1"],
                    "sampled_at": [now - timedelta(days=3), now],
                    "wait_time_minutes": target_values,
                }
            )
        )

    low = build([1.0, 2.0])
    high = build([500.0, 900.0])

    assert low["rolling_1h_avg_wait"].tolist() == high["rolling_1h_avg_wait"].tolist()


def test_real_prior_observation_is_still_used_over_the_fallback() -> None:
    # Guard against "fix the fallback, break the feature": a genuine recent
    # reading must still flow through.
    now = datetime.now(timezone.utc)
    frame = pd.DataFrame(
        {
            "branch_id": ["b1", "b1"],
            "desk_service_id": ["s1", "s1"],
            "sampled_at": [now - timedelta(minutes=10), now],
            "wait_time_minutes": [42.0, 7.0],
        }
    )

    out = add_lag_rolling_features(frame).sort_values("sampled_at")

    assert out["rolling_15min_avg_wait"].iloc[0] == BASELINE_WAIT_MINUTES  # no prior reading
    assert out["rolling_15min_avg_wait"].iloc[1] == 42.0  # the real prior reading, not the fallback


def test_vectorized_people_waiting_matches_the_scalar_serving_version() -> None:
    # These two must never diverge -- training uses the vectorized one and
    # api/service.py uses the scalar one, which is exactly how the original
    # skew arose.
    timestamps = [datetime(2026, 8, 3, hour, 30, tzinfo=timezone.utc) for hour in range(9, 17)]
    attendances = [80.0, 250.0, 918.0, 40.0, 1500.0, 300.0, 610.0, 95.0]

    vectorized = estimate_people_waiting_series(
        pd.Series(attendances), pd.Series(pd.to_datetime(timestamps, utc=True))
    )
    scalar = [estimate_people_waiting(a, t) for a, t in zip(attendances, timestamps)]

    assert vectorized.tolist() == [float(v) for v in scalar]
