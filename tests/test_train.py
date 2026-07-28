"""Unit tests for the proxy-vs-live sample weighting logic in pipeline/train.py."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from pipeline.train import chronological_split_by_source, compute_sample_weights, evaluate_by_source


class _ConstantModel:
    def __init__(self, constant: float) -> None:
        self.constant = constant

    def predict(self, X):
        return np.full(len(X), self.constant)


def _sampled_at_series(n: int, start: datetime, step: timedelta) -> list[datetime]:
    return [start + i * step for i in range(n)]


def test_chronological_split_by_source_gives_the_small_recent_source_real_train_exposure() -> None:
    # Mimics the real shape found 2026-07-27: historical_derived_proxy spans
    # ~2 years, siga_live is a tiny sliver of only the last few weeks. Under
    # one global chronological cutoff, every siga_live row would land in
    # test (it's always the most recent data) and none in train.
    now = datetime.now(timezone.utc)
    proxy_rows = pd.DataFrame(
        {
            "source": ["historical_derived_proxy"] * 1000,
            "sampled_at": _sampled_at_series(1000, now - timedelta(days=700), timedelta(hours=6)),
        }
    )
    live_rows = pd.DataFrame(
        {
            "source": ["siga_live"] * 20,
            "sampled_at": _sampled_at_series(20, now - timedelta(days=14), timedelta(hours=12)),
        }
    )
    frame = pd.concat([proxy_rows, live_rows], ignore_index=True)

    train_frame, test_frame = chronological_split_by_source(frame, test_fraction=0.2)

    live_in_train = (train_frame["source"] == "siga_live").sum()
    live_in_test = (test_frame["source"] == "siga_live").sum()
    assert live_in_train > 0, "siga_live got zero training rows -- the exact bug this split replaced"
    assert live_in_test > 0
    # Each source's own 80/20 split, not one global cutoff dominated by the
    # much larger proxy source.
    assert live_in_train == 16
    assert live_in_test == 4


def test_chronological_split_by_source_preserves_chronology_within_each_source() -> None:
    now = datetime.now(timezone.utc)
    frame = pd.concat(
        [
            pd.DataFrame({"source": ["siga_live"] * 10, "sampled_at": _sampled_at_series(10, now, timedelta(hours=1))}),
            pd.DataFrame(
                {"source": ["historical_derived_proxy"] * 10, "sampled_at": _sampled_at_series(10, now - timedelta(days=1), timedelta(hours=1))}
            ),
        ],
        ignore_index=True,
    )

    train_frame, test_frame = chronological_split_by_source(frame, test_fraction=0.2)

    for source in ("siga_live", "historical_derived_proxy"):
        train_times = train_frame.loc[train_frame["source"] == source, "sampled_at"]
        test_times = test_frame.loc[test_frame["source"] == source, "sampled_at"]
        assert train_times.max() < test_times.min(), f"{source} train/test rows are not chronologically ordered"


def test_live_rows_always_get_full_weight() -> None:
    frame = pd.DataFrame(
        [
            {"branch_id": "a", "desk_service_id": "svc", "source": "siga_live"},
            {"branch_id": "a", "desk_service_id": "svc", "source": "siga_live"},
        ]
    )
    weights = compute_sample_weights(frame, alpha=0.05)
    assert (weights == 1.0).all()


def test_proxy_weight_is_full_when_no_live_data_exists_for_that_combo() -> None:
    frame = pd.DataFrame(
        [
            {"branch_id": "a", "desk_service_id": "svc", "source": "historical_derived_proxy"},
            {"branch_id": "b", "desk_service_id": "other_svc", "source": "siga_live"},  # unrelated combo
        ]
    )
    weights = compute_sample_weights(frame, alpha=0.05)
    # branch a/svc has zero live samples of its own, so its proxy weight must be untouched (1.0),
    # regardless of live data existing for a completely different combo.
    assert weights[0] == 1.0


def test_proxy_weight_decays_as_that_combos_live_count_grows() -> None:
    rows = [{"branch_id": "a", "desk_service_id": "svc", "source": "historical_derived_proxy"}]
    rows += [{"branch_id": "a", "desk_service_id": "svc", "source": "siga_live"} for _ in range(20)]
    frame = pd.DataFrame(rows)

    weights = compute_sample_weights(frame, alpha=0.05)
    proxy_weight = weights[0]

    # 1 / (1 + 0.05 * 20) = 1 / 2 = 0.5
    assert abs(proxy_weight - 0.5) < 1e-9


def test_proxy_weight_never_affected_by_other_combos_live_counts() -> None:
    frame = pd.DataFrame(
        [
            {"branch_id": "a", "desk_service_id": "svc", "source": "historical_derived_proxy"},
            *[{"branch_id": "b", "desk_service_id": "other_svc", "source": "siga_live"} for _ in range(500)],
        ]
    )
    weights = compute_sample_weights(frame, alpha=0.05)
    # branch a/svc's proxy row must stay at full weight no matter how much
    # live data branch b/other_svc has accumulated — this is the whole point
    # of per-combo (not global) weighting.
    assert weights[0] == 1.0


def test_real_daily_avg_weight_scales_with_its_own_sample_size() -> None:
    frame = pd.DataFrame(
        [
            {"branch_id": "a", "desk_service_id": "svc", "source": "historical_real_daily_avg", "sample_size": 30},
            {"branch_id": "a", "desk_service_id": "svc", "source": "historical_real_daily_avg", "sample_size": 1},
        ]
    )
    weights = compute_sample_weights(frame, alpha=0.05, real_avg_reference_sample_size=30)

    # 30 / (30 + 30) = 0.5 -- a "meaningfully sized" day gets a real, non-trivial weight.
    assert abs(weights[0] - 0.5) < 1e-9
    # 1 / (1 + 30) ~= 0.032 -- an n=1 day (the Freixo de Espada a Cinta case
    # found in the real data) must be weighted far lower, not treated as equally reliable.
    assert weights[1] < 0.05
    assert weights[1] < weights[0]


def test_real_daily_avg_weight_is_not_decayed_by_live_count() -> None:
    # Unlike historical_derived_proxy, historical_real_daily_avg's weight
    # reflects how much *that specific row* can be trusted (its own sample
    # size), not how much live siga_live coverage exists for the combo --
    # there's no "combo" for a branch-level daily average to decay against.
    rows = [{"branch_id": "a", "desk_service_id": "svc", "source": "historical_real_daily_avg", "sample_size": 100}]
    rows += [{"branch_id": "a", "desk_service_id": "svc", "source": "siga_live", "sample_size": None} for _ in range(50)]
    frame = pd.DataFrame(rows)

    weights = compute_sample_weights(frame, alpha=0.05, real_avg_reference_sample_size=30)

    # 100 / (100 + 30) regardless of the 50 live rows for the same combo.
    assert abs(weights[0] - (100 / 130)) < 1e-9


def test_evaluate_by_source_computes_r2_when_target_varies() -> None:
    test_frame = pd.DataFrame(
        {"wait_time_minutes": [0.0, 5.0, 10.0, 15.0], "source": ["historical_derived_proxy"] * 4}
    )
    model = _ConstantModel(7.5)  # roughly the mean, so R^2 should be defined and computable

    results = evaluate_by_source(model, test_frame, feature_columns=[], target_column="wait_time_minutes")

    assert results["historical_derived_proxy"]["n"] == 4
    assert results["historical_derived_proxy"]["r2"] is not None
    assert results["historical_derived_proxy"]["note"] is None


def test_evaluate_by_source_flags_constant_target_as_undefined() -> None:
    # Exactly the real-world case that motivated this: a handful of siga_live
    # rows all showing the same (e.g. zero) actual wait — R^2 is mathematically
    # undefined here and must be flagged, not silently reported as some number.
    test_frame = pd.DataFrame({"wait_time_minutes": [0.0, 0.0, 0.0], "source": ["siga_live"] * 3})
    model = _ConstantModel(0.5)

    results = evaluate_by_source(model, test_frame, feature_columns=[], target_column="wait_time_minutes")

    assert results["siga_live"]["r2"] is None
    assert results["siga_live"]["note"] is not None
    assert results["siga_live"]["mae"] == 0.5


def test_evaluate_by_source_flags_too_few_rows() -> None:
    test_frame = pd.DataFrame({"wait_time_minutes": [3.0], "source": ["siga_live"]})
    model = _ConstantModel(3.0)

    results = evaluate_by_source(model, test_frame, feature_columns=[], target_column="wait_time_minutes")

    assert results["siga_live"]["r2"] is None
    assert "too few rows" in results["siga_live"]["note"]


def test_evaluate_by_source_segments_multiple_sources_independently() -> None:
    test_frame = pd.DataFrame(
        {
            "wait_time_minutes": [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
            "source": ["historical_derived_proxy"] * 4 + ["siga_live"] * 2,
        }
    )
    model = _ConstantModel(12.5)

    results = evaluate_by_source(model, test_frame, feature_columns=[], target_column="wait_time_minutes")

    assert set(results.keys()) == {"historical_derived_proxy", "siga_live"}
    assert results["historical_derived_proxy"]["n"] == 4
    assert results["siga_live"]["n"] == 2
