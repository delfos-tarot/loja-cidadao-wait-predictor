"""Unit tests for the proxy-vs-live sample weighting logic in pipeline/train.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.train import compute_sample_weights, evaluate_by_source


class _ConstantModel:
    def __init__(self, constant: float) -> None:
        self.constant = constant

    def predict(self, X):
        return np.full(len(X), self.constant)


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
