"""Unit tests for pipeline/calibrate_constants.py's fitting logic. Builds
small synthetic panels directly (bypassing the real SLC-M/IALC-M parquet
files) so these stay fast and hermetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.calibrate_constants import (
    CATEGORIES,
    _desks_column,
    chronological_split,
    fit_constants,
    predict_branch_day_wait,
    weighted_rmse,
)


def _synthetic_panel(mu_by_category: dict[str, float], n_rows: int = 60, seed: int = 0) -> pd.DataFrame:
    """Builds a panel where avg_wait_minutes is generated EXACTLY by
    predict_branch_day_wait at the given ground-truth mu, with varying
    volume per row/category so the fit is actually identifiable."""
    rng = np.random.default_rng(seed)
    panel = pd.DataFrame(
        {
            "branch_id": [f"branch_{i % 5}" for i in range(n_rows)],
            "date": pd.date_range("2026-01-01", periods=n_rows, freq="D"),
        }
    )
    for category in CATEGORIES:
        panel[category] = rng.integers(5, 500, size=n_rows).astype(float)
        panel[_desks_column(category)] = rng.integers(3, 20, size=n_rows)
    panel["total_attendances"] = panel[list(CATEGORIES)].sum(axis=1)
    panel["avg_wait_minutes"] = predict_branch_day_wait(panel, mu_by_category)
    return panel


def test_chronological_split_respects_fraction_and_order() -> None:
    panel = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=100, freq="D")})
    train, holdout = chronological_split(panel, holdout_fraction=0.2)

    assert len(train) == 80
    assert len(holdout) == 20
    assert train["date"].max() < holdout["date"].min()


def test_predict_branch_day_wait_increases_with_mu() -> None:
    panel = pd.DataFrame(
        {
            "IRN": [100.0],
            "GENERAL_OTHER": [0.0],
            "OTHER_SPECIALIZED": [0.0],
            _desks_column("IRN"): [10],
            _desks_column("GENERAL_OTHER"): [10],
            _desks_column("OTHER_SPECIALIZED"): [10],
            "total_attendances": [100.0],
        }
    )
    low = predict_branch_day_wait(panel, {"IRN": 5.0, "GENERAL_OTHER": 5.0, "OTHER_SPECIALIZED": 5.0})
    high = predict_branch_day_wait(panel, {"IRN": 20.0, "GENERAL_OTHER": 20.0, "OTHER_SPECIALIZED": 20.0})
    assert high[0] > low[0]


def test_weighted_rmse_is_zero_at_ground_truth() -> None:
    true_mu = {"IRN": 12.0, "GENERAL_OTHER": 6.0, "OTHER_SPECIALIZED": 22.0}
    panel = _synthetic_panel(true_mu)

    assert weighted_rmse(panel, true_mu) == pytest.approx(0.0, abs=1e-9)


def test_fit_constants_recovers_ground_truth_and_moves_off_initial_guess() -> None:
    # Regression test for the bug found 2026-07-27: an earlier version of
    # this script re-derived desks from the trial mu at every iteration,
    # which made math.ceil() poison the finite-difference Jacobian and
    # caused least_squares to falsely "converge" after one near-zero step,
    # right at the initial guess (10, 10, 10). Ground truth here is chosen
    # far from that initial guess specifically to catch any regression back
    # to that failure mode.
    true_mu = {"IRN": 18.0, "GENERAL_OTHER": 5.0, "OTHER_SPECIALIZED": 25.0}
    panel = _synthetic_panel(true_mu, n_rows=200)

    fitted = fit_constants(panel)

    initial_guess = {"IRN": 10.0, "GENERAL_OTHER": 10.0, "OTHER_SPECIALIZED": 10.0}
    for category in CATEGORIES:
        assert abs(fitted[category] - initial_guess[category]) > 1.0, "optimizer failed to move off the initial guess"
        assert fitted[category] == pytest.approx(true_mu[category], abs=1.0)
