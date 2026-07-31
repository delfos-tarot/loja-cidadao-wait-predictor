"""Unit tests for pipeline/forward_test.py's scoring and safety rails."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.forward_test import MIN_TRUSTWORTHY_WINDOW_HOURS, run_forward_test, score


def test_score_reports_distribution_stats_not_just_error() -> None:
    # Systematic under-prediction is invisible in MAE alone -- the actual vs
    # predicted means are what exposed it on 2026-07-31.
    actual = pd.Series([10.0, 50.0, 90.0])
    predicted = np.array([5.0, 25.0, 45.0])

    result = score(actual, predicted)

    assert result["n"] == 3
    assert result["actual_mean"] == pytest.approx(50.0)
    assert result["predicted_mean"] == pytest.approx(25.0)
    assert result["predicted_mean"] < result["actual_mean"]


def test_score_handles_an_empty_slice_without_crashing() -> None:
    result = score(pd.Series(dtype=float), np.array([]))

    assert result["n"] == 0
    assert result["mae"] is None
    assert result["r2"] is None


def test_score_returns_none_r2_for_a_constant_target() -> None:
    # R^2 is undefined when the target has no variance; must not raise or
    # report a misleading number.
    result = score(pd.Series([7.0, 7.0, 7.0]), np.array([5.0, 6.0, 7.0]))

    assert result["n"] == 3
    assert result["mae"] is not None
    assert result["r2"] is None


def test_refuses_to_run_against_an_artifact_without_a_data_cutoff(tmp_path) -> None:
    # Guards the core correctness property: falling back to trained_at (wall
    # clock) would silently score the model on its own training rows, since
    # scraping continues while training runs.
    import joblib

    model_path = tmp_path / "old_artifact.joblib"
    joblib.dump({"model": object(), "feature_columns": [], "trained_at": "2026-07-31T00:00:00+00:00"}, model_path)

    with pytest.raises(SystemExit, match="data_cutoff"):
        run_forward_test(db_path=str(tmp_path / "empty.db"), model_path=str(model_path))


def test_min_trustworthy_window_is_at_least_a_full_day() -> None:
    # A partial day is dominated by which hours it covers -- per config.py's
    # DIURNAL_SNAPSHOTS finding, busyness varies ~2.6x across the day.
    assert MIN_TRUSTWORTHY_WINDOW_HOURS >= 24.0
