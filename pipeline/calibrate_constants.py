"""Fits `avg_service_minutes` per service category (see
pipeline/service_categories.py), against real branch-day wait times from
IALC-M (pipeline/load_ialc.py) — replacing the current hand-guessed
DEFAULT_SERVICE_AVG_MINUTES/SERVICE_AVG_MINUTES constants in config.py with
values fit against ~34k real (branch, day) observations instead of
assumption. Desk sizing (TARGET_DESK_UTILIZATION) is intentionally NOT
re-fit here — see below for why.

WHY ONLY 3 FREE PARAMETERS (mu_IRN, mu_GENERAL_OTHER, mu_OTHER_SPECIALIZED),
NOT one per exact service label:
An identifiability check (2026-07-27, see pipeline/service_categories.py's
docstring) found that finer splits aren't separable from real service-mix
variation across branches. This script does NOT attempt anything finer than
that check supports — expanding categories here would just fit noise while
looking precise, the same trap the original blended R^2 evaluation fell
into.

WHY BOUNDED NONLINEAR LEAST SQUARES, NOT AN UNCONSTRAINED FIT:
The M/M/1 formula's wait = mu * rho/(1-rho) term has an asymptote as
rho -> 1. An unconstrained optimizer can exploit that asymptote to match
almost any target by pushing utilization toward 1 rather than genuinely
finding a better mu — so mu is bounded to a plausible range (see
PARAM_BOUNDS below).

WHY DESKS ARE PRECOMPUTED ONCE, NOT RE-DERIVED FROM THE TRIAL mu:
The obvious design — call demand_baseline.estimate_desks_for_volume() with
the trial mu at every optimizer iteration, the same way production code
does — was tried first and failed silently: `least_squares` converged after
a single near-zero step, reporting exact-zero first-order optimality.
Diagnosis (verified with scipy.optimize._numdiff.approx_derivative at
several step sizes): estimate_desks_for_volume's `math.ceil()` makes the
objective piecewise-constant at the scale of the optimizer's default
finite-difference probe, so the numerically-estimated Jacobian is dominated
by discretization noise rather than the true (real, exploitable — confirmed
by manually scanning mu over a wide range and watching RMSE actually move)
underlying slope. There's also a subtler reason beyond the numerical one:
letting desks scale with the *same* mu being fit is nearly self-cancelling
by construction — estimate_desks_for_volume sizes desks specifically to
hold utilization near a target level regardless of volume, which erases
most of the volume-driven variation calibration is trying to explain.
Fix: desks are computed once per (branch, day, category) using the
existing *baseline* (uncalibrated, current config.py) mu and
TARGET_DESK_UTILIZATION — i.e. "assume today's desk-provisioning heuristic
is applied as-is" — held fixed, and only mu is optimized against the
resulting smooth (division/multiplication only, no ceiling) wait formula.
This also means target_utilization is no longer a separately fit
parameter; it's baked into the one-time desk precomputation.

WHY A CHRONOLOGICAL HOLDOUT:
Fitting against the same corpus used to evaluate "how good is this fit" is
exactly the mistake this project already caught once (the original blended
R^2 = 0.9985). The fit is trained on the earlier ~80% of the real IALC-M
history and validated on the most recent ~20%, mirroring
pipeline/train.py's chronological_split.

WHY WEIGHTED BY Total_Atendimentos:
Found during the SLC-M/IALC-M cross-check: many branch-days have very low
attendance counts (some exactly 1), making that day's "average" wait a
single noisy raw observation rather than a stable mean. Unweighted fitting
would let those noisy low-volume days pull the fit as hard as a
high-volume day like Laranjeiras averaging over thousands of people.

Usage:
    python -m pipeline.calibrate_constants
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from config import (
    DEFAULT_DESKS_PER_SERVICE,
    DEFAULT_SERVICE_AVG_MINUTES,
    MAX_DERIVED_WAIT_MINUTES,
    OPERATING_HOURS_PER_DAY,
    SERVICE_AVG_MINUTES,
    TARGET_DESK_UTILIZATION,
)
from pipeline.demand_baseline import estimate_desks_for_volume
from pipeline.service_categories import CATEGORIES, categorize

logger = logging.getLogger(__name__)

SLC_BASELINE_PATH = "data/cleaned_historical_baseline.parquet"
IALC_BASELINE_PATH = "data/cleaned_ialc_baseline.parquet"
OUTPUT_PATH = "data/calibrated_service_constants.json"

# mu (avg service minutes) bounds: a real desk interaction plausibly takes
# somewhere between 2 and 30 minutes.
PARAM_BOUNDS_LOWER = [2.0, 2.0, 2.0]
PARAM_BOUNDS_UPPER = [30.0, 30.0, 30.0]
INITIAL_GUESS = [DEFAULT_SERVICE_AVG_MINUTES, DEFAULT_SERVICE_AVG_MINUTES, DEFAULT_SERVICE_AVG_MINUTES]

HOLDOUT_FRACTION = 0.2


def _desks_column(category: str) -> str:
    return f"{category}__desks"


def build_category_day_panel(baseline_mu: dict[str, float]) -> pd.DataFrame:
    """Merges categorized SLC-M attendance volume (per branch/day/category)
    with real IALC-M wait times (per branch/day), sorted chronologically, and
    precomputes a FIXED desks-per-category column (see module docstring for
    why this must be fixed rather than re-derived during optimization)."""
    slc = pd.read_parquet(SLC_BASELINE_PATH)
    slc["date"] = pd.to_datetime(slc["date"])
    slc["category"] = slc["service_type"].apply(categorize)

    from pipeline.geocode_branches import slugify

    slc["branch_id"] = slc["store_name"].apply(slugify)
    category_volume = (
        slc.groupby(["branch_id", "date", "category"])["total_attendances"].sum().unstack(fill_value=0).reset_index()
    )
    for category in CATEGORIES:
        if category not in category_volume.columns:
            category_volume[category] = 0

    ialc = pd.read_parquet(IALC_BASELINE_PATH)
    ialc["date"] = pd.to_datetime(ialc["date"])

    merged = category_volume.merge(
        ialc[["branch_id", "date", "avg_wait_minutes", "total_attendances"]], on=["branch_id", "date"], how="inner"
    )
    merged = merged[merged["total_attendances"] > 0].sort_values("date").reset_index(drop=True)

    for category in CATEGORIES:
        volume = merged[category].to_numpy()
        merged[_desks_column(category)] = [
            estimate_desks_for_volume(v, baseline_mu[category], target_utilization=TARGET_DESK_UTILIZATION, operating_hours=OPERATING_HOURS_PER_DAY)
            if v > 0
            else DEFAULT_DESKS_PER_SERVICE
            for v in volume
        ]
    return merged


def chronological_split(panel: pd.DataFrame, holdout_fraction: float = HOLDOUT_FRACTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(panel) * (1 - holdout_fraction))
    return panel.iloc[:split_index], panel.iloc[split_index:]


def _wait_given_fixed_desks(volume: np.ndarray, mu: float, desks: np.ndarray) -> np.ndarray:
    """Vectorized, smooth (no ceiling) version of
    demand_baseline.estimate_wait_minutes_from_attendances, for a FIXED
    desks array — mu is the only thing allowed to vary here."""
    capacity_minutes = OPERATING_HOURS_PER_DAY * 60 * desks
    demand_minutes = volume * mu
    utilization = np.divide(demand_minutes, capacity_minutes, out=np.zeros_like(demand_minutes, dtype=float), where=capacity_minutes > 0)
    utilization = np.clip(utilization, 0.0, 0.98)
    wait = np.minimum(mu * utilization / (1 - utilization), MAX_DERIVED_WAIT_MINUTES)
    return np.where(volume > 0, wait, 0.0)


def predict_branch_day_wait(panel: pd.DataFrame, mu_by_category: dict[str, float]) -> np.ndarray:
    """Volume-weighted average predicted wait across categories present that
    day, using fixed (precomputed) desks and the given trial mu per
    category."""
    predictions = np.zeros(len(panel))
    for category, mu in mu_by_category.items():
        volume = panel[category].to_numpy()
        desks = panel[_desks_column(category)].to_numpy()
        predictions += volume * _wait_given_fixed_desks(volume, mu, desks)
    return predictions / panel["total_attendances"].to_numpy()


def _residuals(params: np.ndarray, panel: pd.DataFrame) -> np.ndarray:
    mu_by_category = dict(zip(CATEGORIES, params))
    predicted = predict_branch_day_wait(panel, mu_by_category)
    real = panel["avg_wait_minutes"].to_numpy()
    weight = np.sqrt(panel["total_attendances"].to_numpy())
    return weight * (predicted - real)


def weighted_rmse(panel: pd.DataFrame, mu_by_category: dict[str, float]) -> float:
    predicted = predict_branch_day_wait(panel, mu_by_category)
    real = panel["avg_wait_minutes"].to_numpy()
    weight = panel["total_attendances"].to_numpy()
    return float(np.sqrt(np.average((predicted - real) ** 2, weights=weight)))


def baseline_mu_by_category() -> dict[str, float]:
    """The current hardcoded constants, mapped to categories via a
    volume-weighted average of whichever specific services fall in each
    category (falling back to DEFAULT_SERVICE_AVG_MINUTES) -- i.e. "what
    would today's un-calibrated formula have predicted", as the point of
    comparison for whether calibration actually helps."""
    slc = pd.read_parquet(SLC_BASELINE_PATH)
    slc["category"] = slc["service_type"].apply(categorize)
    slc["mu"] = slc["service_type"].map(SERVICE_AVG_MINUTES).fillna(DEFAULT_SERVICE_AVG_MINUTES)

    result = {}
    for category in CATEGORIES:
        subset = slc[slc["category"] == category]
        if subset["total_attendances"].sum() == 0:
            result[category] = DEFAULT_SERVICE_AVG_MINUTES
        else:
            result[category] = float(np.average(subset["mu"], weights=subset["total_attendances"]))
    return result


def fit_constants(panel: pd.DataFrame) -> dict[str, float]:
    result = least_squares(
        _residuals,
        x0=INITIAL_GUESS,
        bounds=(PARAM_BOUNDS_LOWER, PARAM_BOUNDS_UPPER),
        args=(panel,),
        method="trf",
    )
    logger.info("Optimizer: nfev=%d status=%d message=%s", result.nfev, result.status, result.message)
    return dict(zip(CATEGORIES, (float(v) for v in result.x)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate M/M/1 proxy constants against real IALC-M wait times")
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    baseline_mu = baseline_mu_by_category()
    panel = build_category_day_panel(baseline_mu)
    if panel.empty:
        raise SystemExit("No overlapping (branch, day) rows between SLC-M and IALC-M -- run pipeline/load_ialc.py first")
    logger.info("Built calibration panel: %d (branch, day) rows spanning %s to %s", len(panel), panel["date"].min(), panel["date"].max())

    train_panel, holdout_panel = chronological_split(panel)
    logger.info("Chronological split: %d train rows, %d holdout rows", len(train_panel), len(holdout_panel))

    mu_by_category = fit_constants(train_panel)

    calibrated_train_rmse = weighted_rmse(train_panel, mu_by_category)
    calibrated_holdout_rmse = weighted_rmse(holdout_panel, mu_by_category)
    baseline_train_rmse = weighted_rmse(train_panel, baseline_mu)
    baseline_holdout_rmse = weighted_rmse(holdout_panel, baseline_mu)

    logger.info("Calibrated mu by category: %s", {k: round(v, 2) for k, v in mu_by_category.items()})
    logger.info("Baseline (current hardcoded) mu by category: %s", {k: round(v, 2) for k, v in baseline_mu.items()})
    logger.info(
        "Weighted RMSE (minutes) -- train: baseline=%.2f calibrated=%.2f | holdout: baseline=%.2f calibrated=%.2f",
        baseline_train_rmse,
        calibrated_train_rmse,
        baseline_holdout_rmse,
        calibrated_holdout_rmse,
    )

    output = {
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "corpus_rows": len(panel),
        "corpus_date_range": [str(panel["date"].min().date()), str(panel["date"].max().date())],
        "mu_by_category": mu_by_category,
        "target_utilization": TARGET_DESK_UTILIZATION,
        "holdout_fraction": HOLDOUT_FRACTION,
        "weighted_rmse_minutes": {
            "baseline_train": baseline_train_rmse,
            "calibrated_train": calibrated_train_rmse,
            "baseline_holdout": baseline_holdout_rmse,
            "calibrated_holdout": calibrated_holdout_rmse,
        },
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved calibrated constants to %s", args.output)


if __name__ == "__main__":
    main()
