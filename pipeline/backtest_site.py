"""Backtests the static site's own numbers as a FORECAST.

The page already makes a prediction, it just doesn't admit it: "the typical
Tuesday at Porto is 24 minutes" is a forecast of next Tuesday. Nothing has ever
checked whether that forecast is any good, or whether a slightly different
aggregate would be better. This module answers that from data already on disk.

METHOD: rolling-origin evaluation. For each cutoff, every predictor sees only
branch-days strictly BEFORE it, then predicts the following HOLDOUT_DAYS. No
predictor can see its own holdout, and all predictors share identical cutoffs
and identical holdout rows, so differences are attributable.

Several cutoffs rather than one, deliberately: a single split would let one
unusual month decide the winner. `pipeline/forward_test.py` documents the same
trap for the model (a two-hour window whose target mean swung 60.4 -> 46.9
between retrains, making the numbers incomparable).

WHY THE NAIVE BASELINES ARE NOT PADDING. `naive_persistence` (just repeat the
branch's last observed wait) and `branch_mean` (ignore the calendar entirely)
exist so that any sophistication has to *earn* its place. If the weekday split
the site currently uses cannot beat "ignore the calendar", the site should stop
showing a weekday breakdown -- that is a real possible outcome of this and the
reason to run it before building anything more elaborate.

WHAT THIS DELIBERATELY DOES NOT COVER:
  - Hour of day. Neither SLC-M nor IALC-M has ever had a time field (verified
    2026-08-01: both are daily). Hour is answerable only from siga_live, which
    spans days, not years -- no backtest over this corpus can speak to it.
  - Per-service. IALC-M is branch-level only.
  So this measures a DAY-LEVEL, BRANCH-LEVEL forecaster, which is exactly what
  the page currently is.

Run:
    python -m pipeline.backtest_site
    python -m pipeline.backtest_site --holdout-days 60 --cutoffs 8
"""

from __future__ import annotations

import argparse
import logging
from typing import Callable

import numpy as np
import pandas as pd

from pipeline.build_static import (
    MIN_ADJACENT_DAYS_FOR_UPLIFT,
    annotate_holidays,
    load_stable_branch_days,
    typical_days,
)

logger = logging.getLogger(__name__)

DEFAULT_HOLDOUT_DAYS = 30
DEFAULT_CUTOFFS = 6
# How many recent occurrences of a given weekday to average. Measured, not
# picked: MAE is U-shaped in this value (too few = noisy, too many = stale).
# 8 was chosen by validating across THREE window configurations, not by taking
# the argmin of the headline run — that first pass suggested 12 and a 0.85 min
# gain, which was partly this constant overfitting the six windows it was
# scored on. Across 10x30d / 6x60d / 4x90d, k=8 is best or within noise of best
# in each, and the honest gain settles at ~0.4 min:
#     k     10x30d   6x60d   4x90d
#     4      6.323   6.331   6.358
#     8      6.244   6.288   6.188   <- flat optimum
#     12     6.252   6.236   6.321
#     16     6.287   6.312   6.302
# 8 occurrences of one weekday is roughly two months of history.
RECENT_WINDOW_OCCURRENCES = 8

# A branch's holiday-adjacent multiplier is a ratio of two means; with thin
# history it can go wild. Clipped to a range that still admits the real
# national effect (~1.25) without letting one freak branch-day double a
# forecast.
UPLIFT_CLIP = (0.7, 2.0)


def _weekday_mean(history: pd.DataFrame) -> pd.Series:
    return history.groupby(["branch_id", "day_of_week"])["avg_wait_minutes"].mean()


def _lookup(index_frame: pd.DataFrame, table: pd.Series, keys: list[str]) -> np.ndarray:
    """Vectorized join onto a groupby result, NaN where the combo is unseen."""
    merged = index_frame.merge(
        table.rename("_v").reset_index(), on=keys, how="left", suffixes=("", "_y")
    )
    return merged["_v"].to_numpy()


# --------------------------------------------------------------------------
# Predictors. Each takes (history_before_cutoff, holdout_rows) -> predictions.
# --------------------------------------------------------------------------

def predict_naive_persistence(history: pd.DataFrame, holdout: pd.DataFrame) -> np.ndarray:
    last = history.sort_values("date").groupby("branch_id")["avg_wait_minutes"].last()
    return _lookup(holdout, last, ["branch_id"])


def predict_branch_mean(history: pd.DataFrame, holdout: pd.DataFrame) -> np.ndarray:
    return _lookup(holdout, history.groupby("branch_id")["avg_wait_minutes"].mean(), ["branch_id"])


def predict_weekday_mean_all_days(history: pd.DataFrame, holdout: pd.DataFrame) -> np.ndarray:
    """What the site did BEFORE 2026-08-01 -- holiday-adjacent days included."""
    return _lookup(holdout, _weekday_mean(history), ["branch_id", "day_of_week"])


def predict_weekday_mean_typical(history: pd.DataFrame, holdout: pd.DataFrame) -> np.ndarray:
    """What the site does TODAY -- holiday-adjacent days excluded from the mean."""
    return _lookup(holdout, _weekday_mean(typical_days(history)), ["branch_id", "day_of_week"])


def predict_weekday_plus_month(history: pd.DataFrame, holdout: pd.DataFrame) -> np.ndarray:
    """Weekday mean scaled by that branch's own month-of-year factor."""
    typical = typical_days(history)
    base = _lookup(holdout, _weekday_mean(typical), ["branch_id", "day_of_week"])

    branch_mean = typical.groupby("branch_id")["avg_wait_minutes"].mean()
    month_mean = typical.groupby(["branch_id", "month"])["avg_wait_minutes"].mean()
    factor = (month_mean / branch_mean).rename("f")
    scale = _lookup(holdout, factor, ["branch_id", "month"])
    return base * np.where(np.isnan(scale), 1.0, scale)


def predict_recent_weekday(history: pd.DataFrame, holdout: pd.DataFrame) -> np.ndarray:
    """Mean of only the last few occurrences of that weekday -- catches drift a
    three-year average cannot."""
    typical = typical_days(history).sort_values("date")
    recent = (
        typical.groupby(["branch_id", "day_of_week"])["avg_wait_minutes"]
        .apply(lambda s: s.tail(RECENT_WINDOW_OCCURRENCES).mean())
    )
    return _lookup(holdout, recent, ["branch_id", "day_of_week"])


def _holiday_uplift(history: pd.DataFrame) -> pd.Series:
    typical = typical_days(history)
    adjacent = history[history["is_bridge_day"] | history["is_post_holiday"]]

    adjacent_mean = adjacent.groupby("branch_id")["avg_wait_minutes"].mean()
    typical_mean = typical.groupby("branch_id")["avg_wait_minutes"].mean()
    counts = adjacent.groupby("branch_id")["avg_wait_minutes"].size()

    # Explicit alignment: early in the backtest a branch can appear in one
    # group and not the other (a new branch whose first weeks contain a ponte
    # but no ordinary days yet), and pandas will not divide those safely.
    branches = adjacent_mean.index.intersection(typical_mean.index)
    ratio = (adjacent_mean.loc[branches] / typical_mean.loc[branches]).replace([np.inf, -np.inf], np.nan)
    enough = counts.reindex(branches).fillna(0) >= MIN_ADJACENT_DAYS_FOR_UPLIFT
    return ratio[enough].dropna().clip(*UPLIFT_CLIP)


def predict_recent_weekday_plus_holiday(history: pd.DataFrame, holdout: pd.DataFrame) -> np.ndarray:
    """The site's current typical-day baseline, with the holiday-adjacent
    uplift applied back on days the holdout says ARE adjacent. Tests whether
    excluding those days is enough, or whether the page should also predict
    them."""
    base = predict_recent_weekday(history, holdout)
    uplift = _lookup(holdout, _holiday_uplift(history), ["branch_id"])
    uplift = np.where(np.isnan(uplift), 1.0, uplift)
    adjacent = (holdout["is_bridge_day"] | holdout["is_post_holiday"]).to_numpy()
    return base * np.where(adjacent, uplift, 1.0)


PREDICTORS: dict[str, Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]] = {
    "naive_persistence": predict_naive_persistence,
    "branch_mean": predict_branch_mean,
    "weekday_all_days (site pre-08-01)": predict_weekday_mean_all_days,
    "weekday_typical (SITE TODAY)": predict_weekday_mean_typical,
    "weekday_typical + month": predict_weekday_plus_month,
    "recent_weekday (last 8)": predict_recent_weekday,
    "recent_weekday + holiday": predict_recent_weekday_plus_holiday,
}


def run_backtest(frame: pd.DataFrame, holdout_days: int, cutoffs: int) -> pd.DataFrame:
    frame = frame.sort_values("date")
    frame["month"] = frame["date"].dt.month
    last_date = frame["date"].max()

    records: list[dict] = []
    for i in range(cutoffs):
        end = last_date - pd.Timedelta(days=i * holdout_days)
        start = end - pd.Timedelta(days=holdout_days)
        history = frame[frame["date"] < start]
        holdout = frame[(frame["date"] >= start) & (frame["date"] < end)]
        if len(holdout) < 100 or history.empty:
            continue

        for name, predictor in PREDICTORS.items():
            predicted = predictor(history, holdout)
            actual = holdout["avg_wait_minutes"].to_numpy()
            # A predictor that cannot score a row (unseen branch) is not
            # rewarded for skipping it -- coverage is reported alongside.
            usable = ~np.isnan(predicted)
            error = np.abs(predicted[usable] - actual[usable])
            adjacent = (holdout["is_bridge_day"] | holdout["is_post_holiday"]).to_numpy()[usable]
            records.append(
                {
                    "cutoff": str(start.date()),
                    "predictor": name,
                    "n": int(usable.sum()),
                    "coverage": round(float(usable.mean()), 4),
                    "mae": float(error.mean()),
                    "rmse": float(np.sqrt((error ** 2).mean())),
                    "mae_typical": float(error[~adjacent].mean()) if (~adjacent).any() else np.nan,
                    "mae_adjacent": float(error[adjacent].mean()) if adjacent.any() else np.nan,
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-days", type=int, default=DEFAULT_HOLDOUT_DAYS)
    parser.add_argument("--cutoffs", type=int, default=DEFAULT_CUTOFFS)
    args = parser.parse_args()

    frame = annotate_holidays(load_stable_branch_days())
    logger.info("Corpus: %d stable branch-days, %s to %s",
                len(frame), frame["date"].min().date(), frame["date"].max().date())

    results = run_backtest(frame, args.holdout_days, args.cutoffs)
    if results.empty:
        raise SystemExit("No usable holdout windows.")

    summary = (
        results.groupby("predictor")
        .agg(windows=("mae", "size"), mean_n=("n", "mean"), coverage=("coverage", "mean"),
             mae=("mae", "mean"), rmse=("rmse", "mean"),
             mae_typical=("mae_typical", "mean"), mae_adjacent=("mae_adjacent", "mean"))
        .sort_values("mae")
    )
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"\n=== Rolling-origin backtest: {args.cutoffs} windows x {args.holdout_days} days ===")
    print(summary.round(3).to_string())

    best = summary.index[0]
    incumbent = "weekday_typical (SITE TODAY)"
    if incumbent in summary.index:
        delta = summary.loc[incumbent, "mae"] - summary.loc[best, "mae"]
        print(f"\nBest: {best!r} (MAE {summary.loc[best, 'mae']:.3f})")
        print(f"Site today: {incumbent!r} (MAE {summary.loc[incumbent, 'mae']:.3f}) -> {delta:+.3f} min available")

    print("\nPer-window MAE (a winner should win consistently, not on average):")
    print(results.pivot(index="cutoff", columns="predictor", values="mae").round(2).to_string())


if __name__ == "__main__":
    main()
