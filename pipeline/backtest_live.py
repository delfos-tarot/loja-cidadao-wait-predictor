"""Decides which dimensions the live-only view has earned the right to show.

The live SIGA corpus is the ONLY source with per-service and per-hour
resolution — neither SLC-M nor IALC-M has ever had either (both are daily,
branch-level). It is also young, so a dimension that will eventually carry
signal may currently carry only noise. This module measures that, per
dimension, so the view turns each one on when the data says so rather than
when someone feels ready.

TWO TARGETS, NOT ONE — because the distribution is bimodal, not spread.
Measured 2026-08-05 over 74,466 open-desk readings: 67% report exactly zero
wait (and 90% report zero people waiting, so these are genuinely idle desks,
not missing data), while readings that DO have a queue run to a median of 64
minutes. A citizen either walks straight up or waits about an hour. A single
mean lands at ~51 min and describes almost nobody, which is exactly the flaw
the static site inherits from IALC-M averages.

So the two things worth predicting are:
  1. WALK-IN PROBABILITY — will there be any queue at all? Scored with Brier
     score (a proper scoring rule for probabilities; accuracy would reward
     always guessing the majority class).
  2. BUSY WAIT — given a queue, how long? Scored with MAE over the queued
     subset only, so the 67% of zeros cannot flatter it.

WHY NOT JUST TRUST MORE DATA. Measured on a 6-day train / 2-day test split:
adding weekday made prediction WORSE (MAE 67.87 vs 48.07 without it), because
six days give each weekday roughly one observation and the cells fit noise.
Adding hour gained 1.5%. Both will improve with accumulation, but "will
improve" is not "has improved" — hence this gate.

Run:
    python -m pipeline.backtest_live
    python -m pipeline.backtest_live --folds 4 --test-days 1
"""

from __future__ import annotations

import argparse
import logging
import sqlite3

import numpy as np
import pandas as pd

from config import DEFAULT_DB_PATH, REAL_DATA_MAX_PLAUSIBLE_WAIT_MINUTES

logger = logging.getLogger(__name__)

DEFAULT_TEST_DAYS = 1
DEFAULT_FOLDS = 3

# A dimension must beat the simpler model it extends by more than this to be
# enabled. Not zero: a hair's improvement on a few days of data is noise, and
# switching a dimension on has a cost (thinner cells, a more confident-looking
# page). Expressed as a fraction of the simpler model's score.
MIN_RELATIVE_GAIN = 0.02

# Candidate dimension sets, ordered simplest first. Each is tested against its
# immediate predecessor, so "hour" must beat "combo", not merely beat "global".
DIMENSION_SETS: tuple[tuple[str, list[str]], ...] = (
    ("global", []),
    ("combo", ["branch_id", "desk_service_id"]),
    ("combo+hour", ["branch_id", "desk_service_id", "hour"]),
    ("combo+dow", ["branch_id", "desk_service_id", "dow"]),
    ("combo+dow+hour", ["branch_id", "desk_service_id", "dow", "hour"]),
)


def load_live(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Open-desk readings with a usable wait value, in Lisbon local time.

    `is_open == 1` matters: a closed desk reports zero wait, and counting those
    as walk-ins would make every branch look wonderful overnight.
    """
    with sqlite3.connect(db_path) as connection:
        frame = pd.read_sql(
            "SELECT branch_id, desk_service_id, sampled_at, people_waiting, wait_time_minutes, is_open "
            "FROM queue_samples WHERE source='siga_live' AND wait_time_minutes IS NOT NULL",
            connection,
        )
    frame = frame[(frame["is_open"] == 1) & (frame["wait_time_minutes"] <= REAL_DATA_MAX_PLAUSIBLE_WAIT_MINUTES)]
    local = pd.to_datetime(frame["sampled_at"], utc=True).dt.tz_convert("Europe/Lisbon")
    people = pd.to_numeric(frame["people_waiting"], errors="coerce").fillna(0)
    wait = frame["wait_time_minutes"]
    # A queue requires BOTH signals to agree. Found 2026-08-05: they contradict
    # each other on 29.5% of open-desk readings — 26.0% report a wait with
    # nobody waiting (the documented stale-counter pathology), 3.5% the reverse.
    # Defining a queue as `wait > 0` alone swallows that 26% wholesale and was
    # the first version of this module's mistake.
    frame = frame.assign(
        local=local, hour=local.dt.hour, dow=local.dt.dayofweek, date=local.dt.date,
        people=people,
        has_queue=((wait > 0) & (people > 0)).astype(int),
        coherent=(((wait > 0) & (people > 0)) | ((wait == 0) & (people == 0))).astype(int),
    )
    return frame.sort_values("local").reset_index(drop=True)


def _grouped(train: pd.DataFrame, test: pd.DataFrame, keys: list[str], column: str) -> np.ndarray:
    """Group mean of `column` looked up for each test row; NaN where unseen."""
    if not keys:
        return np.full(len(test), train[column].mean())
    table = train.groupby(keys)[column].mean().rename("_v").reset_index()
    return test.merge(table, on=keys, how="left")["_v"].to_numpy()


def evaluate_fold(train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    """Scores every dimension set on one fold, for both targets."""
    fallback_p = float(train["has_queue"].mean())
    fallback_w = float(train.loc[train["has_queue"] == 1, "people"].mean())
    busy = test["has_queue"] == 1

    records = []
    for name, keys in DIMENSION_SETS:
        # Target 1: probability of any queue at all.
        probability = _grouped(train, test, keys, "has_queue")
        probability = np.where(np.isnan(probability), fallback_p, probability)
        brier = float(np.mean((probability - test["has_queue"].to_numpy()) ** 2))

        # Target 2: HOW MANY PEOPLE, given a queue — deliberately not minutes.
        # `tempoRealEspera` is unusable as a duration: in the subset where both
        # signals agree a queue exists, it reports a median 162 min against a
        # median ONE person waiting, i.e. 162 minutes to serve one citizen,
        # against IALC-M's measured 7.2 min service time. Off by ~22x, so no
        # amount of filtering rescues it. `people_waiting` is a count, it is
        # coherent, and median 1 / mean 2.3 is plausible.
        queued_train = train[train["has_queue"] == 1]
        length = _grouped(queued_train, test, keys, "people") if len(queued_train) else np.full(len(test), np.nan)
        length = np.where(np.isnan(length), fallback_w, length)
        busy_mae = float(np.abs(length[busy.to_numpy()] - test.loc[busy, "people"]).mean()) if busy.any() else np.nan

        coverage = float(
            1.0 - np.isnan(_grouped(train, test, keys, "has_queue")).mean()
        )
        records.append({"dimensions": name, "brier": brier, "busy_mae": busy_mae, "coverage": coverage})
    return records


def run(frame: pd.DataFrame, folds: int, test_days: int) -> pd.DataFrame:
    days = sorted(frame["date"].unique())
    if len(days) < folds + test_days + 1:
        logger.warning("Only %d days of live data — reducing folds", len(days))
        folds = max(1, len(days) - test_days - 1)

    records = []
    for fold in range(folds):
        end = len(days) - fold * test_days
        test_start = end - test_days
        if test_start <= 0:
            break
        train = frame[frame["date"] < days[test_start]]
        test = frame[frame["date"].isin(days[test_start:end])]
        if train.empty or test.empty:
            continue
        for record in evaluate_fold(train, test):
            records.append({**record, "fold": str(days[test_start]), "n_train": len(train), "n_test": len(test)})
    return pd.DataFrame(records)


def decide(summary: pd.DataFrame) -> list[str]:
    """Which dimension set the view should use.

    Walks the candidates simplest-first and keeps extending only while the
    extension earns MIN_RELATIVE_GAIN on BOTH targets. Requiring both stops a
    dimension that sharpens the walk-in guess while blurring the busy estimate
    (or vice versa) from switching itself on.
    """
    ordered = [name for name, _ in DIMENSION_SETS if name in summary.index]
    chosen = ordered[0]
    for candidate in ordered[1:]:
        current, proposed = summary.loc[chosen], summary.loc[candidate]
        brier_gain = (current["brier"] - proposed["brier"]) / current["brier"]
        mae_gain = (current["busy_mae"] - proposed["busy_mae"]) / current["busy_mae"]
        if brier_gain > MIN_RELATIVE_GAIN and mae_gain > MIN_RELATIVE_GAIN:
            chosen = candidate
        else:
            logger.info(
                "  %-16s does not earn its place over %s (brier %+.1f%%, busy MAE %+.1f%%)",
                candidate, chosen, 100 * brier_gain, 100 * mae_gain,
            )
    return dict(DIMENSION_SETS)[chosen], chosen


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS)
    args = parser.parse_args()

    frame = load_live(args.db)
    logger.info(
        "Live corpus: %d open-desk readings, %d days, %d combos | %.0f%% have no queue",
        len(frame), frame["date"].nunique(),
        frame.groupby(["branch_id", "desk_service_id"]).ngroups,
        100 * (1 - frame["has_queue"].mean()),
    )

    results = run(frame, args.folds, args.test_days)
    if results.empty:
        raise SystemExit("Not enough live data for a fold.")

    summary = results.groupby("dimensions").agg(
        folds=("brier", "size"), brier=("brier", "mean"),
        busy_mae=("busy_mae", "mean"), coverage=("coverage", "mean"))
    summary = summary.reindex([n for n, _ in DIMENSION_SETS if n in summary.index])

    pd.set_option("display.width", 200)
    print("\n=== Live-only dimension gate ===")
    print("  brier    = walk-in probability error (lower better)")
    print("  busy_mae = PEOPLE-in-queue error given a queue (lower better)")
    print("  coverage = share of test rows the dimension set could score at all\n")
    print(summary.round(4).to_string())

    keys, name = decide(summary)
    print(f"\nENABLED: {name}  -> group by {keys or ['(global)']}")
    print("Re-run as live data accumulates; hour and dow are expected to earn their place later.")


if __name__ == "__main__":
    main()
