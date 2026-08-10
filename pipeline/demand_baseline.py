"""Two derived uses of the real dados.gov.pt attendance dataset
(data/cleaned_historical_baseline.parquet, built by pipeline/load_historical.py):

1. A historical demand-baseline feature: real average daily attendances by
   (branch, service, day_of_week), persisted to the `historical_demand_baseline`
   SQLite table and consumed by pipeline/feature_engineering.py as the
   `historical_avg_attendances` feature.
2. A grounded (not invented-from-scratch) wait-time PROXY label, derived from
   real attendance volume via a documented queueing approximation, inserted
   into `queue_samples` tagged source='historical_derived_proxy' — distinct
   from 'siga_live' (real measured wait times, once live scraping runs) and
   'synthetic_bootstrap' (fully synthetic demo data with no real grounding).

WAIT-TIME PROXY FORMULA (documented assumptions, not measured ground truth):
    utilization = (attendances * avg_service_minutes) / (desks * operating_hours * 60)
    wait_minutes = avg_service_minutes * utilization / (1 - utilization)   [M/M/1 Wq]
    capped at config.MAX_DERIVED_WAIT_MINUTES once utilization saturates.
Assumes config.DEFAULT_DESKS_PER_SERVICE parallel desks — an unverified
simplification, since the real dataset has no desk-count field.

CALIBRATED avg_service_minutes (2026-07-27): `avg_service_minutes` per row
now comes from data/calibrated_service_constants.json if present — 3
category-level constants (see pipeline/service_categories.py) fit by
pipeline/calibrate_constants.py against real IALC-M branch-day wait times,
replacing the SERVICE_AVG_MINUTES guesses for whichever services fall in
each category (holdout RMSE improved ~20% over the hand-guessed constants
at fit time — see the JSON file's own metrics for the current numbers).
Falls back to the old per-service SERVICE_AVG_MINUTES/DEFAULT_SERVICE_AVG_MINUTES
constants if the calibration hasn't been run yet, so this module still works
standalone on a fresh checkout.

Each day's total is expanded into config.DIURNAL_SNAPSHOTS representative
daypart snapshots rather than one flat noon timestamp, so the proxy rows
carry real hour_of_day variance. Each snapshot's wait estimate uses
config.SNAPSHOT_WINDOW_HOURS (~1 hour) as the M/M/1 capacity window, not the
full operating day, since a snapshot represents an hour-scale slice of
demand — reusing the full day's capacity there would make every snapshot
look far under capacity and erase the variance this expansion is for.

*** THE PROXY LABELS IN (2) ARE RETIRED AND NO LONGER GENERATED (2026-08-01). ***
Job (1), the real demand-baseline feature, is UNAFFECTED and still runs — it is
built from measured attendance counts and remains a genuine feature.

Only the wait-time labels are gated, on config.PROXY_LABEL_TRAINING_WEIGHT, the
same constant that controls their training weight — so generation and weighting
can never disagree and no orphaned rows are left behind. A controlled ablation
found the model got BETTER on real measurements without them (MAE 7.955 ->
7.804, R^2 0.484 -> 0.545 on a real-rows-only test set of 181,381), including
for the thinnest-coverage combos the tier was supposed to serve. They also
carry DIURNAL_SNAPSHOTS' hand-drawn hour curve, which live people_waiting data
contradicts at r = -0.79.

Generating ~6.9M rows nobody reads is not free: they dominated a 1.8GB SQLite
file and were rebuilt on every pipeline run. The stale-row DELETE still runs
unconditionally, so re-running this module clears them rather than leaving
orphaned state that training ignores but ad-hoc queries still pick up.

`derive_proxy_readings` below is deliberately KEPT and still unit-tested — code
is cheap to retain and is what makes this reversible; rows are what cost 1.8GB.
Set config.PROXY_LABEL_TRAINING_WEIGHT to 1.0 and re-run to restore exactly the
previous behaviour.

Usage:
    python -m pipeline.demand_baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
from pathlib import Path

import pandas as pd

from config import (
    BRANCHES,
    DEFAULT_DB_PATH,
    DEFAULT_DESKS_PER_SERVICE,
    DEFAULT_SERVICE_AVG_MINUTES,
    DIURNAL_SNAPSHOTS,
    MAX_DERIVED_WAIT_MINUTES,
    OPERATING_HOURS_PER_DAY,
    SERVICE_AVG_MINUTES,
    SNAPSHOT_WINDOW_HOURS,
    TARGET_DESK_UTILIZATION,
    PROXY_LABEL_TRAINING_WEIGHT,
)
from pipeline.db import delete_samples_by_source, get_connection, init_db, insert_queue_samples, upsert_branch
from pipeline.service_categories import categorize
from pipeline.geocode_branches import slugify
from schemas import QueueReading

logger = logging.getLogger(__name__)

CLEANED_BASELINE_PATH = "data/cleaned_historical_baseline.parquet"
CALIBRATED_CONSTANTS_PATH = "data/calibrated_service_constants.json"

DEMAND_BASELINE_SCHEMA = """
CREATE TABLE IF NOT EXISTS historical_demand_baseline (
    branch_id TEXT NOT NULL,
    desk_service_id TEXT NOT NULL,
    day_of_week INTEGER NOT NULL,
    avg_attendances REAL NOT NULL,
    PRIMARY KEY (branch_id, desk_service_id, day_of_week)
);
"""


def estimate_wait_minutes_from_attendances(
    total_attendances: float,
    avg_service_minutes: float,
    desks: int = DEFAULT_DESKS_PER_SERVICE,
    operating_hours: float = OPERATING_HOURS_PER_DAY,
    max_wait_minutes: float = MAX_DERIVED_WAIT_MINUTES,
) -> float:
    """M/M/1-style waiting-time approximation from an attendance count over
    the given `operating_hours` window. See module docstring for the
    assumptions this rests on. `operating_hours` must match the time window
    `total_attendances` actually represents — passing a full day's count with
    an hour-scale window (or vice versa) will badly over/under-state
    utilization and silently produce meaningless wait times.
    """
    if total_attendances <= 0:
        return 0.0
    capacity_minutes = operating_hours * 60 * desks
    demand_minutes = total_attendances * avg_service_minutes
    utilization = demand_minutes / capacity_minutes
    if utilization >= 0.98:
        return max_wait_minutes
    wait = avg_service_minutes * utilization / (1 - utilization)
    return round(min(wait, max_wait_minutes), 1)


def estimate_desks_for_volume(
    daily_attendance: float,
    avg_service_minutes: float,
    desks_baseline: int = DEFAULT_DESKS_PER_SERVICE,
    target_utilization: float = TARGET_DESK_UTILIZATION,
    operating_hours: float = OPERATING_HOURS_PER_DAY,
) -> int:
    """Scales the assumed desk count with real attendance volume and this
    service's own average ticket duration, instead of one flat constant for
    every service at every branch regardless of scale. Still a documented
    approximation (no real desk-count data exists for any branch), but a
    flat DEFAULT_DESKS_PER_SERVICE badly understates capacity for the
    handful of extreme-volume real combos (some branch/service pairs exceed
    900 attendances/day), producing implausible wait-time proxies for
    exactly those rows. desks_baseline is used as a floor, not a ceiling.
    """
    sustainable_capacity_per_desk = (operating_hours * 60 / avg_service_minutes) * target_utilization
    volume_implied_desks = math.ceil(daily_attendance / sustainable_capacity_per_desk)
    return max(desks_baseline, volume_implied_desks)


def load_calibrated_mu(path: str = CALIBRATED_CONSTANTS_PATH) -> dict[str, float] | None:
    """Loads category-level avg_service_minutes fit by
    pipeline/calibrate_constants.py against real IALC-M wait times. Returns
    None (graceful fallback to the hardcoded SERVICE_AVG_MINUTES constants)
    if calibration hasn't been run yet -- this module must keep working
    standalone on a fresh checkout."""
    if not Path(path).exists():
        return None
    with open(path) as f:
        payload = json.load(f)
    return payload["mu_by_category"]


def avg_service_minutes_for(service_type: str, calibrated_mu: dict[str, float] | None) -> float:
    """Per-service avg_service_minutes: calibrated category-level value if
    available (see pipeline/service_categories.py for why category, not
    per-service, is the granularity that's actually identifiable from real
    data), else the old hand-guessed per-service constant."""
    if calibrated_mu is not None:
        return calibrated_mu[categorize(service_type)]
    return SERVICE_AVG_MINUTES.get(service_type, DEFAULT_SERVICE_AVG_MINUTES)


def load_cleaned_baseline(path: str = CLEANED_BASELINE_PATH) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    # Same slugify() used by pipeline/geocode_branches.py, so store_name here
    # always lines up with config.BRANCHES_BY_ID keys.
    frame["branch_id"] = frame["store_name"].apply(slugify)
    return frame


def build_demand_baseline_table(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["day_of_week"] = pd.to_datetime(working["date"]).dt.dayofweek
    aggregated = (
        working.groupby(["branch_id", "service_type", "day_of_week"])["total_attendances"]
        .mean()
        .reset_index()
        .rename(columns={"service_type": "desk_service_id", "total_attendances": "avg_attendances"})
    )
    return aggregated


def save_demand_baseline(aggregated: pd.DataFrame, db_path: str = DEFAULT_DB_PATH) -> int:
    init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.executescript(DEMAND_BASELINE_SCHEMA)
        connection.execute("DELETE FROM historical_demand_baseline")
        aggregated.to_sql("historical_demand_baseline", connection, if_exists="append", index=False)
        connection.commit()
    return len(aggregated)


def derive_proxy_readings(frame: pd.DataFrame) -> list[QueueReading]:
    """Expands each daily attendance total into config.DIURNAL_SNAPSHOTS
    representative daypart snapshots, each with its own wait-time estimate
    computed over an hour-scale (config.SNAPSHOT_WINDOW_HOURS) capacity
    window — not the full day's capacity, which would understate every
    snapshot's utilization and erase the hour-of-day variance this expansion
    exists to provide. `people_waiting` is populated with the same derived
    per-slot estimate (not a real headcount) so the model has a non-constant
    people_waiting signal to learn from ahead of real siga_live data.
    """
    calibrated_mu = load_calibrated_mu()
    readings: list[QueueReading] = []
    for row in frame.itertuples(index=False):
        avg_service_minutes = avg_service_minutes_for(row.service_type, calibrated_mu)
        avg_hourly_attendance = row.total_attendances / OPERATING_HOURS_PER_DAY
        desks = estimate_desks_for_volume(row.total_attendances, avg_service_minutes)

        for hour, minute, volume_factor in DIURNAL_SNAPSHOTS:
            people_this_slot = avg_hourly_attendance * volume_factor
            wait_minutes = estimate_wait_minutes_from_attendances(
                people_this_slot, avg_service_minutes, desks=desks, operating_hours=SNAPSHOT_WINDOW_HOURS
            )
            sampled_at = pd.Timestamp(row.date).replace(hour=hour, minute=minute).tz_localize("UTC")
            readings.append(
                QueueReading(
                    branch_id=row.branch_id,
                    desk_service_id=row.service_type,
                    sampled_at=sampled_at.to_pydatetime(),
                    people_waiting=round(people_this_slot),
                    last_ticket_called=None,
                    estimated_wait_minutes=wait_minutes,
                    source="historical_derived_proxy",
                    # Grounded in real evidence, not the Mon-Fri/9-17 heuristic:
                    # dados.gov.pt recorded actual attendance this day, which
                    # is direct proof the branch was open — including the
                    # ~15.7k real Saturday rows (437k real attendances) this
                    # dataset contains, which a fixed Mon-Fri assumption would
                    # incorrectly mark closed.
                    is_open=True,
                )
            )
    return readings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive demand-baseline features and proxy wait-time labels from real attendance data"
    )
    parser.add_argument("--baseline", default=CLEANED_BASELINE_PATH)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    frame = load_cleaned_baseline(args.baseline)
    logger.info("Loaded %d cleaned historical rows for %d branches", len(frame), frame["branch_id"].nunique())

    demand_baseline = build_demand_baseline_table(frame)
    stored_baseline_rows = save_demand_baseline(demand_baseline, args.db)
    logger.info("Saved %d demand-baseline rows to historical_demand_baseline", stored_baseline_rows)

    # ----------------------------------------------------------------------
    # Proxy label generation is gated on the SAME constant that controls their
    # training weight, so the two can never disagree. With the labels retired
    # (the default since 2026-08-01), generating ~6.9M rows nobody reads is
    # pure waste: they dominate a 1.8GB SQLite file and are rebuilt on every
    # pipeline run.
    #
    # The generator itself (`derive_proxy_readings`) is deliberately KEPT and
    # still unit-tested. Code is cheap to retain and is what makes the
    # retirement reversible; rows are what cost 1.8GB. Setting
    # config.PROXY_LABEL_TRAINING_WEIGHT back to 1.0 and re-running this module
    # restores the previous behaviour exactly.
    #
    # The stale-row deletion runs either way. That is the point: it is what
    # stops a previous run's proxy rows lingering in queue_samples as orphaned
    # state that training silently ignores but every ad-hoc query still picks up.
    # ----------------------------------------------------------------------
    deleted = delete_samples_by_source(args.db, "historical_derived_proxy")
    if deleted:
        logger.info("Deleted %d historical_derived_proxy rows from a prior run", deleted)

    # Branch registration is unrelated bookkeeping and must not hang off a
    # retired code path — it used to sit inside the proxy-insert block purely
    # because that block happened to open the connection.
    with get_connection(args.db) as connection:
        for branch in BRANCHES:
            upsert_branch(connection, branch.branch_id, branch.name, branch.district, branch.latitude, branch.longitude)

    if PROXY_LABEL_TRAINING_WEIGHT <= 0:
        logger.warning(
            "SKIPPING proxy label generation — config.PROXY_LABEL_TRAINING_WEIGHT=%.3f. "
            "The real demand-baseline feature above was still written; only the "
            "formula-derived wait labels are retired. Set the constant to 1.0 to restore.",
            PROXY_LABEL_TRAINING_WEIGHT,
        )
        if deleted:
            logger.info("Run `VACUUM;` against %s to reclaim the freed space.", args.db)
        return

    readings = derive_proxy_readings(frame)
    with get_connection(args.db) as connection:
        stored_readings = insert_queue_samples(connection, readings)
    logger.info("Inserted %d historical_derived_proxy rows into queue_samples", stored_readings)


if __name__ == "__main__":
    main()
