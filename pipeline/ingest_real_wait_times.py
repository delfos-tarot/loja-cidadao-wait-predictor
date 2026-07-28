"""Turns real IALC-M branch-day wait/duration averages
(data/cleaned_ialc_baseline.parquet, built by pipeline/load_ialc.py) into
`queue_samples` rows tagged source='historical_real_daily_avg' — a real
(not formula-derived) but coarser-grained label than siga_live.

GRANULARITY, HONESTLY: IALC-M has no per-service breakdown (see
pipeline/load_ialc.py's module docstring) — only a single real average per
(branch, day). Rather than invent a per-service split with no real basis
(exactly the identifiability trap pipeline/calibrate_constants.py was built
to avoid), the SAME real branch-day value is broadcast to every
desk_service_id that SLC-M shows as actually active at that branch on that
exact day (data/cleaned_historical_baseline.parquet) — not the full global
config.DESK_SERVICES list, which would fabricate rows for services a given
branch doesn't even offer.

ONE TIMESTAMP PER DAY, NOT ONE PER config.DIURNAL_SNAPSHOTS ENTRY:
demand_baseline.py expands its *formula-derived* proxy into multiple
daypart snapshots because each snapshot gets a genuinely different value
(via a volume_factor). Here there is only one real number for the whole
day — expanding a single day into several identical copies would multiply
this source's weight in training while teaching the model a flat (wrong)
hour_of_day relationship specifically for this source, since sample
weight, not the "source" column, is what the model actually sees.

BUT PINNING EVERY DAY TO THE SAME HOUR IS ALSO WRONG: an earlier version of
this module placed every row at the same fixed hour (12:30) — which meant
this source contributed real training signal at exactly one clock time and
nowhere else, the same asymmetry that caused off-anchor predictions
(10-11am) to come out scrambled relative to the properly-supported hour.
Each (branch, date) is instead deterministically rotated across
config.DIURNAL_SNAPSHOTS' candidate hours via date.toordinal() % N (N not a
divisor of 7, so the phase drifts relative to day-of-week instead of
permanently pairing one weekday with one hour) — every individual row is
still exactly one honest real observation at one plausible hour, but
summed across ~700+ real days, every candidate hour ends up with genuine
(not formula-shaped) grounding instead of just one.

SAMPLE-SIZE WEIGHTING: each row's `sample_size` is the real
Total_Atendimentos backing that branch-day average (found during the
SLC-M/IALC-M cross-check to be as low as 1 on some branch-days) —
pipeline/train.py's compute_sample_weights uses this to down-weight
low-confidence branch-days rather than trusting every row equally.

Usage:
    python -m pipeline.ingest_real_wait_times
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from config import BRANCHES, DEFAULT_DB_PATH, DIURNAL_SNAPSHOTS
from pipeline.db import delete_samples_by_source, get_connection, insert_queue_samples, upsert_branch
from pipeline.geocode_branches import slugify
from schemas import QueueReading

logger = logging.getLogger(__name__)

IALC_BASELINE_PATH = "data/cleaned_ialc_baseline.parquet"
SLC_BASELINE_PATH = "data/cleaned_historical_baseline.parquet"

# Candidate representative hours, reusing config.DIURNAL_SNAPSHOTS' clock
# times (not its volume_factor, which is only meaningful for the formula-
# derived proxy) — see module docstring for why each day rotates across
# these instead of every day sharing one fixed hour.
_CANDIDATE_HOURS: tuple[tuple[int, int], ...] = tuple((hour, minute) for hour, minute, _ in DIURNAL_SNAPSHOTS)


def _representative_hour_for_date(date: pd.Timestamp) -> tuple[int, int]:
    return _CANDIDATE_HOURS[date.toordinal() % len(_CANDIDATE_HOURS)]


def build_real_daily_avg_readings(ialc_frame: pd.DataFrame, slc_frame: pd.DataFrame) -> list[QueueReading]:
    slc_frame = slc_frame.copy()
    # cleaned_historical_baseline.parquet only carries store_name -- branch_id
    # is derived downstream by every consumer (demand_baseline.py,
    # calibrate_constants.py) via the same slugify(), so this must match.
    slc_frame["branch_id"] = slc_frame["store_name"].apply(slugify)
    services_by_branch_day = slc_frame[["branch_id", "date", "service_type"]].drop_duplicates()
    services_by_branch_day["date"] = pd.to_datetime(services_by_branch_day["date"])

    ialc = ialc_frame.copy()
    ialc["date"] = pd.to_datetime(ialc["date"])

    merged = services_by_branch_day.merge(ialc, on=["branch_id", "date"], how="inner")

    readings: list[QueueReading] = []
    for row in merged.itertuples(index=False):
        date = pd.Timestamp(row.date)
        hour, minute = _representative_hour_for_date(date)
        sampled_at = date.replace(hour=hour, minute=minute).tz_localize("UTC")
        readings.append(
            QueueReading(
                branch_id=row.branch_id,
                desk_service_id=row.service_type,
                sampled_at=sampled_at.to_pydatetime(),
                people_waiting=None,
                last_ticket_called=None,
                estimated_wait_minutes=float(row.avg_wait_minutes),
                source="historical_real_daily_avg",
                # Real recorded attendance that day is direct evidence the
                # branch was open, same justification demand_baseline.py uses.
                is_open=True,
                sample_size=int(row.total_attendances),
            )
        )
    return readings


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest real IALC-M branch-day wait times into queue_samples")
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--ialc-baseline", default=IALC_BASELINE_PATH)
    parser.add_argument("--slc-baseline", default=SLC_BASELINE_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ialc_frame = pd.read_parquet(args.ialc_baseline)
    slc_frame = pd.read_parquet(args.slc_baseline)
    logger.info("Loaded %d IALC-M branch-days and %d SLC-M (branch, day, service) rows", len(ialc_frame), len(slc_frame))

    readings = build_real_daily_avg_readings(ialc_frame, slc_frame)
    if not readings:
        raise SystemExit("No overlapping (branch, day) rows between IALC-M and SLC-M -- check both were ingested for the same date range")
    logger.info("Derived %d historical_real_daily_avg readings", len(readings))

    deleted = delete_samples_by_source(args.db, "historical_real_daily_avg")
    if deleted:
        logger.info("Deleted %d stale historical_real_daily_avg rows from a prior run before re-inserting", deleted)

    with get_connection(args.db) as connection:
        for branch in BRANCHES:
            upsert_branch(connection, branch.branch_id, branch.name, branch.district, branch.latitude, branch.longitude)
        stored = insert_queue_samples(connection, readings)
    logger.info("Inserted %d historical_real_daily_avg rows into queue_samples", stored)


if __name__ == "__main__":
    main()
