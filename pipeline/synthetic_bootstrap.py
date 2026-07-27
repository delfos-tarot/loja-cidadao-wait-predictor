"""Synthetic queue-history generator for cold-start development/testing.

Used when neither live SIGA polling nor the real dados.gov.pt attendance
dataset has produced enough rows yet to build/exercise the rest of the
pipeline (feature engineering, training, API). Tagged source
'synthetic_bootstrap' so it is never confused with real observations —
contrast with 'historical_derived_proxy' (pipeline/demand_baseline.py), which
is grounded in real government attendance data.

Usage:
    python -m pipeline.synthetic_bootstrap --days 30
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone

import numpy as np

from config import BRANCHES, DEFAULT_DB_PATH, SERVICE_AVG_MINUTES
from pipeline.db import get_connection, insert_queue_samples, upsert_branch
from schemas import QueueReading

logger = logging.getLogger(__name__)


def generate_synthetic_bootstrap(
    days: int = 30, seed: int = 42, branches: tuple | None = None, max_services_per_branch: int | None = None
) -> list[QueueReading]:
    """Seed a plausible-but-synthetic dataset for cold-start development/testing.

    Wait times follow a diurnal pattern (busier mid-morning and mid-afternoon,
    quiet at open/close), heavier on Mondays and paydays, with random noise.

    `branches` defaults to the full config.BRANCHES registry; pass a smaller
    subset (e.g. for a fast test fixture) to bound the output size, since the
    real registry has up to ~40 services per branch. `max_services_per_branch`
    similarly caps how many of each branch's real desk_service_ids are used.
    """
    rng = np.random.default_rng(seed)
    readings: list[QueueReading] = []
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)

    for branch in branches if branches is not None else BRANCHES:
        service_ids = branch.desk_service_ids[:max_services_per_branch] if max_services_per_branch else branch.desk_service_ids
        for desk_service_id in service_ids:
            avg_service_minutes = SERVICE_AVG_MINUTES.get(desk_service_id, 10.0)
            timestamp = start
            while timestamp <= end:
                hour = timestamp.hour
                weekday = timestamp.weekday()
                is_open_hour = 9 <= hour < 17
                if not is_open_hour:
                    timestamp += timedelta(minutes=15)
                    continue

                diurnal = np.sin((hour - 9) / 8 * np.pi) * 10
                monday_bump = 6.0 if weekday == 0 else 0.0
                payday_bump = 5.0 if timestamp.day >= 25 else 0.0
                noise = rng.normal(0, 3)
                people_waiting = max(0, round(diurnal + monday_bump + payday_bump + noise))
                wait_time_minutes = max(0.0, people_waiting * avg_service_minutes + rng.normal(0, 4))

                readings.append(
                    QueueReading(
                        branch_id=branch.branch_id,
                        desk_service_id=desk_service_id,
                        sampled_at=timestamp,
                        people_waiting=int(people_waiting),
                        last_ticket_called=None,
                        estimated_wait_minutes=round(wait_time_minutes, 1),
                        source="synthetic_bootstrap",
                        is_open=True,  # only ever generated within the 9-17 window checked above
                    )
                )
                timestamp += timedelta(minutes=15)

    return readings


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the SQLite history DB with synthetic bootstrap data")
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    readings = generate_synthetic_bootstrap(days=args.days)
    with get_connection(args.db) as connection:
        for branch in BRANCHES:
            upsert_branch(connection, branch.branch_id, branch.name, branch.district, branch.latitude, branch.longitude)
        stored = insert_queue_samples(connection, readings)
    logger.info("Inserted %d synthetic bootstrap rows", stored)


if __name__ == "__main__":
    main()
