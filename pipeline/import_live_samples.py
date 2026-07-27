"""Merges data/live_samples.csv (appended by CI, e.g. a GitHub Actions
scraper workflow) into the local data/queue_history.db.

Kept as a separate manual step rather than something the scraper does
automatically: the main DB is 400MB+ and gitignored (dominated by proxy
rows), while live_samples.csv is small and git-tracked — they're deliberately
different files with different lifecycles. This script is the bridge: pull
the repo to get CI's latest live_samples.csv, then run this to fold those
real rows into your local training data.

Deduplicates against what's already in queue_history.db by
(branch_id, desk_service_id, sampled_at, source) — safe to re-run on a CSV
that already contains previously-imported rows.

Usage:
    git pull
    python -m pipeline.import_live_samples
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from config import DEFAULT_DB_PATH, REAL_DATA_MAX_PLAUSIBLE_WAIT_MINUTES
from pipeline.db import get_connection, init_db, insert_queue_samples
from schemas import QueueReading

logger = logging.getLogger(__name__)

LIVE_SAMPLES_CSV_PATH = "data/live_samples.csv"


def load_existing_keys(db_path: str) -> set[tuple[str, str, str, str]]:
    """Returns the (branch_id, desk_service_id, sampled_at, source) keys
    already present, so re-running this script on an already-imported CSV
    is a safe no-op rather than inserting duplicates."""
    init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT branch_id, desk_service_id, sampled_at, source FROM queue_samples WHERE source = 'siga_live'"
        ).fetchall()
    return set(rows)


def load_csv_readings(csv_path: str) -> pd.DataFrame:
    if not Path(csv_path).exists():
        raise SystemExit(f"{csv_path} not found — nothing to import yet")
    frame = pd.read_csv(csv_path)
    return frame


def import_live_samples(csv_path: str = LIVE_SAMPLES_CSV_PATH, db_path: str = DEFAULT_DB_PATH) -> tuple[int, int]:
    """Returns (imported_count, skipped_duplicate_count)."""
    frame = load_csv_readings(csv_path)
    existing_keys = load_existing_keys(db_path)

    # Backward compatibility: CSVs collected before raw_wait_time_minutes
    # existed have no such column, and their estimated_wait_minutes was never
    # plausibility-filtered (may contain the same implausible values found
    # 2026-07-27 — see config.REAL_DATA_MAX_PLAUSIBLE_WAIT_MINUTES). Treat
    # that old value as the raw one, and (re-)apply the filter here so
    # already-collected rows get retroactively cleaned on import too.
    has_raw_column = "raw_wait_time_minutes" in frame.columns

    readings: list[QueueReading] = []
    skipped = 0
    for row in frame.itertuples(index=False):
        key = (row.branch_id, row.desk_service_id, row.sampled_at, row.source)
        if key in existing_keys:
            skipped += 1
            continue

        estimated = None if pd.isna(row.estimated_wait_minutes) else float(row.estimated_wait_minutes)
        if has_raw_column:
            raw = None if pd.isna(row.raw_wait_time_minutes) else float(row.raw_wait_time_minutes)
        else:
            raw = estimated
            if estimated is not None and estimated > REAL_DATA_MAX_PLAUSIBLE_WAIT_MINUTES:
                estimated = None

        readings.append(
            QueueReading(
                branch_id=row.branch_id,
                desk_service_id=row.desk_service_id,
                sampled_at=pd.Timestamp(row.sampled_at).to_pydatetime(),
                people_waiting=None if pd.isna(row.people_waiting) else int(row.people_waiting),
                last_ticket_called=None if pd.isna(row.last_ticket_called) else row.last_ticket_called,
                estimated_wait_minutes=estimated,
                source=row.source,
                is_open=None if pd.isna(row.is_open) else bool(row.is_open),
                raw_wait_time_minutes=raw,
            )
        )

    with get_connection(db_path) as connection:
        imported = insert_queue_samples(connection, readings)

    return imported, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Import CI-collected live_samples.csv into the local queue_history.db")
    parser.add_argument("--csv", default=LIVE_SAMPLES_CSV_PATH)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    imported, skipped = import_live_samples(args.csv, args.db)
    logger.info("Imported %d new live samples (%d already present, skipped)", imported, skipped)


if __name__ == "__main__":
    main()
