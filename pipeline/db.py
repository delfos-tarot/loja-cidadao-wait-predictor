"""SQLite persistence layer for queue history samples.

Schema:
  branches      - reference table of known Loja do Cidadao locations
  queue_samples - time-stamped polling samples (live SIGA reads and/or
                  cleaned historical CSV rows), the raw training source for
                  the wait-time model.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd

from schemas import QueueReading

SCHEMA = """
CREATE TABLE IF NOT EXISTS branches (
    branch_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    district TEXT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS queue_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id TEXT NOT NULL,
    desk_service_id TEXT NOT NULL,
    sampled_at TEXT NOT NULL,
    people_waiting INTEGER,
    last_ticket_called TEXT,
    wait_time_minutes REAL,
    source TEXT NOT NULL DEFAULT 'siga_live',
    is_open INTEGER,
    raw_wait_time_minutes REAL,
    sample_size INTEGER
);

CREATE INDEX IF NOT EXISTS idx_queue_samples_lookup
    ON queue_samples (branch_id, desk_service_id, sampled_at);
"""


def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.executescript(SCHEMA)
        # CREATE TABLE IF NOT EXISTS doesn't add columns to an already-existing
        # table from an older schema version — migrate existing DBs in place.
        try:
            connection.execute("ALTER TABLE queue_samples ADD COLUMN raw_wait_time_minutes REAL")
        except sqlite3.OperationalError:
            pass  # column already exists
        try:
            connection.execute("ALTER TABLE queue_samples ADD COLUMN sample_size INTEGER")
        except sqlite3.OperationalError:
            pass  # column already exists


@contextmanager
def get_connection(db_path: str) -> Iterator[sqlite3.Connection]:
    init_db(db_path)
    connection = sqlite3.connect(db_path)
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def upsert_branch(
    connection: sqlite3.Connection,
    branch_id: str,
    name: str,
    district: str,
    latitude: float,
    longitude: float,
) -> None:
    connection.execute(
        """
        INSERT INTO branches (branch_id, name, district, latitude, longitude)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(branch_id) DO UPDATE SET
            name = excluded.name,
            district = excluded.district,
            latitude = excluded.latitude,
            longitude = excluded.longitude
        """,
        (branch_id, name, district, latitude, longitude),
    )


def insert_queue_sample(connection: sqlite3.Connection, reading: QueueReading) -> None:
    connection.execute(
        """
        INSERT INTO queue_samples
            (branch_id, desk_service_id, sampled_at, people_waiting,
             last_ticket_called, wait_time_minutes, source, is_open, raw_wait_time_minutes, sample_size)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            reading.branch_id,
            reading.desk_service_id,
            reading.sampled_at.isoformat(),
            reading.people_waiting,
            reading.last_ticket_called,
            reading.estimated_wait_minutes,
            reading.source,
            None if reading.is_open is None else int(reading.is_open),
            reading.raw_wait_time_minutes,
            reading.sample_size,
        ),
    )


def insert_queue_samples(connection: sqlite3.Connection, readings: list[QueueReading]) -> int:
    for reading in readings:
        insert_queue_sample(connection, reading)
    return len(readings)


def delete_samples_by_source(db_path: str, source: str) -> int:
    """Deletes every queue_samples row for a fully-regenerable source
    (historical_derived_proxy, historical_real_daily_avg, synthetic_bootstrap)
    before re-inserting a fresh batch — without this, re-running a generator
    script (e.g. after changing config.DIURNAL_SNAPSHOTS) would leave stale
    rows from the old run sitting alongside the new ones instead of being
    replaced. Never call this with source='siga_live' — that's real,
    non-regenerable data."""
    init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute("DELETE FROM queue_samples WHERE source = ?", (source,))
        connection.commit()
        return cursor.rowcount


def load_all_samples(db_path: str) -> pd.DataFrame:
    init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        frame = pd.read_sql_query(
            "SELECT * FROM queue_samples ORDER BY sampled_at",
            connection,
        )
    if not frame.empty:
        # format="ISO8601" (not a single inferred format): proxy/synthetic
        # rows are whole-second timestamps while real siga_live rows carry
        # microsecond precision (datetime.now().isoformat()) — a single
        # inferred format breaks the moment both appear in the same table.
        frame["sampled_at"] = pd.to_datetime(frame["sampled_at"], format="ISO8601", utc=True)
    return frame


def count_samples(db_path: str) -> int:
    init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute("SELECT COUNT(*) FROM queue_samples")
        return int(cursor.fetchone()[0])


def get_latest_people_waiting(db_path: str, branch_id: str, desk_service_id: str) -> tuple[int | None, str | None]:
    """Returns (people_waiting, sampled_at_iso) for the most recent live/historical
    sample, or (None, None) if no reading is available for this branch/service."""
    init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            """
            SELECT people_waiting, sampled_at FROM queue_samples
            WHERE branch_id = ? AND desk_service_id = ? AND people_waiting IS NOT NULL
            ORDER BY sampled_at DESC LIMIT 1
            """,
            (branch_id, desk_service_id),
        )
        row = cursor.fetchone()
    if row is None:
        return None, None
    return int(row[0]), row[1]


def get_latest_open_status(db_path: str, branch_id: str, desk_service_id: str) -> bool | None:
    """Returns the most recent recorded is_open state for this branch/service,
    or None if no reading with a known is_open value exists yet (e.g. only
    proxy/synthetic rows recorded before live scraping started, or the
    real siga_live scraper hasn't polled this combo)."""
    init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            """
            SELECT is_open FROM queue_samples
            WHERE branch_id = ? AND desk_service_id = ? AND is_open IS NOT NULL
            ORDER BY sampled_at DESC LIMIT 1
            """,
            (branch_id, desk_service_id),
        )
        row = cursor.fetchone()
    return bool(row[0]) if row is not None else None


def get_rolling_wait_stats(
    db_path: str, branch_id: str, desk_service_id: str, as_of: datetime
) -> tuple[float | None, float | None]:
    """Returns (avg_wait_last_15min, avg_wait_last_1h) ending at `as_of`.

    Either element is None if no samples with a known wait_time_minutes fall
    inside that window; callers should apply a statistical baseline fallback.
    """
    init_db(db_path)
    as_of_iso = as_of.isoformat()
    window_15min_start = (as_of - timedelta(minutes=15)).isoformat()
    window_1h_start = (as_of - timedelta(hours=1)).isoformat()

    query = """
        SELECT AVG(wait_time_minutes) FROM queue_samples
        WHERE branch_id = ? AND desk_service_id = ?
          AND sampled_at BETWEEN ? AND ? AND wait_time_minutes IS NOT NULL
    """
    with sqlite3.connect(db_path) as connection:
        avg_15min = connection.execute(query, (branch_id, desk_service_id, window_15min_start, as_of_iso)).fetchone()[0]
        avg_1h = connection.execute(query, (branch_id, desk_service_id, window_1h_start, as_of_iso)).fetchone()[0]

    return (
        float(avg_15min) if avg_15min is not None else None,
        float(avg_1h) if avg_1h is not None else None,
    )


def get_historical_avg_attendances(
    db_path: str, branch_id: str, desk_service_id: str, day_of_week: int
) -> float | None:
    """Returns the real historical average daily attendances for this
    (branch, service, day_of_week), or None if the `historical_demand_baseline`
    table doesn't exist yet (pipeline/demand_baseline.py hasn't been run) or
    has no row for this combination — callers must apply a baseline fallback.
    """
    init_db(db_path)
    try:
        with sqlite3.connect(db_path) as connection:
            cursor = connection.execute(
                """
                SELECT avg_attendances FROM historical_demand_baseline
                WHERE branch_id = ? AND desk_service_id = ? AND day_of_week = ?
                """,
                (branch_id, desk_service_id, day_of_week),
            )
            row = cursor.fetchone()
    except sqlite3.OperationalError:
        return None
    return float(row[0]) if row is not None else None
