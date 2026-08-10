"""SQLite persistence layer for queue history samples.

Schema:
  branches      - reference table of known Loja do Cidadao locations
  queue_samples - time-stamped polling samples (live SIGA reads and/or
                  cleaned historical CSV rows), the raw training source for
                  the wait-time model.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from config import (
    SIGA_MAX_PLAUSIBLE_WAIT_CHANGE_PER_MINUTE,
    SIGA_STALENESS_CHECK_MAX_GAP_MINUTES,
    SIGA_STALENESS_CHECK_MIN_WAIT_MINUTES,
)
from schemas import QueueReading

logger = logging.getLogger(__name__)

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
    sample_size INTEGER,
    opening_hours TEXT,
    service_state TEXT,
    reported_service_minutes REAL,
    web_ticketing INTEGER
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
        # Added 2026-08-05: SIGA returns these on every servico object and the
        # scraper had been discarding them. Existing rows keep NULL — the
        # history genuinely was not captured and must not be back-filled with
        # a guess.
        for column, sql_type in (
            ("opening_hours", "TEXT"),
            ("service_state", "TEXT"),
            ("reported_service_minutes", "REAL"),
            ("web_ticketing", "INTEGER"),
        ):
            try:
                connection.execute(f"ALTER TABLE queue_samples ADD COLUMN {column} {sql_type}")
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
             last_ticket_called, wait_time_minutes, source, is_open, raw_wait_time_minutes, sample_size,
             opening_hours, service_state, reported_service_minutes, web_ticketing)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            reading.opening_hours,
            reading.service_state,
            reading.reported_service_minutes,
            None if reading.web_ticketing is None else int(reading.web_ticketing),
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


SERVICE_CROSSWALK_PATH = "data/siga_desk_service_crosswalk.json"


def load_service_crosswalk(crosswalk_path: str = SERVICE_CROSSWALK_PATH) -> dict[str, str]:
    """Returns {siga_service_name: canonical_dados_gov_name} from
    data/siga_desk_service_crosswalk.json (pipeline/reconcile_siga_services.py),
    or {} if it hasn't been generated yet — see that module and
    pipeline/coverage_report.py's docstrings for the problem it solves: a
    SIGA name like "Câmara - Atendimento Geral" and the dados.gov.pt name
    "Atendimento Geral" are the same real service, but
    compute_sample_weights joins on the raw string, so an unreconciled
    combo's proxy rows never see their weight decay from real live coverage.

    **Deliberately returns the mapping instead of applying it**, and callers
    must only use it for *joining/counting* — never to rewrite
    `desk_service_id` on the frame itself. Tried the latter first
    (2026-07-30, renaming siga_live rows inside load_all_samples) and it
    actively corrupted the data: several branches run multiple physically
    distinct desks whose SIGA names all reduce to the same canonical name
    (Coimbra alone has 5 — "ATENDIMENTO GERAL", "Atendimento Geral
    Empresa", "Câmara - Atendimento Geral", ...), and every desk at a
    branch is scraped in the same sweep, so they share an identical
    `sampled_at`. Renaming collapsed them into one series whose consecutive
    rows had a zero-minute gap, which made clean_siga_live_readings'
    `max_plausible_delta` zero — so any genuine difference between two real
    desks was flagged erratic and clamped away. Measured: clamps jumped
    336 -> 764 and the siga_live segment got worse (MAE 29.5 -> 30.3,
    R^2 0.496 -> 0.481) on an otherwise-identical retrain. Grouping keys
    for time-series work must stay one-per-real-desk.
    """
    try:
        entries = json.loads(Path(crosswalk_path).read_text())
    except FileNotFoundError:
        return {}
    return {entry["siga_service_name"]: entry["canonical_service_name"] for entry in entries}


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


def clean_siga_live_readings(
    frame: pd.DataFrame,
    target_column: str = "wait_time_minutes",
    min_wait_minutes: float = SIGA_STALENESS_CHECK_MIN_WAIT_MINUTES,
    max_gap_minutes: float = SIGA_STALENESS_CHECK_MAX_GAP_MINUTES,
    max_change_per_minute: float = SIGA_MAX_PLAUSIBLE_WAIT_CHANGE_PER_MINUTE,
) -> pd.DataFrame:
    """Corrects two failure modes found 2026-07-30 in `source='siga_live'`
    rows with `wait_time_minutes` near config.REAL_DATA_MAX_PLAUSIBLE_WAIT_MINUTES's
    ceiling -- see config.py's docstring above SIGA_STALENESS_CHECK_MIN_WAIT_MINUTES
    for the full diagnosis. Compares each reading only to that same
    (branch_id, desk_service_id)'s immediately preceding poll, and only when
    they're close enough in time (<= max_gap_minutes) to be meaningfully
    comparable -- a gap spanning an overnight/outage tells you nothing about
    staleness or rate of change.

    - Frozen repeats (identical to the prior poll despite real elapsed time,
      at >= min_wait_minutes): target set to NaN, so a caller doing its own
      dropna(subset=[target_column]) excludes them. True information loss is
      zero -- a stuck value adds nothing beyond the already-kept prior
      reading.
    - Erratic swings (change far exceeding max_change_per_minute's plausible
      rate): clamped to the nearest plausible bound around the prior
      reading, not dropped -- preserves a corrected data point instead of
      losing the row entirely.

    Only ever touches siga_live rows; every other source passes through
    completely untouched, since none of them showed this pattern (checked
    2026-07-30: max wait_time_minutes of 180 and 417 respectively for
    historical_derived_proxy/historical_real_daily_avg, both far under the
    ceiling where this pathology was found).

    Lives here (not pipeline/train.py, its original home) so both the
    offline training frame (pipeline/train.py's load_training_frame) and
    the online single-row rolling-stats lookup (get_rolling_wait_stats
    below, used by api/service.py for near-now predictions) share the exact
    same cleaning logic -- the same reason QueueFeatureTransformer is
    shared between the two rather than reimplemented per call site.
    """
    is_live = frame["source"] == "siga_live"
    if not is_live.any():
        return frame

    frame = frame.copy()
    live = frame.loc[is_live, ["branch_id", "desk_service_id", "sampled_at", target_column]].sort_values(
        ["branch_id", "desk_service_id", "sampled_at"]
    )
    grouped = live.groupby(["branch_id", "desk_service_id"], sort=False)
    prev_wait = grouped[target_column].shift(1)
    prev_time = grouped["sampled_at"].shift(1)
    gap_minutes = (live["sampled_at"] - prev_time).dt.total_seconds() / 60.0

    comparable = gap_minutes.notna() & (gap_minutes <= max_gap_minutes)
    is_frozen = comparable & (live[target_column] >= min_wait_minutes) & (live[target_column] == prev_wait)

    max_plausible_delta = gap_minutes * max_change_per_minute
    delta = (live[target_column] - prev_wait).abs()
    is_erratic = comparable & ~is_frozen & prev_wait.notna() & (delta > max_plausible_delta)

    dropped = int(is_frozen.sum())
    clamped = int(is_erratic.sum())

    if dropped:
        frame.loc[live.index[is_frozen], target_column] = np.nan

    if clamped:
        bounded = live[target_column].clip(lower=prev_wait - max_plausible_delta, upper=prev_wait + max_plausible_delta)
        bounded = bounded.clip(lower=0.0)
        frame.loc[live.index[is_erratic], target_column] = bounded[is_erratic]

    if dropped or clamped:
        logger.info(
            "clean_siga_live_readings: dropped %d frozen repeats, clamped %d erratic swings (of %d siga_live rows)",
            dropped, clamped, len(live),
        )
    return frame


def get_rolling_wait_stats(
    db_path: str, branch_id: str, desk_service_id: str, as_of: datetime
) -> tuple[float | None, float | None]:
    """Returns (avg_wait_last_15min, avg_wait_last_1h) ending at `as_of`.

    Either element is None if no samples with a known wait_time_minutes fall
    inside that window; callers should apply a statistical baseline fallback.

    Runs readings through clean_siga_live_readings before averaging (found
    2026-07-30 -- see that function's docstring) rather than a plain SQL
    AVG(): a frozen or erratic siga_live reading landing inside the window
    would otherwise corrupt this feature for a live near-now prediction the
    same way it corrupted training labels before that fix. Queries an extra
    max_gap_minutes of lookback beyond the 1h window so the earliest
    in-window reading still has a real prior reading to compare against,
    matching how the offline training path always has full history
    available for that same comparison.
    """
    init_db(db_path)
    lookback_start = as_of - timedelta(hours=1, minutes=SIGA_STALENESS_CHECK_MAX_GAP_MINUTES)

    query = """
        SELECT branch_id, desk_service_id, sampled_at, wait_time_minutes, source
        FROM queue_samples
        WHERE branch_id = ? AND desk_service_id = ?
          AND source = 'siga_live'
          AND sampled_at BETWEEN ? AND ?
        ORDER BY sampled_at
    """
    with sqlite3.connect(db_path) as connection:
        frame = pd.read_sql_query(query, connection, params=(branch_id, desk_service_id, lookback_start.isoformat(), as_of.isoformat()))

    if frame.empty:
        return None, None

    frame["sampled_at"] = pd.to_datetime(frame["sampled_at"], utc=True)
    frame = clean_siga_live_readings(frame)
    frame = frame.dropna(subset=["wait_time_minutes"])

    window_15min_start = as_of - timedelta(minutes=15)
    window_1h_start = as_of - timedelta(hours=1)
    in_15min = frame[frame["sampled_at"] >= window_15min_start]
    in_1h = frame[frame["sampled_at"] >= window_1h_start]

    return (
        float(in_15min["wait_time_minutes"].mean()) if len(in_15min) else None,
        float(in_1h["wait_time_minutes"].mean()) if len(in_1h) else None,
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
