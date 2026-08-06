"""Unit tests for pipeline/db.py's rolling-stats lookup used by online inference,
and the SIGA service-name crosswalk applied at load time."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd

import sqlite3
from datetime import datetime, timezone

from pipeline.db import get_connection, get_rolling_wait_stats, init_db, insert_queue_samples, load_all_samples, load_service_crosswalk, upsert_branch
from schemas import QueueReading


def _reading(sampled_at: datetime, wait: float, people_waiting: int = 1) -> QueueReading:
    return QueueReading(
        branch_id="branch_a",
        desk_service_id="Atendimento Geral",
        sampled_at=sampled_at,
        people_waiting=people_waiting,
        last_ticket_called=None,
        estimated_wait_minutes=wait,
        source="siga_live",
        is_open=True,
        raw_wait_time_minutes=wait,
    )


def test_get_rolling_wait_stats_returns_none_when_no_samples(tmp_path) -> None:
    db_path = str(tmp_path / "queue_history.db")
    avg_15min, avg_1h = get_rolling_wait_stats(db_path, "branch_a", "Atendimento Geral", datetime.now(timezone.utc))
    assert avg_15min is None
    assert avg_1h is None


def test_get_rolling_wait_stats_averages_plausible_readings(tmp_path) -> None:
    db_path = str(tmp_path / "queue_history.db")
    now = datetime.now(timezone.utc)
    with get_connection(db_path) as connection:
        upsert_branch(connection, "branch_a", "Branch A", "Lisboa", 38.7, -9.1)
        insert_queue_samples(
            connection,
            [_reading(now - timedelta(minutes=10), 10.0), _reading(now - timedelta(minutes=5), 20.0)],
        )

    avg_15min, avg_1h = get_rolling_wait_stats(db_path, "branch_a", "Atendimento Geral", now)

    assert avg_15min == 15.0
    assert avg_1h == 15.0


def test_get_rolling_wait_stats_excludes_a_frozen_reading_from_the_window(tmp_path) -> None:
    # Found 2026-07-30: a value identical to the prior poll despite real
    # elapsed time is a stale/stuck reading. Without cleaning, this would
    # corrupt the live feature fed to a near-now prediction the same way it
    # corrupted training labels before that fix.
    db_path = str(tmp_path / "queue_history.db")
    now = datetime.now(timezone.utc)
    with get_connection(db_path) as connection:
        upsert_branch(connection, "branch_a", "Branch A", "Lisboa", 38.7, -9.1)
        insert_queue_samples(
            connection,
            [
                # Anchor sits outside both the 15min and 1h windows (65 > 60)
                # -- present only so the 30-min-ago reading has something
                # <=40min back to be compared against and classified frozen.
                _reading(now - timedelta(minutes=65), 450.0),
                # Frozen repeat: the only reading physically inside the 1h
                # window, so it alone determines whether avg_1h is corrupted.
                _reading(now - timedelta(minutes=30), 450.0, people_waiting=0),
            ],
        )

    _, avg_1h = get_rolling_wait_stats(db_path, "branch_a", "Atendimento Geral", now)

    # The only reading inside the 1h window is frozen and gets dropped --
    # nothing left to average over, not a corrupted 450.0.
    assert avg_1h is None


def test_get_rolling_wait_stats_clamps_an_erratic_reading_instead_of_corrupting_the_average(tmp_path) -> None:
    db_path = str(tmp_path / "queue_history.db")
    now = datetime.now(timezone.utc)
    with get_connection(db_path) as connection:
        upsert_branch(connection, "branch_a", "Branch A", "Lisboa", 38.7, -9.1)
        insert_queue_samples(
            connection,
            [
                _reading(now - timedelta(minutes=50), 10.0),  # anchor, outside the 15/1h windows
                _reading(now - timedelta(minutes=10), 5000.0),  # impossible jump, inside window
            ],
        )

    avg_15min, _ = get_rolling_wait_stats(db_path, "branch_a", "Atendimento Geral", now)

    # Clamped to prev(10.0) + 40min-gap * 10.0/min = 410.0, not the raw 5000.0.
    assert avg_15min == 410.0


def test_load_service_crosswalk_returns_the_mapping(tmp_path) -> None:
    crosswalk_path = tmp_path / "crosswalk.json"
    crosswalk_path.write_text(
        json.dumps([{"siga_service_name": "Câmara - Atendimento Geral", "canonical_service_name": "Atendimento Geral", "match_confidence": 0.9}])
    )

    mapping = load_service_crosswalk(crosswalk_path=str(crosswalk_path))

    assert mapping == {"Câmara - Atendimento Geral": "Atendimento Geral"}


def test_load_service_crosswalk_is_empty_when_the_file_is_missing(tmp_path) -> None:
    assert load_service_crosswalk(crosswalk_path=str(tmp_path / "does_not_exist.json")) == {}


def test_load_all_samples_does_not_rename_service_names(tmp_path) -> None:
    # Regression guard for a bug introduced and reverted 2026-07-30: renaming
    # siga_live desk_service_id to its canonical name at load time collapsed
    # multiple physically distinct desks (which share one sweep's timestamp)
    # into a single zero-gap series, so clean_siga_live_readings clamped away
    # genuine between-desk differences. Grouping keys must stay one-per-desk;
    # the crosswalk belongs only in compute_sample_weights' join.
    db_path = str(tmp_path / "queue_history.db")
    now = datetime.now(timezone.utc)
    with get_connection(db_path) as connection:
        upsert_branch(connection, "branch_a", "Branch A", "Lisboa", 38.7, -9.1)
        reading = _reading(now, 10.0)
        reading.desk_service_id = "Câmara - Atendimento Geral"
        insert_queue_samples(connection, [reading])

    frame = load_all_samples(db_path)

    assert frame["desk_service_id"].tolist() == ["Câmara - Atendimento Geral"]


def test_new_live_columns_round_trip(tmp_path) -> None:
    """Fields added 2026-08-05 must survive insert and read-back. Without this,
    the scraper could capture them and the DB silently drop them."""
    from schemas import QueueReading

    db = str(tmp_path / "q.db")
    with get_connection(db) as connection:
        upsert_branch(connection, "branch_a", "Branch A", "Lisboa", 38.7, -9.1)
        insert_queue_samples(connection, [
            QueueReading(
                branch_id="branch_a", desk_service_id="svc",
                sampled_at=datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc),
                people_waiting=2, last_ticket_called=None, estimated_wait_minutes=12.0,
                source="siga_live", is_open=True,
                opening_hours="09:00 - 12:30", service_state="SENHA MANUAL",
                reported_service_minutes=8.0, web_ticketing=False,
            )
        ])
    with sqlite3.connect(db) as connection:
        row = connection.execute(
            "SELECT opening_hours, service_state, reported_service_minutes, web_ticketing "
            "FROM queue_samples"
        ).fetchone()
    assert row == ("09:00 - 12:30", "SENHA MANUAL", 8.0, 0)


def test_migration_adds_columns_to_a_preexisting_db(tmp_path) -> None:
    """An existing DB predates these columns. init_db must ALTER them in rather
    than requiring a rebuild — the live corpus is not regenerable."""
    db = str(tmp_path / "old.db")
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE TABLE queue_samples (id INTEGER PRIMARY KEY AUTOINCREMENT, branch_id TEXT NOT NULL, "
            "desk_service_id TEXT NOT NULL, sampled_at TEXT NOT NULL, people_waiting INTEGER, "
            "last_ticket_called TEXT, wait_time_minutes REAL, source TEXT NOT NULL DEFAULT 'siga_live')"
        )
    init_db(db)
    with sqlite3.connect(db) as connection:
        columns = {r[1] for r in connection.execute("PRAGMA table_info(queue_samples)")}
    assert {"opening_hours", "service_state", "reported_service_minutes", "web_ticketing"} <= columns
