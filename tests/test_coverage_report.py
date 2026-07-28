"""Unit tests for the real-data coverage report (pipeline/coverage_report.py)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from pipeline.coverage_report import build_coverage_report
from pipeline.db import get_connection, insert_queue_samples, upsert_branch
from schemas import QueueReading


def test_build_coverage_report_empty_when_no_live_data(tmp_path) -> None:
    db_path = str(tmp_path / "empty.db")
    report = build_coverage_report(db_path)
    assert report.empty


def test_build_coverage_report_counts_per_combo(tmp_path) -> None:
    db_path = str(tmp_path / "queue_history.db")
    now = datetime.now(timezone.utc)
    with get_connection(db_path) as connection:
        upsert_branch(connection, "branch_a", "Branch A", "Lisboa", 38.7, -9.1)
        insert_queue_samples(
            connection,
            [
                QueueReading(
                    branch_id="branch_a",
                    desk_service_id="Atendimento Geral",
                    sampled_at=now - timedelta(minutes=15 * i),
                    people_waiting=5,
                    last_ticket_called=None,
                    estimated_wait_minutes=10.0,
                    source="siga_live",
                    is_open=True,
                )
                for i in range(3)
            ]
            + [
                # A different source must not count toward live coverage.
                QueueReading(
                    branch_id="branch_a",
                    desk_service_id="Atendimento Geral",
                    sampled_at=now,
                    people_waiting=5,
                    last_ticket_called=None,
                    estimated_wait_minutes=10.0,
                    source="historical_derived_proxy",
                    is_open=True,
                )
            ],
        )

    report = build_coverage_report(db_path)

    assert len(report) == 1
    row = report.iloc[0]
    assert row["branch_id"] == "branch_a"
    assert row["desk_service_id"] == "Atendimento Geral"
    assert row["live_count"] == 3  # only the siga_live rows, not the proxy row
    assert row["matches_proxy_vocabulary"] == True  # noqa: E712 — pandas bool, not Python bool


def test_build_coverage_report_excludes_filtered_implausible_readings_from_live_count(tmp_path) -> None:
    # Found 2026-07-27: a reading filtered as implausible (estimated_wait_minutes
    # None, per config.REAL_DATA_MAX_PLAUSIBLE_WAIT_MINUTES) never reaches
    # training -- load_training_frame drops null-target rows before
    # compute_sample_weights ever runs. Counting it toward live_count here
    # would overstate real coverage for exactly the question this report
    # exists to answer.
    #
    # Also covers a real bug found the same day in the *first* version of
    # this fix: a closed-desk reading (no raw_wait_time_minutes at all --
    # nothing for SIGA to even attempt) must NOT count toward
    # implausible_rate's denominator alongside genuinely garbage readings.
    # Conflating the two inflated a true ~32% implausible rate into a
    # falsely-reported ~86%.
    db_path = str(tmp_path / "queue_history.db")
    now = datetime.now(timezone.utc)
    with get_connection(db_path) as connection:
        upsert_branch(connection, "branch_a", "Branch A", "Lisboa", 38.7, -9.1)
        insert_queue_samples(
            connection,
            [
                QueueReading(
                    branch_id="branch_a",
                    desk_service_id="Atendimento Geral",
                    sampled_at=now - timedelta(minutes=15 * i),
                    people_waiting=5,
                    last_ticket_called=None,
                    estimated_wait_minutes=10.0,
                    source="siga_live",
                    is_open=True,
                    raw_wait_time_minutes=10.0,
                )
                for i in range(4)
            ]
            + [
                QueueReading(
                    branch_id="branch_a",
                    desk_service_id="Atendimento Geral",
                    sampled_at=now,
                    people_waiting=120,
                    last_ticket_called=None,
                    estimated_wait_minutes=None,  # filtered as implausible
                    source="siga_live",
                    is_open=True,
                    raw_wait_time_minutes=5000.0,  # untouched raw value is kept regardless
                )
            ]
            + [
                # A closed-desk reading: SIGA had no wait value to return at
                # all, so this must not count as an "implausible" reading --
                # it's not garbage, it's just not applicable.
                QueueReading(
                    branch_id="branch_a",
                    desk_service_id="Atendimento Geral",
                    sampled_at=now - timedelta(hours=8),
                    people_waiting=None,
                    last_ticket_called=None,
                    estimated_wait_minutes=None,
                    source="siga_live",
                    is_open=False,
                    raw_wait_time_minutes=None,
                )
            ],
        )

    report = build_coverage_report(db_path)

    assert len(report) == 1
    row = report.iloc[0]
    assert row["live_count"] == 4  # only the usable readings
    assert row["raw_attempt_count"] == 6  # every siga_live row, including the closed one
    assert row["raw_wait_attempt_count"] == 5  # desk-open readings only -- excludes the closed one
    assert row["implausible_rate"] == pytest.approx(0.2)  # 1 of 5 desk-open readings, NOT 2 of 6


def test_build_coverage_report_flags_orphaned_service_names(tmp_path) -> None:
    db_path = str(tmp_path / "queue_history.db")
    now = datetime.now(timezone.utc)
    with get_connection(db_path) as connection:
        upsert_branch(connection, "branch_a", "Branch A", "Lisboa", 38.7, -9.1)
        insert_queue_samples(
            connection,
            [
                QueueReading(
                    branch_id="branch_a",
                    desk_service_id="Geral",  # SIGA's short name, not in DESK_SERVICES
                    sampled_at=now,
                    people_waiting=2,
                    last_ticket_called=None,
                    estimated_wait_minutes=5.0,
                    source="siga_live",
                    is_open=True,
                )
            ],
        )

    report = build_coverage_report(db_path)

    assert len(report) == 1
    assert report.iloc[0]["matches_proxy_vocabulary"] == False  # noqa: E712
