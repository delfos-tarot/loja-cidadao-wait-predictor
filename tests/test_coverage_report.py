"""Unit tests for the real-data coverage report (pipeline/coverage_report.py)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
