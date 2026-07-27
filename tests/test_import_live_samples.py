"""Unit tests for pipeline/import_live_samples.py — the CSV-to-DB merge step."""

from __future__ import annotations

from datetime import datetime, timezone

from pipeline.db import load_all_samples
from pipeline.import_live_samples import import_live_samples
from scrapers.siga_scraper import append_readings_to_csv
from schemas import QueueReading


def _reading(branch_id: str, desk_service_id: str, sampled_at: datetime) -> QueueReading:
    return QueueReading(
        branch_id=branch_id,
        desk_service_id=desk_service_id,
        sampled_at=sampled_at,
        people_waiting=3,
        last_ticket_called=None,
        estimated_wait_minutes=7.5,
        source="siga_live",
        is_open=True,
    )


def test_import_live_samples_inserts_new_rows(tmp_path) -> None:
    csv_path = str(tmp_path / "live_samples.csv")
    db_path = str(tmp_path / "queue_history.db")
    now = datetime.now(timezone.utc)
    append_readings_to_csv([_reading("branch_a", "Atendimento Geral", now)], csv_path)

    imported, skipped = import_live_samples(csv_path, db_path)

    assert imported == 1
    assert skipped == 0
    frame = load_all_samples(db_path)
    assert len(frame) == 1
    assert frame.iloc[0]["source"] == "siga_live"


def test_import_live_samples_is_idempotent_on_rerun(tmp_path) -> None:
    csv_path = str(tmp_path / "live_samples.csv")
    db_path = str(tmp_path / "queue_history.db")
    now = datetime.now(timezone.utc)
    append_readings_to_csv([_reading("branch_a", "Atendimento Geral", now)], csv_path)

    import_live_samples(csv_path, db_path)
    imported_second_run, skipped_second_run = import_live_samples(csv_path, db_path)

    # Re-running on a CSV that's already been imported must not duplicate rows.
    assert imported_second_run == 0
    assert skipped_second_run == 1
    frame = load_all_samples(db_path)
    assert len(frame) == 1


def test_import_live_samples_appends_incrementally(tmp_path) -> None:
    csv_path = str(tmp_path / "live_samples.csv")
    db_path = str(tmp_path / "queue_history.db")
    first_time = datetime.now(timezone.utc)

    append_readings_to_csv([_reading("branch_a", "Atendimento Geral", first_time)], csv_path)
    import_live_samples(csv_path, db_path)

    # CI runs again later, appending a genuinely new row to the same CSV.
    second_time = datetime.now(timezone.utc)
    append_readings_to_csv([_reading("branch_a", "Atendimento Geral", second_time)], csv_path)
    imported, skipped = import_live_samples(csv_path, db_path)

    assert imported == 1
    assert skipped == 1  # the first row, already present
    frame = load_all_samples(db_path)
    assert len(frame) == 2
