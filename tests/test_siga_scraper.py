"""Unit tests for scrapers/siga_scraper.py's dedup logic — no network calls."""

from __future__ import annotations

from datetime import datetime, timezone

from scrapers.siga_scraper import _dedupe_readings
from schemas import QueueReading


def _reading(branch_id: str, desk_service_id: str, is_open: bool, people_waiting: int | None) -> QueueReading:
    return QueueReading(
        branch_id=branch_id,
        desk_service_id=desk_service_id,
        sampled_at=datetime.now(timezone.utc),
        people_waiting=people_waiting,
        last_ticket_called=None,
        estimated_wait_minutes=float(people_waiting) if people_waiting is not None else None,
        source="siga_live",
        is_open=is_open,
    )


def test_dedupe_collapses_same_combo_reached_via_multiple_query_paths() -> None:
    # Same physical (branch, service) reachable through two different
    # (entidade, senha) paths in a single poll — the exact scenario found via
    # pipeline/coverage_report.py (duplicate entidade entries in SIGA's data).
    readings = [
        _reading("branch_a", "Atendimento Geral", True, 5),
        _reading("branch_a", "Atendimento Geral", True, 5),
    ]

    deduped = _dedupe_readings(readings)

    assert len(deduped) == 1
    assert deduped[0].branch_id == "branch_a"


def test_dedupe_prefers_open_reading_over_closed_duplicate() -> None:
    readings = [
        _reading("branch_a", "Atendimento Geral", False, None),
        _reading("branch_a", "Atendimento Geral", True, 8),
    ]

    deduped = _dedupe_readings(readings)

    assert len(deduped) == 1
    assert deduped[0].is_open is True
    assert deduped[0].people_waiting == 8


def test_dedupe_keeps_distinct_combos_separate() -> None:
    readings = [
        _reading("branch_a", "Atendimento Geral", True, 5),
        _reading("branch_a", "Tesouraria", True, 2),
        _reading("branch_b", "Atendimento Geral", True, 1),
    ]

    deduped = _dedupe_readings(readings)

    assert len(deduped) == 3
