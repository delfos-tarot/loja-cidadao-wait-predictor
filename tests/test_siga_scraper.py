"""Unit tests for scrapers/siga_scraper.py's dedup and plausibility-filtering
logic — no network calls."""

from __future__ import annotations

from datetime import datetime, timezone

from config import REAL_DATA_MAX_PLAUSIBLE_WAIT_MINUTES
from scrapers.siga_scraper import _dedupe_readings, poll_once
from schemas import QueueReading


class _StubSigaClient:
    """Returns a fixed set of locais for any query, for testing poll_once's
    parsing/filtering logic without a real network call."""

    def __init__(self, locais: list[dict]) -> None:
        self._locais = locais

    def get_locais(self, id_distrito, id_entidade, id_senha, id_instituicao=0):
        return self._locais


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


def _location(location_id: int, estado: str, utentes: int | None, tempo_espera: float | None) -> dict:
    return {
        "id": location_id,
        "nome": "Loja de Cidadao Test",
        "servico": {"nome": "Atendimento Geral", "estado": estado, "utentesEmEspera": utentes, "tempoRealEspera": tempo_espera},
    }


def test_poll_once_keeps_plausible_wait_time_in_both_fields() -> None:
    client = _StubSigaClient([_location(1, "ABERTO", 3, 12.0)])
    readings = poll_once(client, [{"distrito_id": 11, "entidade_id": 1, "senha_id": 1, "senha_nome": "Atendimento Geral"}], {1: "branch_a"})

    assert len(readings) == 1
    assert readings[0].estimated_wait_minutes == 12.0
    assert readings[0].raw_wait_time_minutes == 12.0


def test_poll_once_filters_implausible_wait_time_but_keeps_raw() -> None:
    # The exact real-world pattern found 2026-07-27: 0 people waiting, huge "wait".
    implausible = REAL_DATA_MAX_PLAUSIBLE_WAIT_MINUTES + 1000
    client = _StubSigaClient([_location(1, "ABERTO", 0, implausible)])
    readings = poll_once(client, [{"distrito_id": 11, "entidade_id": 1, "senha_id": 1, "senha_nome": "Atendimento Geral"}], {1: "branch_a"})

    assert len(readings) == 1
    # Filtered out of the field everything else (training, API) reads...
    assert readings[0].estimated_wait_minutes is None
    # ...but the untouched value is never lost.
    assert readings[0].raw_wait_time_minutes == implausible


def test_poll_once_sets_both_fields_none_when_closed() -> None:
    client = _StubSigaClient([_location(1, "FECHADO", None, None)])
    readings = poll_once(client, [{"distrito_id": 11, "entidade_id": 1, "senha_id": 1, "senha_nome": "Atendimento Geral"}], {1: "branch_a"})

    assert len(readings) == 1
    assert readings[0].estimated_wait_minutes is None
    assert readings[0].raw_wait_time_minutes is None
    assert readings[0].is_open is False
