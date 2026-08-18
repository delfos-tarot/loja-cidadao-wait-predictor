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


# ---------------------------------------------------------------------------
# Fields SIGA returns that the scraper discarded until 2026-08-05
# ---------------------------------------------------------------------------

class _StubClient:
    """Returns one location with the exact `servico` shape a real GetLocais
    response carries — captured by probing the live API on 2026-08-05."""

    def __init__(self, servico: dict) -> None:
        self._servico = servico
        self.calls = 0

    def get_locais(self, *_args, **_kwargs):
        self.calls += 1
        return [{"id": 999, "nome": "Loja X", "servico": self._servico}] if self.calls == 1 else []


_REAL_SERVICO = {
    "estado": "SENHA MANUAL",
    "horario": "09:00 - 12:30",
    "horarioInfoApp": "",
    "id": 1175,
    "idEntidade": 33,
    "idInstituicao": 2,
    "nome": "Atendimento com Marcação",
    "senhaWeb": False,
    "tempoMedAtendimento": 8,
    "tempoRealEspera": 0,
    "utentesEmEspera": 0,
}


def _poll(servico: dict):
    from scrapers.siga_scraper import poll_once

    query = {"distrito_id": 1, "entidade_id": 33, "senha_id": 1175, "senha_nome": "X"}
    return poll_once(_StubClient(servico), [query], {999: "branch_a"})


def test_scraper_captures_real_opening_hours() -> None:
    """The field that replaces config.py's hardcoded Mon-Fri 9-17. A 282-record
    sample found 17 distinct real schedules, the commonest closing at 12:30 —
    five hours before the assumption."""
    reading = _poll(_REAL_SERVICO)[0]
    assert reading.opening_hours == "09:00 - 12:30"


def test_scraper_keeps_the_full_state_string_not_just_a_boolean() -> None:
    reading = _poll(_REAL_SERVICO)[0]
    assert reading.service_state == "SENHA MANUAL"
    assert reading.is_open is True, "'SENHA MANUAL' is a manual-ticketing desk, not a closed one"


def test_opening_hours_captured_even_when_the_desk_is_closed() -> None:
    """A schedule is as true at 22:00 as at noon. Gating it on is_open would
    mean only ever learning the hours of desks that happen to be open."""
    reading = _poll({**_REAL_SERVICO, "estado": "FECHADO"})[0]
    assert reading.opening_hours == "09:00 - 12:30"
    assert reading.is_open is False
    # Live counters, by contrast, mean nothing on a closed desk.
    assert reading.reported_service_minutes is None
    assert reading.people_waiting is None


def test_reported_service_minutes_is_captured_untrusted() -> None:
    """Captured so it CAN be evaluated later; never filtered on the way in.
    A 282-record sample gave median 318 min against IALC-M's measured 7.2, so
    no consumer should read it yet — but a field not captured cannot be
    assessed at all, which is how last_ticket_called ended up 100% empty."""
    reading = _poll({**_REAL_SERVICO, "estado": "ABERTO", "tempoMedAtendimento": 2476})[0]
    assert reading.reported_service_minutes == 2476, "stored raw, not filtered or clamped"


def test_missing_optional_fields_degrade_to_none() -> None:
    """An older or partial SIGA response must not break a sweep."""
    reading = _poll({"estado": "ABERTO", "nome": "Y", "utentesEmEspera": 3})[0]
    assert reading.opening_hours is None
    assert reading.reported_service_minutes is None
    assert reading.web_ticketing is None
    assert reading.people_waiting == 3


def test_csv_fields_all_exist_on_the_reading_dataclass() -> None:
    """CSV_FIELDS and QueueReading must not drift apart. `sampled_at` is the
    only field needing conversion; every other column must be a real attribute
    or the row builder writes a blank."""
    import dataclasses
    from scrapers.siga_scraper import CSV_FIELDS

    attributes = {f.name for f in dataclasses.fields(QueueReading)}
    missing = [c for c in CSV_FIELDS if c not in attributes]
    assert not missing, f"CSV columns with no matching QueueReading field: {missing}"


def test_csv_row_populates_every_declared_field(tmp_path) -> None:
    """The regression guard for the 2026-08-18 bug: CSV_FIELDS gained four
    columns, the row dict was a hand-written literal that did not, and
    DictWriter filled the difference with '' — silently, for 8 days and
    ~120,000 readings, while the header looked perfectly correct."""
    import csv as csv_module
    from datetime import datetime, timezone
    from scrapers.siga_scraper import CSV_FIELDS, append_readings_to_csv

    reading = QueueReading(
        branch_id="b", desk_service_id="s",
        sampled_at=datetime(2026, 8, 18, 10, 0, tzinfo=timezone.utc),
        people_waiting=3, last_ticket_called="A12", estimated_wait_minutes=12.0,
        source="siga_live", is_open=True, raw_wait_time_minutes=12.0,
        opening_hours="09:00 - 12:30", service_state="SENHA MANUAL",
        reported_service_minutes=8.0, web_ticketing=False,
    )
    path = tmp_path / "live.csv"
    append_readings_to_csv([reading], str(path))

    row = next(iter(csv_module.DictReader(path.open())))
    assert set(row) == set(CSV_FIELDS)
    blank = [k for k, v in row.items() if v == ""]
    assert not blank, f"declared CSV columns written blank: {blank}"
    assert row["opening_hours"] == "09:00 - 12:30"
    assert row["reported_service_minutes"] == "8.0"
