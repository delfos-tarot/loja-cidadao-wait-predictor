"""Shared dataclasses used across scrapers, pipeline, and API layers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class QueueReading:
    branch_id: str
    desk_service_id: str
    sampled_at: datetime
    people_waiting: int | None
    last_ticket_called: str | None
    estimated_wait_minutes: float | None
    source: str = "siga_live"
    is_open: bool | None = None
    # Untouched original value before any plausibility filtering was applied
    # to estimated_wait_minutes (see config.REAL_DATA_MAX_PLAUSIBLE_WAIT_MINUTES).
    # Kept so a value nulled out of estimated_wait_minutes for looking
    # implausible is never actually lost — just not trusted by default.
    raw_wait_time_minutes: float | None = None
    # How many real attendances back this reading's estimated_wait_minutes —
    # only meaningful for source='historical_real_daily_avg' (IALC-M branch-day
    # averages, see pipeline/ingest_real_wait_times.py), where some branch-days
    # carry as few as 1 real attendance, making that day's "average" a single
    # noisy raw observation. None for every other source (a single siga_live
    # reading or a formula-derived proxy row isn't an aggregate of anything).
    sample_size: int | None = None

    # ----------------------------------------------------------------------
    # Fields SIGA returns on every `servico` object that were discarded on all
    # 304,190 readings collected before 2026-08-05. Capturing a field costs
    # nothing; NOT capturing it means the history can never be reconstructed —
    # exactly the position `last_ticket_called` left us in (0 non-null rows).
    # siga_live only; None for every other source.
    # ----------------------------------------------------------------------

    # SIGA's real per-service schedule, e.g. '09:00 - 12:30'. THE VALUABLE ONE.
    # config.ASSUMED_BUSINESS_HOUR_START/END hardcode Mon-Fri 9-17 for every
    # branch, with a comment stating no real schedule data exists anywhere. It
    # does, and SIGA has returned it all along: a 282-record sample found 17
    # distinct schedules, the most common ('09:00 - 12:30', 154 of 282) closing
    # five hours before the hardcoded assumption.
    opening_hours: str | None = None

    # Full `estado` string, rather than the boolean `is_open` collapses it to.
    # Observed: 'FECHADO', 'SENHA MANUAL' — the latter meaning manual ticketing,
    # where live queue figures may not mean what they do elsewhere.
    service_state: str | None = None

    # SIGA's `tempoMedAtendimento`. CAPTURED BUT NOT TRUSTED: a 282-record
    # sample gave median 318 min (range 0-2476) against IALC-M's MEASURED 7.2.
    # Named `reported_` to keep that distance visible at every call site.
    # Evaluate against a DAYTIME sample before any consumer reads it — the
    # probe behind those numbers ran at 22:00 with most desks closed, so stale
    # end-of-day counters are not ruled out.
    reported_service_minutes: float | None = None

    # Whether the service offers web ticketing (`senhaWeb`) — plausibly changes
    # what a physical queue length means.
    web_ticketing: bool | None = None
