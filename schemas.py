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
