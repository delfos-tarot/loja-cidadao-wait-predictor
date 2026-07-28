"""Unit tests for pipeline/ingest_real_wait_times.py — turning real IALC-M
branch-day averages into queue_samples rows tagged historical_real_daily_avg.
"""

from __future__ import annotations

import pandas as pd

from config import DIURNAL_SNAPSHOTS
from pipeline.ingest_real_wait_times import _representative_hour_for_date, build_real_daily_avg_readings


def test_broadcasts_real_branch_day_average_to_every_active_service() -> None:
    ialc_frame = pd.DataFrame(
        [{"branch_id": "loja_de_cidadao_das_laranjeiras", "date": "2026-01-05", "avg_wait_minutes": 25.0, "total_attendances": 400}]
    )
    slc_frame = pd.DataFrame(
        [
            {"store_name": "Loja de Cidadão das Laranjeiras", "date": "2026-01-05", "service_type": "Atendimento Geral"},
            {"store_name": "Loja de Cidadão das Laranjeiras", "date": "2026-01-05", "service_type": "Passaporte - Pedido"},
        ]
    )

    readings = build_real_daily_avg_readings(ialc_frame, slc_frame)

    assert len(readings) == 2  # one per active service that day, not one per branch-day
    assert {r.desk_service_id for r in readings} == {"Atendimento Geral", "Passaporte - Pedido"}
    for r in readings:
        assert r.branch_id == "loja_de_cidadao_das_laranjeiras"
        assert r.estimated_wait_minutes == 25.0  # same real value broadcast to every service
        assert r.sample_size == 400
        assert r.source == "historical_real_daily_avg"
        assert r.is_open is True
        assert r.people_waiting is None


def test_uses_single_representative_timestamp_not_diurnal_expansion() -> None:
    ialc_frame = pd.DataFrame(
        [{"branch_id": "branch_a", "date": "2026-01-05", "avg_wait_minutes": 10.0, "total_attendances": 50}]
    )
    slc_frame = pd.DataFrame([{"store_name": "Branch A", "date": "2026-01-05", "service_type": "Atendimento Geral"}])

    readings = build_real_daily_avg_readings(ialc_frame, slc_frame)

    # Deliberately NOT expanded into one row per config.DIURNAL_SNAPSHOTS
    # entry like the proxy -- there's only one real number for the whole
    # day, so N identical copies would just multiply this row's weight
    # without adding information.
    assert len(readings) == 1
    expected_hour, expected_minute = _representative_hour_for_date(pd.Timestamp("2026-01-05"))
    assert readings[0].sampled_at.hour == expected_hour
    assert readings[0].sampled_at.minute == expected_minute


def test_representative_hour_rotates_across_candidate_hours_by_date() -> None:
    # Every individual day still gets exactly one honest real observation,
    # but different days must land on different candidate hours -- pinning
    # every day to the same fixed hour was the earlier design this
    # replaced, and it starved every other hour of real training signal.
    hours_seen = {_representative_hour_for_date(pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)) for i in range(len(DIURNAL_SNAPSHOTS))}
    assert len(hours_seen) == len(DIURNAL_SNAPSHOTS)  # a full rotation cycle hits every candidate hour once


def test_representative_hour_is_deterministic_for_the_same_date() -> None:
    date = pd.Timestamp("2026-03-14")
    assert _representative_hour_for_date(date) == _representative_hour_for_date(date)


def test_skips_branch_days_with_no_matching_slc_service_rows() -> None:
    # IALC-M has a branch-day SLC-M doesn't (e.g. a snapshot-timing gap) --
    # there's no way to know which services were active, so it must be
    # dropped rather than guessed.
    ialc_frame = pd.DataFrame(
        [{"branch_id": "branch_a", "date": "2026-01-05", "avg_wait_minutes": 10.0, "total_attendances": 50}]
    )
    slc_frame = pd.DataFrame([{"store_name": "Branch A", "date": "2026-02-01", "service_type": "Atendimento Geral"}])

    readings = build_real_daily_avg_readings(ialc_frame, slc_frame)

    assert readings == []
