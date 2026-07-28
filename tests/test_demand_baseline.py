"""Unit tests for the demand-baseline aggregation and wait-time proxy formula
derived from real dados.gov.pt attendance counts (pipeline/demand_baseline.py).
No network calls, no trained model required.
"""

from __future__ import annotations

import json

import pandas as pd

from config import DEFAULT_SERVICE_AVG_MINUTES, DIURNAL_SNAPSHOTS, SERVICE_AVG_MINUTES
from pipeline.demand_baseline import (
    avg_service_minutes_for,
    build_demand_baseline_table,
    derive_proxy_readings,
    estimate_wait_minutes_from_attendances,
    load_calibrated_mu,
)
from pipeline.service_categories import GENERAL_OTHER, categorize


def test_estimate_wait_minutes_is_zero_for_no_attendances() -> None:
    assert estimate_wait_minutes_from_attendances(0, avg_service_minutes=10.0) == 0.0


def test_estimate_wait_minutes_increases_with_attendances() -> None:
    low = estimate_wait_minutes_from_attendances(5, avg_service_minutes=10.0, desks=3, operating_hours=8.0)
    high = estimate_wait_minutes_from_attendances(80, avg_service_minutes=10.0, desks=3, operating_hours=8.0)
    assert 0.0 <= low < high


def test_estimate_wait_minutes_caps_at_max_when_over_capacity() -> None:
    # Wildly over capacity: 10000 attendances against a tiny single-desk day.
    wait = estimate_wait_minutes_from_attendances(
        10000, avg_service_minutes=10.0, desks=1, operating_hours=8.0, max_wait_minutes=180.0
    )
    assert wait == 180.0


def test_estimate_wait_minutes_scales_with_operating_hours_window() -> None:
    # Same attendance count, but modeled as an hour-scale slice of demand
    # rather than a full day, must show far higher utilization/wait — this
    # is the exact bug the diurnal-expansion fix depends on getting right.
    same_count = 14.0
    daily_window_wait = estimate_wait_minutes_from_attendances(
        same_count, avg_service_minutes=10.0, desks=3, operating_hours=8.0
    )
    hourly_window_wait = estimate_wait_minutes_from_attendances(
        same_count, avg_service_minutes=10.0, desks=3, operating_hours=1.0
    )
    assert hourly_window_wait > daily_window_wait


def test_derive_proxy_readings_expands_into_diurnal_snapshots() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-05").date(),
                "branch_id": "branch_a",
                "service_type": "Tesouraria",
                "total_attendances": 80,
            }
        ]
    )

    readings = derive_proxy_readings(frame)

    # Asserted against config.DIURNAL_SNAPSHOTS itself (not a hardcoded
    # count/hour list) so this test doesn't need editing every time the
    # snapshot resolution changes.
    assert len(readings) == len(DIURNAL_SNAPSHOTS)
    hours = sorted(r.sampled_at.hour for r in readings)
    assert hours == sorted(hour for hour, _, _ in DIURNAL_SNAPSHOTS)
    # The highest-volume_factor snapshot must show a real, non-flat variance
    # against the lowest — this is the whole point of the expansion,
    # previously every snapshot would have been identical.
    waits_by_hour = {r.sampled_at.hour: r.estimated_wait_minutes for r in readings}
    peak_hour = max(DIURNAL_SNAPSHOTS, key=lambda s: s[2])[0]
    trough_hour = min(DIURNAL_SNAPSHOTS, key=lambda s: s[2])[0]
    assert waits_by_hour[peak_hour] > waits_by_hour[trough_hour]
    assert all(r.people_waiting is not None for r in readings)
    assert all(r.source == "historical_derived_proxy" for r in readings)


def test_build_demand_baseline_table_averages_by_day_of_week() -> None:
    # Two Mondays (2026-01-05 and 2026-01-12) and one Tuesday (2026-01-06).
    frame = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-05"), "branch_id": "branch_a", "service_type": "Tesouraria", "total_attendances": 10},
            {"date": pd.Timestamp("2026-01-12"), "branch_id": "branch_a", "service_type": "Tesouraria", "total_attendances": 20},
            {"date": pd.Timestamp("2026-01-06"), "branch_id": "branch_a", "service_type": "Tesouraria", "total_attendances": 100},
        ]
    )

    aggregated = build_demand_baseline_table(frame)

    monday_row = aggregated[(aggregated["branch_id"] == "branch_a") & (aggregated["day_of_week"] == 0)]
    tuesday_row = aggregated[(aggregated["branch_id"] == "branch_a") & (aggregated["day_of_week"] == 1)]

    assert monday_row["avg_attendances"].iloc[0] == 15.0  # mean of 10 and 20
    assert tuesday_row["avg_attendances"].iloc[0] == 100.0


def test_load_calibrated_mu_returns_none_when_file_missing(tmp_path) -> None:
    assert load_calibrated_mu(str(tmp_path / "does_not_exist.json")) is None


def test_load_calibrated_mu_reads_mu_by_category(tmp_path) -> None:
    fixture_path = tmp_path / "calibrated.json"
    fixture_path.write_text(json.dumps({"mu_by_category": {"IRN": 11.0, "GENERAL_OTHER": 7.0, "OTHER_SPECIALIZED": 21.0}}))

    calibrated = load_calibrated_mu(str(fixture_path))

    assert calibrated == {"IRN": 11.0, "GENERAL_OTHER": 7.0, "OTHER_SPECIALIZED": 21.0}


def test_avg_service_minutes_for_uses_calibrated_category_value_when_available() -> None:
    calibrated = {"IRN": 11.0, "GENERAL_OTHER": 7.0, "OTHER_SPECIALIZED": 21.0}

    # "Atendimento Geral" categorizes as GENERAL_OTHER -- must use that
    # category's calibrated value, not the old per-service constant.
    assert categorize("Atendimento Geral") == GENERAL_OTHER
    assert avg_service_minutes_for("Atendimento Geral", calibrated) == 7.0
    # A never-seen service label falls back to GENERAL_OTHER too, same as categorize().
    assert avg_service_minutes_for("Some Brand New Service", calibrated) == 7.0


def test_avg_service_minutes_for_falls_back_to_hardcoded_constants_when_uncalibrated() -> None:
    known_service = next(iter(SERVICE_AVG_MINUTES))
    assert avg_service_minutes_for(known_service, None) == SERVICE_AVG_MINUTES[known_service]
    assert avg_service_minutes_for("Some Brand New Service", None) == DEFAULT_SERVICE_AVG_MINUTES
