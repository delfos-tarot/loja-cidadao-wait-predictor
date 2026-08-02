"""Unit tests for the proxy-vs-live sample weighting logic in pipeline/train.py."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from pipeline.db import clean_siga_live_readings
from config import PROXY_LABEL_TRAINING_WEIGHT
from pipeline.train import (
    chronological_split_by_source,
    compute_sample_weights,
    drop_zero_weight_rows,
    evaluate_by_source,
)


class _ConstantModel:
    def __init__(self, constant: float) -> None:
        self.constant = constant

    def predict(self, X):
        return np.full(len(X), self.constant)


def _sampled_at_series(n: int, start: datetime, step: timedelta) -> list[datetime]:
    return [start + i * step for i in range(n)]


def test_chronological_split_by_source_gives_the_small_recent_source_real_train_exposure() -> None:
    # Mimics the real shape found 2026-07-27: historical_derived_proxy spans
    # ~2 years, siga_live is a tiny sliver of only the last few weeks. Under
    # one global chronological cutoff, every siga_live row would land in
    # test (it's always the most recent data) and none in train.
    now = datetime.now(timezone.utc)
    proxy_rows = pd.DataFrame(
        {
            "source": ["historical_derived_proxy"] * 1000,
            "sampled_at": _sampled_at_series(1000, now - timedelta(days=700), timedelta(hours=6)),
        }
    )
    live_rows = pd.DataFrame(
        {
            "source": ["siga_live"] * 20,
            "sampled_at": _sampled_at_series(20, now - timedelta(days=14), timedelta(hours=12)),
        }
    )
    frame = pd.concat([proxy_rows, live_rows], ignore_index=True)

    train_frame, test_frame = chronological_split_by_source(frame, test_fraction=0.2)

    live_in_train = (train_frame["source"] == "siga_live").sum()
    live_in_test = (test_frame["source"] == "siga_live").sum()
    assert live_in_train > 0, "siga_live got zero training rows -- the exact bug this split replaced"
    assert live_in_test > 0
    # Each source's own 80/20 split, not one global cutoff dominated by the
    # much larger proxy source.
    assert live_in_train == 16
    assert live_in_test == 4


def test_chronological_split_by_source_preserves_chronology_within_each_source() -> None:
    now = datetime.now(timezone.utc)
    frame = pd.concat(
        [
            pd.DataFrame({"source": ["siga_live"] * 10, "sampled_at": _sampled_at_series(10, now, timedelta(hours=1))}),
            pd.DataFrame(
                {"source": ["historical_derived_proxy"] * 10, "sampled_at": _sampled_at_series(10, now - timedelta(days=1), timedelta(hours=1))}
            ),
        ],
        ignore_index=True,
    )

    train_frame, test_frame = chronological_split_by_source(frame, test_fraction=0.2)

    for source in ("siga_live", "historical_derived_proxy"):
        train_times = train_frame.loc[train_frame["source"] == source, "sampled_at"]
        test_times = test_frame.loc[test_frame["source"] == source, "sampled_at"]
        assert train_times.max() < test_times.min(), f"{source} train/test rows are not chronologically ordered"


def test_live_rows_always_get_full_weight() -> None:
    frame = pd.DataFrame(
        [
            {"branch_id": "a", "desk_service_id": "svc", "source": "siga_live"},
            {"branch_id": "a", "desk_service_id": "svc", "source": "siga_live"},
        ]
    )
    weights = compute_sample_weights(frame, alpha=0.05, proxy_label_training_weight=1.0)
    assert (weights == 1.0).all()


def test_proxy_weight_is_full_when_no_live_data_exists_for_that_combo() -> None:
    frame = pd.DataFrame(
        [
            {"branch_id": "a", "desk_service_id": "svc", "source": "historical_derived_proxy"},
            {"branch_id": "b", "desk_service_id": "other_svc", "source": "siga_live"},  # unrelated combo
        ]
    )
    weights = compute_sample_weights(frame, alpha=0.05, proxy_label_training_weight=1.0)
    # branch a/svc has zero live samples of its own, so its proxy weight must be untouched (1.0),
    # regardless of live data existing for a completely different combo.
    assert weights[0] == 1.0


def test_proxy_weight_decays_as_that_combos_live_count_grows() -> None:
    rows = [{"branch_id": "a", "desk_service_id": "svc", "source": "historical_derived_proxy"}]
    rows += [{"branch_id": "a", "desk_service_id": "svc", "source": "siga_live"} for _ in range(20)]
    frame = pd.DataFrame(rows)

    weights = compute_sample_weights(frame, alpha=0.05, proxy_label_training_weight=1.0)
    proxy_weight = weights[0]

    # 1 / (1 + 0.05 * 20) = 1 / 2 = 0.5
    assert abs(proxy_weight - 0.5) < 1e-9


def test_proxy_weight_decays_from_live_rows_filed_under_a_siga_specific_name(monkeypatch) -> None:
    # The point of the service crosswalk (pipeline/reconcile_siga_services.py):
    # real coverage recorded as "Câmara - Atendimento Geral" must still decay
    # the proxy rows filed under the dados.gov.pt name "Atendimento Geral".
    # Without it these are different join keys and the proxy never decays.
    monkeypatch.setattr(
        "pipeline.train.load_service_crosswalk",
        lambda: {"Câmara - Atendimento Geral": "Atendimento Geral"},
    )
    rows = [{"branch_id": "a", "desk_service_id": "Atendimento Geral", "source": "historical_derived_proxy"}]
    rows += [{"branch_id": "a", "desk_service_id": "Câmara - Atendimento Geral", "source": "siga_live"} for _ in range(20)]
    frame = pd.DataFrame(rows)

    weights = compute_sample_weights(frame, alpha=0.05, proxy_label_training_weight=1.0)

    # Same 1 / (1 + 0.05 * 20) = 0.5 as if the names had matched exactly.
    assert abs(weights[0] - 0.5) < 1e-9


def test_raising_alpha_cannot_affect_a_combo_with_no_live_coverage() -> None:
    # The safety property behind raising PROXY_WEIGHT_DECAY_ALPHA 0.05 -> 0.5
    # on 2026-07-31: weight is 1/(1 + alpha*live_count), so live_count=0
    # gives exactly 1.0 for ANY alpha. This is what makes the change a
    # per-combo adjustment rather than a global down-weighting -- the ~1,500
    # combos whose only signal is the proxy tier must be untouched.
    frame = pd.DataFrame([{"branch_id": "no_live", "desk_service_id": "svc", "source": "historical_derived_proxy"}])

    for alpha in [0.05, 0.5, 5.0, 500.0]:
        assert compute_sample_weights(frame, alpha=alpha, proxy_label_training_weight=1.0)[0] == 1.0


def test_higher_alpha_decays_a_covered_combos_proxy_faster() -> None:
    rows = [{"branch_id": "a", "desk_service_id": "svc", "source": "historical_derived_proxy"}]
    rows += [{"branch_id": "a", "desk_service_id": "svc", "source": "siga_live"} for _ in range(30)]
    frame = pd.DataFrame(rows)

    weak = compute_sample_weights(frame, alpha=0.05, proxy_label_training_weight=1.0)[0]
    strong = compute_sample_weights(frame, alpha=0.5, proxy_label_training_weight=1.0)[0]

    assert strong < weak
    assert abs(weak - 1 / (1 + 0.05 * 30)) < 1e-9
    assert abs(strong - 1 / (1 + 0.5 * 30)) < 1e-9


def test_proxy_weight_never_affected_by_other_combos_live_counts() -> None:
    frame = pd.DataFrame(
        [
            {"branch_id": "a", "desk_service_id": "svc", "source": "historical_derived_proxy"},
            *[{"branch_id": "b", "desk_service_id": "other_svc", "source": "siga_live"} for _ in range(500)],
        ]
    )
    weights = compute_sample_weights(frame, alpha=0.05, proxy_label_training_weight=1.0)
    # branch a/svc's proxy row must stay at full weight no matter how much
    # live data branch b/other_svc has accumulated — this is the whole point
    # of per-combo (not global) weighting.
    assert weights[0] == 1.0


def test_real_daily_avg_weight_scales_with_its_own_sample_size() -> None:
    frame = pd.DataFrame(
        [
            {"branch_id": "a", "desk_service_id": "svc", "source": "historical_real_daily_avg", "sample_size": 30},
            {"branch_id": "a", "desk_service_id": "svc", "source": "historical_real_daily_avg", "sample_size": 1},
        ]
    )
    weights = compute_sample_weights(frame, alpha=0.05, real_avg_reference_sample_size=30)

    # 30 / (30 + 30) = 0.5 -- a "meaningfully sized" day gets a real, non-trivial weight.
    assert abs(weights[0] - 0.5) < 1e-9
    # 1 / (1 + 30) ~= 0.032 -- an n=1 day (the Freixo de Espada a Cinta case
    # found in the real data) must be weighted far lower, not treated as equally reliable.
    assert weights[1] < 0.05
    assert weights[1] < weights[0]


def test_real_daily_avg_weight_is_not_decayed_by_live_count() -> None:
    # Unlike historical_derived_proxy, historical_real_daily_avg's weight
    # reflects how much *that specific row* can be trusted (its own sample
    # size), not how much live siga_live coverage exists for the combo --
    # there's no "combo" for a branch-level daily average to decay against.
    rows = [{"branch_id": "a", "desk_service_id": "svc", "source": "historical_real_daily_avg", "sample_size": 100}]
    rows += [{"branch_id": "a", "desk_service_id": "svc", "source": "siga_live", "sample_size": None} for _ in range(50)]
    frame = pd.DataFrame(rows)

    weights = compute_sample_weights(frame, alpha=0.05, real_avg_reference_sample_size=30)

    # 100 / (100 + 30) regardless of the 50 live rows for the same combo.
    assert abs(weights[0] - (100 / 130)) < 1e-9


def test_evaluate_by_source_computes_r2_when_target_varies() -> None:
    test_frame = pd.DataFrame(
        {"wait_time_minutes": [0.0, 5.0, 10.0, 15.0], "source": ["historical_derived_proxy"] * 4}
    )
    model = _ConstantModel(7.5)  # roughly the mean, so R^2 should be defined and computable

    results = evaluate_by_source(model, test_frame, feature_columns=[], target_column="wait_time_minutes")

    assert results["historical_derived_proxy"]["n"] == 4
    assert results["historical_derived_proxy"]["r2"] is not None
    assert results["historical_derived_proxy"]["note"] is None


def test_evaluate_by_source_flags_constant_target_as_undefined() -> None:
    # Exactly the real-world case that motivated this: a handful of siga_live
    # rows all showing the same (e.g. zero) actual wait — R^2 is mathematically
    # undefined here and must be flagged, not silently reported as some number.
    test_frame = pd.DataFrame({"wait_time_minutes": [0.0, 0.0, 0.0], "source": ["siga_live"] * 3})
    model = _ConstantModel(0.5)

    results = evaluate_by_source(model, test_frame, feature_columns=[], target_column="wait_time_minutes")

    assert results["siga_live"]["r2"] is None
    assert results["siga_live"]["note"] is not None
    assert results["siga_live"]["mae"] == 0.5


def test_evaluate_by_source_flags_too_few_rows() -> None:
    test_frame = pd.DataFrame({"wait_time_minutes": [3.0], "source": ["siga_live"]})
    model = _ConstantModel(3.0)

    results = evaluate_by_source(model, test_frame, feature_columns=[], target_column="wait_time_minutes")

    assert results["siga_live"]["r2"] is None
    assert "too few rows" in results["siga_live"]["note"]


def test_evaluate_by_source_segments_multiple_sources_independently() -> None:
    test_frame = pd.DataFrame(
        {
            "wait_time_minutes": [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
            "source": ["historical_derived_proxy"] * 4 + ["siga_live"] * 2,
        }
    )
    model = _ConstantModel(12.5)

    results = evaluate_by_source(model, test_frame, feature_columns=[], target_column="wait_time_minutes")

    assert set(results.keys()) == {"historical_derived_proxy", "siga_live"}
    assert results["historical_derived_proxy"]["n"] == 4
    assert results["siga_live"]["n"] == 2


def _live_row(branch_id: str, service: str, sampled_at: datetime, wait: float) -> dict:
    return {"branch_id": branch_id, "desk_service_id": service, "sampled_at": sampled_at, "wait_time_minutes": wait, "source": "siga_live"}


def test_clean_siga_live_readings_drops_frozen_high_wait_repeats() -> None:
    # Found 2026-07-30: a value identical to the prior poll despite ~20 real
    # minutes elapsed is a stuck/stale reading, not a genuine observation --
    # dropping it costs nothing since it adds no information beyond the
    # already-kept prior row.
    now = datetime.now(timezone.utc)
    frame = pd.DataFrame(
        [
            _live_row("b1", "s1", now, 450.0),
            _live_row("b1", "s1", now + timedelta(minutes=20), 450.0),  # frozen repeat
        ]
    )

    cleaned = clean_siga_live_readings(frame)

    assert cleaned["wait_time_minutes"].iloc[0] == 450.0
    assert pd.isna(cleaned["wait_time_minutes"].iloc[1]), "frozen repeat must become NaN so it gets dropped downstream"


def test_clean_siga_live_readings_leaves_low_value_repeats_alone() -> None:
    # A branch legitimately sitting at 0 min wait across consecutive polls is
    # a real, mundane pattern -- not the pathology this targets. Restricting
    # the frozen check to >= SIGA_STALENESS_CHECK_MIN_WAIT_MINUTES avoids
    # discarding genuine quiet periods.
    now = datetime.now(timezone.utc)
    frame = pd.DataFrame(
        [
            _live_row("b1", "s1", now, 0.0),
            _live_row("b1", "s1", now + timedelta(minutes=15), 0.0),
        ]
    )

    cleaned = clean_siga_live_readings(frame)

    assert cleaned["wait_time_minutes"].tolist() == [0.0, 0.0]


def test_clean_siga_live_readings_clamps_erratic_swings_instead_of_dropping() -> None:
    # Found 2026-07-30: deltas up to +-18,000 min between polls 20-40 min
    # apart -- physically impossible. Clamped to a plausible bound around
    # the prior reading rather than dropped, so a corrected point survives.
    now = datetime.now(timezone.utc)
    frame = pd.DataFrame(
        [
            _live_row("b1", "s1", now, 10.0),
            _live_row("b1", "s1", now + timedelta(minutes=20), 5000.0),  # impossible jump
        ]
    )

    cleaned = clean_siga_live_readings(frame)

    second = cleaned["wait_time_minutes"].iloc[1]
    assert second is not None and not pd.isna(second), "erratic row must be clamped, not dropped"
    assert second < 5000.0
    assert second == 10.0 + 20 * 10.0  # prev_wait + gap_minutes * max_change_per_minute


def test_clean_siga_live_readings_leaves_plausible_changes_untouched() -> None:
    now = datetime.now(timezone.utc)
    frame = pd.DataFrame(
        [
            _live_row("b1", "s1", now, 50.0),
            _live_row("b1", "s1", now + timedelta(minutes=15), 40.0),  # plausible drop
        ]
    )

    cleaned = clean_siga_live_readings(frame)

    assert cleaned["wait_time_minutes"].tolist() == [50.0, 40.0]


def test_clean_siga_live_readings_ignores_gaps_too_large_to_compare() -> None:
    # A multi-hour gap (e.g. overnight or an outage) says nothing about
    # staleness or rate of change -- must not be compared at all.
    now = datetime.now(timezone.utc)
    frame = pd.DataFrame(
        [
            _live_row("b1", "s1", now, 450.0),
            _live_row("b1", "s1", now + timedelta(hours=12), 450.0),  # same value, but not comparable
        ]
    )

    cleaned = clean_siga_live_readings(frame)

    assert cleaned["wait_time_minutes"].tolist() == [450.0, 450.0]


def test_clean_siga_live_readings_never_touches_other_sources() -> None:
    now = datetime.now(timezone.utc)
    frame = pd.DataFrame(
        [
            {
                "branch_id": "b1",
                "desk_service_id": "s1",
                "sampled_at": now,
                "wait_time_minutes": 450.0,
                "source": "historical_derived_proxy",
            },
            {
                "branch_id": "b1",
                "desk_service_id": "s1",
                "sampled_at": now + timedelta(minutes=20),
                "wait_time_minutes": 450.0,
                "source": "historical_derived_proxy",
            },
        ]
    )

    cleaned = clean_siga_live_readings(frame)

    assert cleaned["wait_time_minutes"].tolist() == [450.0, 450.0]


def test_clean_siga_live_readings_first_reading_for_a_combo_is_untouched() -> None:
    frame = pd.DataFrame([_live_row("b1", "s1", datetime.now(timezone.utc), 450.0)])

    cleaned = clean_siga_live_readings(frame)

    assert cleaned["wait_time_minutes"].iloc[0] == 450.0


# ---------------------------------------------------------------------------
# Proxy-label retirement (2026-08-01) — see config.PROXY_LABEL_TRAINING_WEIGHT
# ---------------------------------------------------------------------------

def test_proxy_labels_are_retired_by_default() -> None:
    """The shipped default excludes formula-derived labels from training.

    An ablation measured this as an improvement on real data (MAE 7.955 ->
    7.804, R^2 0.484 -> 0.545, scored on real rows only). If this flips back
    silently, the model is being trained on ~6.9M manufactured labels again --
    including DIURNAL_SNAPSHOTS' hand-drawn hour curve that live data
    contradicts.
    """
    assert PROXY_LABEL_TRAINING_WEIGHT == 0.0

    frame = pd.DataFrame(
        [
            {"branch_id": "a", "desk_service_id": "svc", "source": "historical_derived_proxy"},
            {"branch_id": "a", "desk_service_id": "svc", "source": "siga_live"},
        ]
    )
    weights = compute_sample_weights(frame, alpha=0.05)
    assert weights[0] == 0.0, "proxy row should carry no training weight"
    assert weights[1] == 1.0, "live row must be unaffected by the retirement"


def test_retirement_leaves_the_decay_mechanism_intact_for_restore() -> None:
    """Retiring is a multiplier, not a deletion. Setting the constant back to
    1.0 must reproduce the previous behaviour exactly, or 'recoverable' is a
    claim we cannot honour."""
    rows = [{"branch_id": "a", "desk_service_id": "svc", "source": "historical_derived_proxy"}]
    rows += [{"branch_id": "a", "desk_service_id": "svc", "source": "siga_live"} for _ in range(20)]
    frame = pd.DataFrame(rows)

    restored = compute_sample_weights(frame, alpha=0.05, proxy_label_training_weight=1.0)
    assert abs(restored[0] - 0.5) < 1e-9  # 1 / (1 + 0.05 * 20)


def test_real_tiers_are_never_dropped_by_the_zero_weight_filter() -> None:
    """The filter must remove only what carries zero weight. A real row with a
    tiny sample_size gets a SMALL weight, never zero -- dropping those would
    silently discard genuine measurements."""
    frame = pd.DataFrame(
        [
            {"branch_id": "a", "desk_service_id": "svc", "source": "historical_derived_proxy", "sample_size": 0},
            {"branch_id": "a", "desk_service_id": "svc", "source": "historical_real_daily_avg", "sample_size": 1},
            {"branch_id": "a", "desk_service_id": "svc", "source": "siga_live", "sample_size": 0},
        ]
    )
    weights = compute_sample_weights(frame, alpha=0.05)
    kept, kept_weights = drop_zero_weight_rows(frame, weights)

    assert set(kept["source"]) == {"historical_real_daily_avg", "siga_live"}
    assert (kept_weights > 0).all()
    assert len(kept) == 2
