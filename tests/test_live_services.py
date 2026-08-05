"""Tests for the live-only per-service view.

The load-bearing assertions are about what this view must NEVER do: publish a
wait in minutes, or count a closed desk as a walk-in. Both are choices a future
change could plausibly undo while every other test still passes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.backtest_live import DIMENSION_SETS, MIN_RELATIVE_GAIN, decide, evaluate_fold
from pipeline.build_live_services import GROUPING, build_corpus, build_payload


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["has_queue"] = ((frame["wait_time_minutes"] > 0) & (frame["people"] > 0)).astype(int)
    frame["coherent"] = (
        ((frame["wait_time_minutes"] > 0) & (frame["people"] > 0))
        | ((frame["wait_time_minutes"] == 0) & (frame["people"] == 0))
    ).astype(int)
    frame["date"] = pd.to_datetime(frame["local"]).dt.date
    return frame


def _rows(branch: str, service: str, n: int, people: int, wait: int) -> list[dict]:
    return [
        {"branch_id": branch, "desk_service_id": service, "people": people,
         "wait_time_minutes": wait, "local": f"2026-08-0{1 + i % 4}T10:00:00"}
        for i in range(n)
    ]


def test_payload_never_publishes_a_wait_in_minutes() -> None:
    """tempoRealEspera reports ~162 min against ONE person waiting, versus a
    measured 7.2 min service time — wrong by ~22x. No field of this payload may
    carry a duration derived from it."""
    frame = _frame(_rows("loja_de_cidadao_do_porto", "Atendimento Geral", 60, 2, 300))
    payload = build_payload(frame)
    for branch in payload.values():
        for record in branch["s"].values():
            assert "bw" not in record, "queue LENGTH only — never a wait in minutes"
            assert set(record) <= {"wi", "n", "ql", "qn"}


def test_walk_in_rate_requires_both_signals_to_agree() -> None:
    """A wait with nobody waiting is the documented pathology (26% of readings).
    It must not count as a queue."""
    frame = _frame(
        _rows("loja_de_cidadao_do_porto", "A", 50, 0, 300)  # wait but nobody there
    )
    payload = build_payload(frame)
    record = payload["loja_de_cidadao_do_porto"]["s"]["A"]
    assert record["wi"] == 100, "incoherent readings must not reduce the walk-in rate"


def test_thin_combos_are_dropped_rather_than_shown() -> None:
    frame = _frame(_rows("loja_de_cidadao_do_porto", "Rare", 5, 1, 10))
    assert build_payload(frame) == {}


def test_queue_length_omitted_when_too_few_queued_readings() -> None:
    rows = _rows("loja_de_cidadao_do_porto", "A", 50, 0, 0)
    rows += _rows("loja_de_cidadao_do_porto", "A", 3, 2, 30)
    payload = build_payload(_frame(rows))
    record = payload["loja_de_cidadao_do_porto"]["s"]["A"]
    assert "ql" not in record, "3 queued readings is not a median"
    assert record["wi"] > 0


def test_dimension_gate_requires_gain_on_both_targets() -> None:
    """A dimension that sharpens the walk-in guess while blurring queue length
    must not switch itself on — the real 'combo+hour' failure mode."""
    summary = pd.DataFrame(
        {"brier": [0.10, 0.05, 0.04], "busy_mae": [2.0, 1.5, 1.9]},
        index=["global", "combo", "combo+hour"],
    )
    keys, name = decide(summary)
    assert name == "combo"
    assert keys == GROUPING


def test_dimension_gate_ignores_gains_below_the_threshold() -> None:
    summary = pd.DataFrame(
        {"brier": [0.10, 0.10 * (1 - MIN_RELATIVE_GAIN / 2)],
         "busy_mae": [2.0, 2.0 * (1 - MIN_RELATIVE_GAIN / 2)]},
        index=["global", "combo"],
    )
    _, name = decide(summary)
    assert name == "global", "a sub-threshold gain is noise, not a dimension"


def test_corpus_reports_signal_coherence() -> None:
    """The page must be able to state how much of the corpus is self-consistent
    — 70.5% nationally, which is context a reader needs."""
    rows = _rows("loja_de_cidadao_do_porto", "A", 50, 0, 0)
    rows += _rows("loja_de_cidadao_do_porto", "A", 50, 0, 300)
    corpus = build_corpus(_frame(rows))
    assert corpus["coherent_pct"] == 50.0
    assert corpus["dimensions"] == GROUPING


def test_corpus_survives_a_corpus_with_no_queues() -> None:
    """A fresh install, or a quiet subset, has no queued readings at all. That
    must not crash the build, and must not report 'median 0 people' — which
    reads as a measurement rather than the absence of one."""
    corpus = build_corpus(_frame(_rows("loja_de_cidadao_do_porto", "A", 20, 0, 0)))
    assert corpus["queue_median_national"] is None
    assert corpus["walk_in_national"] == 100
