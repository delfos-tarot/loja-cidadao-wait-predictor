"""Tests for holiday handling in the static site aggregation.

Focused on the two things a citizen-visible regression would look like: a
"typical Tuesday" number quietly inflated by pontes, and a municipal holiday
guessed for a branch whose holiday was never established.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from config import BRANCHES_BY_ID
from pipeline.build_static import annotate_holidays, typical_days
from pipeline.holidays_pt import load_municipal_holidays

BRANCH = "loja_de_cidadao_do_porto"


def _frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(d) for d, _ in rows],
            "branch_id": [BRANCH] * len(rows),
            "avg_wait_minutes": [w for _, w in rows],
            "day_of_week": [pd.Timestamp(d).dayofweek for d, _ in rows],
        }
    )


@pytest.mark.skipif(BRANCH not in BRANCHES_BY_ID, reason="real branch registry not present")
def test_typical_days_excludes_pontes_and_first_days_back() -> None:
    frame = annotate_holidays(
        _frame(
            [
                ("2026-06-05", 60.0),  # ponte after Corpo de Deus (Thu 4 June)
                ("2026-06-11", 55.0),  # first day back after Dia de Portugal
                ("2026-03-10", 20.0),  # ordinary Tuesday
                ("2026-03-17", 22.0),  # ordinary Tuesday
            ]
        )
    )
    typical = typical_days(frame)
    assert sorted(typical["avg_wait_minutes"]) == [20.0, 22.0]
    # The inflated days are excluded from the mean, not deleted from the frame
    # -- the corpus summary still counts them as real measurements.
    assert len(frame) == 4


@pytest.mark.skipif(BRANCH not in BRANCHES_BY_ID, reason="real branch registry not present")
def test_municipal_holiday_closes_only_its_own_branch() -> None:
    if BRANCH not in load_municipal_holidays():
        pytest.skip("municipal holidays not derived in this checkout")
    sao_joao = _frame([("2026-06-24", 30.0)])
    assert annotate_holidays(sao_joao)["is_holiday_closure"].tolist() == [True]

    elsewhere = sao_joao.assign(branch_id="loja_de_cidadao_do_saldanha")
    assert annotate_holidays(elsewhere)["is_holiday_closure"].tolist() == [False]


def test_undetermined_branches_are_omitted_rather_than_guessed() -> None:
    """A branch whose holiday could be established by neither route must be
    absent from the table entirely. Defaulting it to any date would silently
    exclude a real trading day from that branch's averages."""
    municipal = load_municipal_holidays()
    if not municipal:
        pytest.skip("municipal holidays not derived in this checkout")
    assert set(municipal) <= set(BRANCHES_BY_ID)
    # Every entry must declare where it came from, so a reviewer can tell a
    # corpus-proven date from a published one the corpus could not check.
    assert all("derived_from" in entry for entry in municipal.values())


def test_intermittent_branches_are_never_data_derived() -> None:
    """Mobile and intermittently-open branches must not reach the table via the
    closure-signature route -- their absence carries no information, and
    Palmela Móvel alone matched ~180 spurious month-days before the presence
    filter. The published list MAY still cover them (its evidence is external),
    so the invariant is about provenance, not membership."""
    municipal = load_municipal_holidays()
    entry = municipal.get("loja_de_cidadao_de_palmela_movel")
    if entry is None:
        return  # absent entirely is also correct
    assert entry["derived_from"] != "IALC-M closure signature"


def test_holiday_closures_are_absent_from_the_real_corpus() -> None:
    """Load-bearing assumption for the whole design: a closed branch-day is a
    missing row, so `is_holiday` can never be a model feature. If this ever
    fails, pipeline/holidays_pt.py's central premise needs revisiting."""
    try:
        from pipeline.build_static import load_stable_branch_days

        frame = annotate_holidays(load_stable_branch_days())
    except FileNotFoundError:
        pytest.skip("IALC-M baseline not built in this checkout")
    assert int(frame["is_holiday_closure"].sum()) == 0
