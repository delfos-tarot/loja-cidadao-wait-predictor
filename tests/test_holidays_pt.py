"""Tests for Portuguese holiday handling.

The high-value assertions here are the ones that pin down *design* decisions
rather than arithmetic — that `is_holiday` stays out of FEATURE_COLUMNS, and
that a real observed `is_open` still beats the holiday calendar. Both are
choices a future change could plausibly undo while every other test passes.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from pipeline.feature_engineering import (
    FEATURE_COLUMNS,
    add_is_open_feature,
    estimate_is_open_heuristic,
)
from pipeline.holidays_pt import (
    add_holiday_features,
    easter_sunday,
    holiday_closure_mask,
    is_bridge_day,
    is_closed_for_holiday,
    is_national_holiday,
    is_post_holiday,
    load_municipal_holidays,
    national_holidays,
    resolve_rule,
    verify_against_corpus,
)


@pytest.mark.parametrize(
    "year,expected",
    [
        (2024, dt.date(2024, 3, 31)),
        (2025, dt.date(2025, 4, 20)),
        (2026, dt.date(2026, 4, 5)),
        (2027, dt.date(2027, 3, 28)),
    ],
)
def test_easter_sunday_known_dates(year: int, expected: dt.date) -> None:
    assert easter_sunday(year) == expected


def test_movable_holidays_track_easter() -> None:
    holidays = national_holidays(2025)
    assert holidays[dt.date(2025, 4, 18)] == "Sexta-feira Santa"
    assert holidays[dt.date(2025, 6, 19)] == "Corpo de Deus"
    assert holidays[dt.date(2025, 3, 4)] == "Carnaval"


def test_fixed_national_holidays() -> None:
    assert is_national_holiday(dt.date(2026, 4, 25))
    assert is_national_holiday(dt.date(2026, 12, 25))
    assert not is_national_holiday(dt.date(2026, 4, 24))


@pytest.mark.parametrize(
    "rule,year,expected",
    [
        # Dates cross-checked against the published 2026/2027 calendar.
        ({"kind": "easter_offset", "offset": 1}, 2026, dt.date(2026, 4, 6)),    # Seg. de Páscoa
        ({"kind": "easter_offset", "offset": 1}, 2027, dt.date(2027, 3, 29)),
        ({"kind": "easter_offset", "offset": 9}, 2026, dt.date(2026, 4, 14)),   # N. Sra de Mércoles
        ({"kind": "easter_offset", "offset": 39}, 2026, dt.date(2026, 5, 14)),  # Ascensão / Espiga
        ({"kind": "easter_offset", "offset": 39}, 2027, dt.date(2027, 5, 6)),
        ({"kind": "easter_offset", "offset": 50}, 2026, dt.date(2026, 5, 25)),  # Pentecostes
        ({"kind": "monday_after_nth_sunday", "month": 10, "nth": 1}, 2026, dt.date(2026, 10, 5)),
        ({"kind": "monday_after_nth_sunday", "month": 7, "nth": 3}, 2026, dt.date(2026, 7, 20)),
        ({"kind": "monday_after_nth_sunday", "month": 8, "nth": 4}, 2027, dt.date(2027, 8, 23)),
        ({"kind": "fixed", "month": 6, "day": 24}, 2026, dt.date(2026, 6, 24)),
    ],
)
def test_resolve_rule_matches_published_calendar(rule, year, expected) -> None:
    assert resolve_rule(rule, year) == expected


def test_unknown_rule_kind_degrades_to_none_rather_than_raising() -> None:
    assert resolve_rule({"kind": "phase_of_the_moon"}, 2026) is None


def test_movable_municipal_holiday_moves_between_years() -> None:
    """Castelo Branco and Gondomar are movable; a fixed-date table would put
    them on the wrong day every year. Both were originally recalled as fixed
    dates and the corpus contradicted both — see the seed file."""
    ascensao = {"kind": "easter_offset", "offset": 39}
    assert resolve_rule(ascensao, 2026) != resolve_rule(ascensao, 2027).replace(year=2026)


def test_municipal_holiday_is_branch_specific() -> None:
    """Santo António closes Lisbon and leaves Porto open."""
    municipal = load_municipal_holidays()
    if "loja_de_cidadao_do_saldanha" not in municipal:
        pytest.skip("municipal holidays not derived in this checkout")
    santo_antonio = dt.date(2026, 6, 13)
    assert is_closed_for_holiday("loja_de_cidadao_do_saldanha", santo_antonio)
    assert not is_closed_for_holiday("loja_de_cidadao_do_porto", santo_antonio)


def test_unknown_branch_is_not_treated_as_holiday_free() -> None:
    """An unlisted branch falls back to national holidays, never to 'open'."""
    assert not is_closed_for_holiday("branch_that_does_not_exist", dt.date(2026, 6, 13))
    assert is_closed_for_holiday("branch_that_does_not_exist", dt.date(2026, 4, 25))


def test_bridge_day_friday_after_thursday_holiday() -> None:
    # 2026-06-10 (Dia de Portugal) is a Wednesday; Corpo de Deus 2026 is
    # Thursday 4 June, making Friday 5 June a textbook ponte.
    assert is_bridge_day(None, dt.date(2026, 6, 5))
    assert not is_bridge_day(None, dt.date(2026, 6, 12))


def test_holiday_itself_is_not_a_bridge_day_or_post_holiday() -> None:
    natal = dt.date(2026, 12, 25)
    assert not is_bridge_day(None, natal)
    assert not is_post_holiday(None, natal)


def test_post_holiday_requires_a_holiday_not_just_a_weekend() -> None:
    """An ordinary Monday must not qualify — day_of_week already carries that."""
    ordinary_monday = dt.date(2026, 3, 9)
    assert ordinary_monday.weekday() == 0
    assert not is_post_holiday(None, ordinary_monday)

    # Monday 27 April 2026 follows Saturday 25 April (Dia da Liberdade).
    after_holiday_weekend = dt.date(2026, 4, 27)
    assert is_post_holiday(None, after_holiday_weekend)


def test_is_holiday_is_deliberately_not_a_model_feature() -> None:
    """Holiday rows are absent from the corpus, so the feature would be
    constant-0 and unlearnable. Holidays act through is_open instead. If this
    fails, read pipeline/holidays_pt.py's docstring before 'fixing' it."""
    assert "is_holiday" not in FEATURE_COLUMNS
    assert "is_bridge_day" in FEATURE_COLUMNS
    assert "is_post_holiday" in FEATURE_COLUMNS


def test_heuristic_closes_branch_on_national_holiday() -> None:
    liberdade = dt.datetime(2026, 4, 25, 11, 0)
    ordinary = dt.datetime(2026, 4, 24, 11, 0)
    assert not estimate_is_open_heuristic(liberdade)
    assert estimate_is_open_heuristic(ordinary)


def test_heuristic_is_municipality_aware_only_with_branch_id() -> None:
    sao_joao = dt.datetime(2026, 6, 24, 11, 0)
    if "loja_de_cidadao_do_porto" not in load_municipal_holidays():
        pytest.skip("municipal holidays not derived in this checkout")
    assert not estimate_is_open_heuristic(sao_joao, "loja_de_cidadao_do_porto")
    assert estimate_is_open_heuristic(sao_joao, "loja_de_cidadao_do_saldanha")


def test_real_observed_is_open_overrides_the_holiday_calendar() -> None:
    """A measurement beats a calendar guess — SIGA reporting a desk open on a
    holiday must survive. Guards the ordering inside add_is_open_feature."""
    frame = pd.DataFrame(
        {
            "branch_id": ["loja_de_cidadao_do_porto", "loja_de_cidadao_do_porto"],
            "sampled_at": ["2026-04-25T11:00:00+00:00", "2026-04-25T11:00:00+00:00"],
            "is_open": [1, None],
        }
    )
    result = add_is_open_feature(frame)
    assert result["is_open"].tolist() == [1, 0]


def test_add_holiday_features_shapes_and_dtypes() -> None:
    frame = pd.DataFrame(
        {
            "branch_id": ["loja_de_cidadao_do_porto"] * 3,
            "sampled_at": [
                "2026-06-05T11:00:00+00:00",  # ponte
                "2026-06-11T11:00:00+00:00",  # day after Dia de Portugal
                "2026-03-10T11:00:00+00:00",  # ordinary Tuesday
            ],
        }
    )
    result = add_holiday_features(frame)
    assert result["is_bridge_day"].tolist() == [1, 0, 0]
    # The ponte is *also* a first-day-back: Corpo de Deus 2026 falls on
    # Thursday 4 June, so Friday the 5th genuinely satisfies both. The flags
    # overlap by design rather than partitioning — a Monday before a Tuesday
    # holiday is a bridge day and not a post-holiday one, so neither flag is
    # recoverable from the other.
    assert result["is_post_holiday"].tolist() == [1, 1, 0]
    assert result["is_bridge_day"].dtype.kind == "i"


def test_bridge_and_post_holiday_are_not_the_same_flag() -> None:
    # Monday 8 June 2026 precedes Dia de Portugal (Wednesday 10 June)? No —
    # the clean case is a Monday immediately before a Tuesday holiday, which
    # is a ponte with an ordinary weekend behind it.
    holidays = national_holidays(2026)
    tuesday_holiday = next(
        (d for d in holidays if d.weekday() == 1), None
    )
    if tuesday_holiday is None:
        pytest.skip("no Tuesday national holiday in 2026")
    monday_before = tuesday_holiday - dt.timedelta(days=1)
    assert is_bridge_day(None, monday_before)
    assert not is_post_holiday(None, monday_before)


def test_holiday_features_survive_a_frame_without_branch_id() -> None:
    """Graceful degradation: national holidays only, never an exception."""
    frame = pd.DataFrame({"sampled_at": ["2026-06-05T11:00:00+00:00"]})
    result = add_holiday_features(frame)
    assert result["is_bridge_day"].tolist() == [1]


def test_holiday_closure_mask_without_branch_column() -> None:
    frame = pd.DataFrame({"date": ["2026-04-25", "2026-04-24"]})
    assert holiday_closure_mask(frame).tolist() == [True, False]


def test_verify_against_corpus_flags_a_holiday_that_looks_open() -> None:
    """A date the branches actually traded through must be reported, not
    silently absorbed — this is how a stale holiday table gets noticed."""
    # Dia de Portugal 2026 is a Wednesday — a weekend holiday would be skipped
    # by the weekday filter and prove nothing.
    dates = pd.date_range("2026-06-05", "2026-06-15", freq="D")
    frame = pd.DataFrame({"date": [d for d in dates for _ in range(50)]})
    report = verify_against_corpus(frame)
    assert any("Dia de Portugal" in entry for entry in report["observed_open"])


def test_verify_against_corpus_confirms_a_genuinely_closed_holiday() -> None:
    """The inverse: a holiday with no rows must register as observed_closed,
    which is what the real corpus produces for all 33 of them."""
    dates = [d for d in pd.date_range("2026-06-05", "2026-06-15", freq="D") if d.date() != dt.date(2026, 6, 10)]
    frame = pd.DataFrame({"date": [d for d in dates for _ in range(50)]})
    report = verify_against_corpus(frame)
    assert any("Dia de Portugal" in entry for entry in report["observed_closed"])
    assert not report["observed_open"]
