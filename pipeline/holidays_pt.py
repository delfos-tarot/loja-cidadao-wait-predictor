"""Portuguese public holidays — national, movable, and per-municipality.

WHY THIS EXISTS, AND WHY IT IS NOT SHAPED THE WAY YOU'D EXPECT
--------------------------------------------------------------
The obvious design is an `is_holiday` model feature. That feature would be
useless here, and understanding why determines this module's whole shape.

Verified 2026-08-01 against the real IALC-M corpus: on a national holiday the
branch-day row is **absent from the dataset entirely** — not present with zero
attendances. All 15 weekday national holidays across 2024-2025 produce zero
rows. Closed days are a gap in the data, not a value in it.

Two consequences follow, and they point in opposite directions:

  1. `is_holiday` would be constant-0 across every training row, because rows
     on holidays do not exist. The model could never learn from it, and at
     inference `is_holiday=1` would be pure extrapolation off the edge of the
     training distribution — the same class of mistake that made far-future
     predictions erratic in `api/service.py`'s `estimate_people_waiting` (see
     CLAUDE.md). So holidays are wired into **`is_open`**, not into the
     regression features. That is also where they change what a citizen
     actually sees: today's Mon-Fri/9-17 heuristic will happily report a
     branch as open on 25 de Abril.

  2. The days *adjacent* to a holiday are fully present in the data, and they
     are where the real predictive signal lives. Measured on the same corpus,
     normalized per branch (stable branch-days only, >=30 attendances):

         bridge days ("pontes")   23.3 min vs 18.8 baseline   (+26%, n=1,769)
         first day after a break  23.0 min vs 18.8 baseline   (+24%, n=1,909)

     Both effects are larger than anything `rain_mm` has ever contributed, and
     unlike `rain_mm` they are available for every historical row without an
     external API. These become real features.

So: holidays suppress `is_open`; their neighbours become features.

CARNIVAL IS CONDITIONAL, AND THE DATA SAYS SO
---------------------------------------------
Carnival Tuesday is not a statutory national holiday — it is granted yearly as
`tolerância de ponto`, at the government's discretion. It behaves like a
holiday in 2024 and 2025 (zero rows) and in 2026 (a single branch reporting a
single attendance, against a 75-branch/198-attendance norm on the surrounding
days) — so it is treated as a holiday here, but flagged `statutory=False`
because a future year may genuinely differ. Do not hardcode the assumption
that it recurs; `verify_against_corpus()` re-checks it.

MUNICIPAL HOLIDAYS ARE DERIVED FROM DATA, NOT RECALLED
------------------------------------------------------
Every Portuguese municipality has its own `feriado municipal`, and these
branches are indexed by municipality, so this matters — but a half-remembered
table of 74 dates is worse than none: a wrong date excludes a normal busy day
and admits a real closure. Instead `pipeline/derive_municipal_holidays.py`
mines them from three years of IALC-M by their closure signature (this branch
absent, the rest of the country open) and writes `data/municipal_holidays.json`
with the supporting evidence. This module only *reads* that file.

The derivation independently reproduced known feriados municipais it was never
told about — Guarda 27/11, Leiria 22/05, Santarém 19/03, Coimbra 04/07,
Setúbal 15/09, Santiago do Cacém 25/07 — which is the check that it works.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MUNICIPAL_HOLIDAYS_PATH = "data/municipal_holidays.json"
MUNICIPAL_HOLIDAYS_SEED_PATH = "data/municipal_holidays_seed.json"

# Statutory national holidays with fixed calendar dates.
NATIONAL_FIXED: tuple[tuple[int, int, str], ...] = (
    (1, 1, "Ano Novo"),
    (4, 25, "Dia da Liberdade"),
    (5, 1, "Dia do Trabalhador"),
    (6, 10, "Dia de Portugal"),
    (8, 15, "Assunção de Nossa Senhora"),
    (10, 5, "Implantação da República"),
    (11, 1, "Todos os Santos"),
    (12, 1, "Restauração da Independência"),
    (12, 8, "Imaculada Conceição"),
    (12, 25, "Natal"),
)

# Movable feasts, as offsets in days from Easter Sunday. `statutory=False`
# marks Carnival's discretionary status — see the module docstring.
NATIONAL_MOVABLE: tuple[tuple[int, str, bool], ...] = (
    (-47, "Carnaval", False),
    (-2, "Sexta-feira Santa", True),
    (0, "Páscoa", True),
    (60, "Corpo de Deus", True),
)

# A holiday-adjacent working day counts as "post-holiday" only if the break it
# follows ended within this many days. Bounds the backward scan and keeps the
# flag meaning "first day back", rather than reaching across a long absence.
MAX_BREAK_LOOKBACK_DAYS = 5


def easter_sunday(year: int) -> dt.date:
    """Anonymous Gregorian algorithm (Meeus/Jones/Butcher).

    Implemented rather than taking a dependency: it is ~10 lines, exact for
    all Gregorian years, and every date it produces is verified against the
    real corpus by `verify_against_corpus()` — a dependency would be more
    code to audit, not less.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


@lru_cache(maxsize=None)
def national_holidays(year: int) -> dict[dt.date, str]:
    """Every national holiday in `year`, mapped to its Portuguese name."""
    holidays = {dt.date(year, month, day): name for month, day, name in NATIONAL_FIXED}
    easter = easter_sunday(year)
    for offset, name, _statutory in NATIONAL_MOVABLE:
        holidays[easter + dt.timedelta(days=offset)] = name
    return holidays


@lru_cache(maxsize=None)
def _national_holiday_dates(year: int) -> frozenset[dt.date]:
    return frozenset(national_holidays(year))


def is_national_holiday(date: dt.date) -> bool:
    return date in _national_holiday_dates(date.year)


def monday_after_nth_sunday(year: int, month: int, nth: int) -> dt.date:
    """The Monday following the `nth` Sunday of `month` — e.g. Gondomar's
    Nossa Senhora do Rosário (October, 1st) or Carregal do Sal's Nossa Senhora
    das Febres (July, 3rd)."""
    first = dt.date(year, month, 1)
    first_sunday = first + dt.timedelta(days=(6 - first.weekday()) % 7)
    return first_sunday + dt.timedelta(weeks=nth - 1, days=1)


def resolve_rule(rule: dict[str, object], year: int) -> dt.date | None:
    """Turns one municipal-holiday rule into a concrete date in `year`.

    Three kinds, all present among the real 74 municipalities:
      fixed                    — a calendar date (the majority)
      easter_offset            — days after Easter Sunday. Covers Segunda-feira
                                 de Páscoa (+1), Castelo Branco's Nossa Senhora
                                 de Mércoles (+9), Quinta-feira da Ascensão /
                                 Dia da Espiga (+39), and Águeda's Pentecostes
                                 (+50).
      monday_after_nth_sunday  — Monday following the nth Sunday of a month.

    The movable kinds are not an edge case worth skipping: **the two branches
    whose recalled fixed dates the corpus contradicted (Castelo Branco,
    Gondomar) both turned out to be movable.** Treating every municipality as
    fixed is precisely how a holiday table goes silently wrong.
    """
    kind = rule.get("kind", "fixed")
    if kind == "fixed":
        return dt.date(year, int(rule["month"]), int(rule["day"]))
    if kind == "easter_offset":
        return easter_sunday(year) + dt.timedelta(days=int(rule["offset"]))
    if kind == "monday_after_nth_sunday":
        return monday_after_nth_sunday(year, int(rule["month"]), int(rule["nth"]))
    logger.warning("Unknown municipal holiday rule kind %r — ignoring.", kind)
    return None


@lru_cache(maxsize=1)
def load_municipal_holidays(path: str = MUNICIPAL_HOLIDAYS_PATH) -> dict[str, dict[str, object]]:
    """Maps branch_id -> its municipal-holiday rule (see `resolve_rule`).

    Absent file or unlisted branch means "unknown", never "no holiday" — the
    caller sees the same graceful-degradation behaviour as a missing weather
    signal. `derive_municipal_holidays.py` writes the file; branches whose
    holiday could not be established are deliberately omitted rather than
    guessed at.
    """
    file_path = Path(path)
    if not file_path.exists():
        logger.warning("No municipal holiday file at %s — national holidays only.", path)
        return {}
    document = json.loads(file_path.read_text(encoding="utf-8"))
    return dict(document.get("branches", {}))


@lru_cache(maxsize=None)
def _municipal_dates_for_year(branch_id: str, year: int, path: str) -> frozenset[dt.date]:
    rule = load_municipal_holidays(path).get(str(branch_id))
    if rule is None:
        return frozenset()
    resolved = resolve_rule(rule, year)
    return frozenset() if resolved is None else frozenset({resolved})


def municipal_holiday_name(branch_id: str, date: dt.date, path: str = MUNICIPAL_HOLIDAYS_PATH) -> str | None:
    if date in _municipal_dates_for_year(str(branch_id), date.year, path):
        rule = load_municipal_holidays(path).get(str(branch_id), {})
        return str(rule.get("name") or "Feriado municipal")
    return None


def holiday_name(branch_id: str | None, date: dt.date) -> str | None:
    """The holiday falling on `date` for this branch, or None.

    `branch_id=None` checks national holidays only.
    """
    national = national_holidays(date.year).get(date)
    if national is not None:
        return national
    if branch_id is None:
        return None
    return municipal_holiday_name(branch_id, date)


def is_closed_for_holiday(branch_id: str | None, date: dt.date) -> bool:
    return holiday_name(branch_id, date) is not None


def _is_non_working(branch_id: str | None, date: dt.date) -> bool:
    return date.weekday() >= 5 or is_closed_for_holiday(branch_id, date)


def is_bridge_day(branch_id: str | None, date: dt.date) -> bool:
    """A working day wedged between two non-working stretches — a "ponte".

    The classic Portuguese cases are the Friday after a Thursday holiday and
    the Monday before a Tuesday holiday, but the general rule (both neighbours
    non-working) also catches a Wednesday between two holidays without needing
    a special case for it.
    """
    if _is_non_working(branch_id, date):
        return False
    previous_non_working = _is_non_working(branch_id, date - dt.timedelta(days=1))
    next_non_working = _is_non_working(branch_id, date + dt.timedelta(days=1))
    return previous_non_working and next_non_working


def is_post_holiday(branch_id: str | None, date: dt.date) -> bool:
    """First working day after a break that included at least one holiday.

    An ordinary Monday does not qualify — the weekend alone is the baseline
    that `day_of_week` already carries. Only a break containing a real holiday
    counts, which is what produces the measured pent-up demand.
    """
    if _is_non_working(branch_id, date):
        return False
    cursor = date - dt.timedelta(days=1)
    saw_holiday = False
    for _ in range(MAX_BREAK_LOOKBACK_DAYS):
        if not _is_non_working(branch_id, cursor):
            break
        if is_closed_for_holiday(branch_id, cursor):
            saw_holiday = True
        cursor -= dt.timedelta(days=1)
    return saw_holiday


# --------------------------------------------------------------------------
# Vectorized helpers (pandas) — used by feature engineering and the static build
# --------------------------------------------------------------------------

def _apply_per_branch(
    branch_ids: pd.Series, dates: pd.Series, function
) -> pd.Series:
    """Evaluates a scalar (branch, date) predicate over a frame.

    Deliberately evaluated on the *unique* (branch, date) pairs and merged
    back, not row by row: the training corpus expands each branch-day into
    several `DIURNAL_SNAPSHOTS` rows, so the naive path would recompute the
    identical calendar answer up to eight times per branch-day across ~7.8M
    rows.
    """
    pairs = pd.DataFrame({"_branch": branch_ids.astype(str).to_numpy(), "_date": dates.to_numpy()})
    unique_pairs = pairs.drop_duplicates()
    unique_pairs["_value"] = [
        function(branch, date) for branch, date in zip(unique_pairs["_branch"], unique_pairs["_date"])
    ]
    merged = pairs.merge(unique_pairs, on=["_branch", "_date"], how="left")
    return merged["_value"].astype(int).to_numpy()


def add_holiday_features(
    frame: pd.DataFrame,
    timestamp_column: str = "sampled_at",
    branch_column: str = "branch_id",
) -> pd.DataFrame:
    """Adds `is_bridge_day` and `is_post_holiday`.

    Note what is *not* added: `is_holiday`. Holiday rows do not exist in the
    training corpus, so the feature would be constant-0 and unlearnable — see
    the module docstring. Holidays reach the model through `is_open` instead.
    """
    frame = frame.copy()
    timestamps = pd.to_datetime(frame[timestamp_column], utc=True)
    dates = timestamps.dt.date

    if branch_column in frame.columns:
        branch_ids = frame[branch_column]
    else:
        branch_ids = pd.Series([None] * len(frame), index=frame.index)

    frame["is_bridge_day"] = _apply_per_branch(branch_ids, dates, is_bridge_day)
    frame["is_post_holiday"] = _apply_per_branch(branch_ids, dates, is_post_holiday)
    return frame


def holiday_closure_mask(
    frame: pd.DataFrame,
    date_column: str = "date",
    branch_column: str = "branch_id",
) -> pd.Series:
    """True where a row falls on a holiday for its own branch.

    Used by the static build to flag (rather than silently average) any
    branch-day that a holiday should have closed.

    A frame without a branch column degrades to national holidays only rather
    than raising — the same graceful-degradation rule the weather and demand
    baseline signals follow.
    """
    dates = pd.to_datetime(frame[date_column]).dt.date
    if branch_column in frame.columns:
        branch_ids = frame[branch_column]
    else:
        branch_ids = pd.Series([None] * len(frame), index=frame.index)
    return pd.Series(
        _apply_per_branch(branch_ids, dates, is_closed_for_holiday).astype(bool),
        index=frame.index,
    )


def verify_against_corpus(frame: pd.DataFrame, date_column: str = "date") -> dict[str, object]:
    """Re-checks the hardcoded national list against real data.

    A holiday the branches actually observed shows up as a near-total absence
    of rows. Anything this reports as `observed_open` is either a wrong date
    here or a genuine change in practice (Carnival is the live example) — in
    both cases something a maintainer needs to see rather than a silent
    mismatch. Called by `derive_municipal_holidays.py`; cheap enough to run
    from a test.
    """
    dates = pd.to_datetime(frame[date_column])
    rows_per_date = dates.dt.date.value_counts()
    typical = float(rows_per_date.median())
    years = sorted({d.year for d in rows_per_date.index})

    observed_closed: list[str] = []
    observed_open: list[str] = []
    for year in years:
        for date, name in sorted(national_holidays(year).items()):
            if date.weekday() >= 5 or not (dates.min().date() <= date <= dates.max().date()):
                continue
            # A stray row or two is not "open" — Carnival 2026 left exactly one
            # branch reporting one attendance against a 75-branch norm.
            if rows_per_date.get(date, 0) < 0.1 * typical:
                observed_closed.append(f"{date} {name}")
            else:
                observed_open.append(f"{date} {name} ({rows_per_date.get(date, 0)} rows)")
    return {
        "typical_rows_per_weekday": typical,
        "observed_closed": observed_closed,
        "observed_open": observed_open,
    }
