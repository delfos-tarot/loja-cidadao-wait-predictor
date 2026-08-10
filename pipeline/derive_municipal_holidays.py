"""Derives each branch's `feriado municipal` from its closure signature in IALC-M.

Every Portuguese municipality has its own municipal holiday, and these branches
are indexed by municipality — so a Lisbon branch closes on 13 June and a Porto
one on 24 June. Encoding that from memory was rejected: a table of 74 half-
recalled dates fails in the worst direction, since a wrong date both excludes a
normal trading day and admits a real closure, and nothing downstream would ever
flag it.

The corpus can answer the question directly instead. Verified 2026-08-01: a
closed branch-day is *absent* from IALC-M rather than present with zero
attendances, so a municipal holiday has a clean, searchable signature:

    this branch is missing, while the rest of the country is open

Run:
    python -m pipeline.derive_municipal_holidays
    python -m pipeline.derive_municipal_holidays --min-years 3   # stricter

Output: data/municipal_holidays.json, carrying the evidence behind every entry
(which years it was observed, how many branches were open that day) so a
reviewer can audit a date rather than trust it.

WHY THE FILTERS BELOW ARE NOT OPTIONAL
--------------------------------------
Run naively, the top of the results is dominated by two false-positive classes
that have nothing to do with holidays:

  - `loja_de_cidadao_de_palmela_movel` is a **mobile unit**. It is absent most
    weekdays by design, so it "matches" ~180 distinct month-days — every one a
    false positive. `MIN_WEEKDAY_PRESENCE_RATE` excludes branches that are not
    reliably open in the first place, which is the only reason absence carries
    information for the rest.
  - Freixo de Espada à Cinta reports intermittently for a different reason
    (a small branch with sparse operation), producing ~25 spurious dates. The
    same filter catches it.

The remaining filters encode what makes an absence *municipal* rather than
national or accidental: it recurs on the same calendar date across years
(`--min-years`), and the rest of the network is open that day
(`MIN_NATIONAL_OPEN_RATE`) — otherwise it is a national holiday, or a data
outage, and belongs in neither table.

CONFIDENCE THAT THIS WORKS
--------------------------
The derivation reproduced feriados municipais it was never given, at dates
independently known to be correct: Guarda 27/11, Leiria 22/05, Santarém 19/03,
Coimbra 04/07, Setúbal 15/09, Batalha 14/08, Torres Vedras 11/11, and — the
nicest confirmation, because the name and date agree on a fact nobody encoded —
Santiago do Cacém on 25/07, the feast of Santiago.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from config import BRANCHES_BY_ID
from pipeline.holidays_pt import (
    MUNICIPAL_HOLIDAYS_PATH,
    MUNICIPAL_HOLIDAYS_SEED_PATH,
    is_national_holiday,
    resolve_rule,
    verify_against_corpus,
)

logger = logging.getLogger(__name__)

IALC_BASELINE_PATH = "data/cleaned_ialc_baseline.parquet"

# A branch must actually be open most weekdays for its absence to mean
# anything. Mobile units and intermittently-staffed branches sit far below
# this; every ordinary branch sits far above it, so the threshold is not
# finely tuned and does not need to be.
MIN_WEEKDAY_PRESENCE_RATE = 0.85

# Share of *other* active branches that must be reporting that day, for the
# absence to be branch-specific rather than national.
MIN_NATIONAL_OPEN_RATE = 0.70

# Same calendar date must recur across at least this many distinct years.
DEFAULT_MIN_YEARS = 2


def load_weekday_frame(path: str = IALC_BASELINE_PATH) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame[frame["date"].dt.dayofweek < 5].copy()


def branch_active_spans(frame: pd.DataFrame) -> dict[str, tuple[dt.date, dt.date]]:
    """First and last date each branch appears.

    A branch that opened in 2025 is not "absent" for all of 2024; without this
    every pre-opening date would score as a closure.
    """
    spans = frame.groupby("branch_id")["date"].agg(["min", "max"])
    return {str(b): (row["min"].date(), row["max"].date()) for b, row in spans.iterrows()}


def regular_branches(frame: pd.DataFrame, spans: dict[str, tuple[dt.date, dt.date]]) -> set[str]:
    """Branches open on at least MIN_WEEKDAY_PRESENCE_RATE of their own span."""
    all_dates = sorted({d.date() for d in frame["date"].unique()})
    observed = frame.groupby("branch_id")["date"].apply(lambda s: {d.date() for d in s})

    regular: set[str] = set()
    for branch_id, (start, end) in spans.items():
        eligible = [
            d for d in all_dates
            if start <= d <= end and not is_national_holiday(d)
        ]
        if not eligible:
            continue
        seen = observed.get(branch_id, set())
        rate = sum(1 for d in eligible if d in seen) / len(eligible)
        if rate >= MIN_WEEKDAY_PRESENCE_RATE:
            regular.add(str(branch_id))
        else:
            logger.info("Excluding %s from derivation: open only %.0f%% of weekdays", branch_id, 100 * rate)
    return regular


def derive(frame: pd.DataFrame, min_years: int = DEFAULT_MIN_YEARS) -> dict[str, dict[str, Any]]:
    spans = branch_active_spans(frame)
    regular = regular_branches(frame, spans)
    present_by_date = frame.groupby(frame["date"].dt.date)["branch_id"].apply(lambda s: {str(x) for x in s})

    # (branch, month, day) -> years in which the branch was absent nationally-open
    absences: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    open_rates: dict[tuple[str, int, int], list[float]] = defaultdict(list)

    for date, present in present_by_date.items():
        if is_national_holiday(date):
            continue
        active = {b for b in regular if spans[b][0] <= date <= spans[b][1]}
        if not active:
            continue
        open_rate = len(active & present) / len(active)
        if open_rate < MIN_NATIONAL_OPEN_RATE:
            continue  # network-wide outage or unlisted national closure
        for branch_id in active - present:
            key = (branch_id, date.month, date.day)
            absences[key].add(date.year)
            open_rates[key].append(open_rate)

    # A branch may show several recurring absences; keep the strongest one,
    # since a municipality has exactly one feriado municipal.
    best: dict[str, dict[str, Any]] = {}
    for (branch_id, month, day), years in absences.items():
        if len(years) < min_years:
            continue
        candidate = {
            "month": month,
            "day": day,
            "years_observed": sorted(years),
            "municipality": getattr(BRANCHES_BY_ID.get(branch_id), "municipality", None),
            "mean_national_open_rate": round(
                sum(open_rates[(branch_id, month, day)]) / len(open_rates[(branch_id, month, day)]), 3
            ),
            "derived_from": "IALC-M closure signature",
        }
        incumbent = best.get(branch_id)
        if incumbent is None or (
            len(candidate["years_observed"]),
            candidate["mean_national_open_rate"],
        ) > (len(incumbent["years_observed"]), incumbent["mean_national_open_rate"]):
            best[branch_id] = candidate
    return best


def load_seed(path: str) -> dict[str, dict[str, Any]]:
    """Published feriados municipais, keyed by municipality.

    A seed is a *candidate*, never an answer — see `reconcile`.
    """
    file_path = Path(path)
    if not file_path.exists():
        logger.info("No seed list at %s — deriving from data alone.", path)
        return {}
    document = json.loads(file_path.read_text(encoding="utf-8"))
    return {
        "municipalities": dict(document.get("municipalities", {})),
        "branch_overrides": dict(document.get("branch_overrides", {})),
    }


def observed_status(
    branch_id: str,
    rule: dict[str, Any],
    present_by_date: "pd.Series",
    spans: dict[str, tuple[dt.date, dt.date]],
) -> tuple[int, int, int]:
    """How the corpus behaved on this branch's seeded holiday.

    Returns (closed_years, open_years, unobservable_years). "Unobservable"
    covers weekends and dates outside the branch's active span — the reason
    Faro and Viseu could not be derived despite the seed being correct: their
    holiday fell on a Saturday or Sunday in two of the three corpus years.
    """
    closed = opened = unobservable = 0
    for year in sorted({d.year for d in present_by_date.index}):
        date = resolve_rule(rule, year)
        if date is None:
            continue
        start, end = spans.get(branch_id, (None, None))
        if (
            date.weekday() >= 5
            or is_national_holiday(date)
            or start is None
            or not (start <= date <= end)
            or date not in present_by_date.index
        ):
            unobservable += 1
            continue
        if branch_id in present_by_date[date]:
            opened += 1
        else:
            closed += 1
    return closed, opened, unobservable


def reconcile(
    derived: dict[str, dict[str, Any]],
    seed: dict[str, dict[str, Any]],
    present_by_date: "pd.Series",
    spans: dict[str, tuple[dt.date, dt.date]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Merges the published list into the data-derived table. **The data wins.**

    Four outcomes per branch, all reported rather than collapsed into a count:

      confirmed   seed and corpus agree — the strongest entries
      added       corpus never had an observable occurrence (holiday fell on a
                  weekend every year, or predates the branch's coverage), so the
                  seed supplies what the data could not. This is the case the
                  seed exists for: Faro, Viseu, Sintra, Seixal.
      contradicted  corpus shows the branch OPEN on the seeded date, in a year
                  it could have been closed. The seed is rejected and the
                  derived value (if any) is kept. Municipal holidays are
                  *optional* under the Código do Trabalho — a municipality
                  having one does not oblige this branch to close for it, so a
                  contradiction is a real signal about this branch, not
                  necessarily an error in the list.
      conflict    both resolve, to different dates. Derived wins; logged loudly.
    """
    outcomes: dict[str, list[str]] = {"confirmed": [], "added": [], "contradicted": [], "conflict": []}
    merged = dict(derived)

    for branch_id, branch in BRANCHES_BY_ID.items():
        # A branch-level override wins over the municipality lookup: the registry's
        # own `municipality` field is not always right (see the seed file's note on
        # Pinhal Novo, which the corpus caught).
        rule = seed.get("branch_overrides", {}).get(branch_id) or seed.get("municipalities", {}).get(
            getattr(branch, "municipality", "") or ""
        )
        if rule is None:
            continue
        closed, opened, _unobservable = observed_status(branch_id, rule, present_by_date, spans)
        seeded_entry = {
            **rule,
            "municipality": getattr(branch, "municipality", None),
            "derived_from": "published list, corpus-checked",
            "corpus_closed_years": closed,
            "corpus_open_years": opened,
        }
        existing = merged.get(branch_id)

        if opened and not closed:
            # The branch demonstrably traded on this date. Never accept.
            outcomes["contradicted"].append(f"{branch_id} ({rule.get('name')}) open in {opened} year(s)")
            continue
        if existing is None:
            merged[branch_id] = seeded_entry
            outcomes["confirmed" if closed else "added"].append(branch_id)
            continue
        # Both present: does the derived (month, day) match what the rule resolves to?
        sample_year = max({d.year for d in present_by_date.index})
        resolved = resolve_rule(rule, sample_year)
        if resolved is not None and (existing.get("month"), existing.get("day")) == (resolved.month, resolved.day):
            merged[branch_id] = {**seeded_entry, "years_observed": existing.get("years_observed")}
            outcomes["confirmed"].append(branch_id)
        else:
            outcomes["conflict"].append(
                f"{branch_id}: derived {existing.get('month'):02d}-{existing.get('day'):02d} "
                f"vs seed {rule.get('name')} — keeping derived"
            )
    return merged, outcomes


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-years", type=int, default=DEFAULT_MIN_YEARS)
    parser.add_argument("--input", default=IALC_BASELINE_PATH)
    parser.add_argument("--output", default=MUNICIPAL_HOLIDAYS_PATH)
    parser.add_argument("--seed", default=MUNICIPAL_HOLIDAYS_SEED_PATH)
    args = parser.parse_args()

    frame = load_weekday_frame(args.input)

    verification = verify_against_corpus(frame)
    logger.info(
        "National list check: %d holidays confirmed closed, %d observed open",
        len(verification["observed_closed"]),
        len(verification["observed_open"]),
    )
    for entry in verification["observed_open"]:
        logger.warning("National holiday appears OPEN in data: %s", entry)

    derived = derive(frame, min_years=args.min_years)
    logger.info("Derived from data alone: %d of %d branches", len(derived), len(BRANCHES_BY_ID))

    seed = load_seed(args.seed)
    spans = branch_active_spans(frame)
    present_by_date = frame.groupby(frame["date"].dt.date)["branch_id"].apply(lambda s: {str(x) for x in s})
    merged, outcomes = reconcile(derived, seed, present_by_date, spans)

    logger.info(
        "Reconciled against seed list: %d confirmed, %d added, %d contradicted, %d conflicts",
        len(outcomes["confirmed"]), len(outcomes["added"]), len(outcomes["contradicted"]), len(outcomes["conflict"]),
    )
    for entry in outcomes["contradicted"]:
        logger.warning("SEED REJECTED (branch observed open): %s", entry)
    for entry in outcomes["conflict"]:
        logger.warning("SEED/DATA CONFLICT: %s", entry)

    missing = sorted(set(BRANCHES_BY_ID) - set(merged))
    document = {
        "branches": merged,
        "undetermined_branches": missing,
        "method": {
            "source": args.input,
            "seed": args.seed,
            "min_years": args.min_years,
            "min_weekday_presence_rate": MIN_WEEKDAY_PRESENCE_RATE,
            "min_national_open_rate": MIN_NATIONAL_OPEN_RATE,
            "precedence": "corpus evidence overrides the seed list on any contradiction",
        },
        "reconciliation": outcomes,
        "national_list_verification": verification,
    }
    Path(args.output).write_text(json.dumps(document, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Wrote %s (%d undetermined branches left unlisted, not guessed)", args.output, len(missing))

    for branch_id, entry in sorted(merged.items(), key=lambda kv: str(kv[1].get("municipality"))):
        kind = entry.get("kind", "fixed")
        when = f'{entry["month"]:02d}-{entry["day"]:02d}' if kind == "fixed" else f'[{kind}]'
        logger.info("  %-24s %-28s %-38s %s", when, entry.get("municipality") or "?", entry.get("name") or "", branch_id)


if __name__ == "__main__":
    main()
