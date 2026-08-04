"""Emits the static site's data file from the real IALC-M corpus.

The citizen-facing page (site/index.html) is deliberately a *planning* tool,
not a live-queue display -- the official SIGA site already shows live state and
the page links out to it. That single positioning decision is what makes a
static build viable: nothing on the page needs to be fresh-to-the-minute, so
the entire payload is precomputable (~19 KB for all 78 branches, ~5 KB
gzipped) and GitHub Pages can serve it with no backend, no hosted 400MB+ DB,
and no cold starts.

Output is written as `window.LOJA_DATA = {...}` in a plain .js file rather than
a .json fetched at runtime. Two reasons:
  1. `fetch()` against a file:// URL is blocked by CORS, so a .json build
     could only ever be tested through a local web server. A <script> include
     works identically from file://, GitHub Pages, and any static host --
     meaning the page can be opened by double-clicking it.
  2. No build chain, bundler, or framework is needed to consume it.

Numbers here are **measured averages from IALC-M**, not model output. For the
day-level view that is both simpler and easier to defend: a planning-ahead
/predict call already falls back to baselines for people_waiting, rolling wait
stats, and live weather, so the model would largely be reconstructing these
same aggregates. The trained model earns its place when this page gains a
per-service dimension (IALC-M is branch-level only) -- at which point it runs
*here*, at build time, and still ships as static JSON.

Run:
    python -m pipeline.build_static
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from config import BRANCHES_BY_ID, REROUTE_RADIUS_KM
from pipeline.holidays_pt import (
    add_holiday_features,
    holiday_closure_mask,
    load_municipal_holidays,
    resolve_rule,
)

IALC_BASELINE_PATH = "data/cleaned_ialc_baseline.parquet"
OUTPUT_PATH = "site/data.js"

# A branch-day whose "average" rests on very few attendances is a single noisy
# reading, not a stable mean -- the same concern config.py documents for
# HISTORICAL_REAL_AVG_REFERENCE_SAMPLE_SIZE. 30 reuses that threshold rather
# than introducing an unrelated number.
MIN_ATTENDANCES_FOR_STABLE_MEAN = 30

SPARKLINE_WEEKS = 12
MAX_ALTERNATIVES = 3

# A branch sees roughly 10-15 holiday-adjacent days a year, so three years of
# corpus gives ~30-45. Requiring 10 keeps the uplift a mean rather than an
# anecdote, without discarding branches that opened partway through the span.
MIN_ADJACENT_DAYS_FOR_UPLIFT = 10

# How many years ahead the payload resolves movable municipal holidays for.
# Two covers "this year and next", which is as far ahead as anyone plans a
# trip to a government office.
PLANNABLE_YEARS = 2

# Occurrences of a given weekday averaged for the headline number. Measured by
# pipeline/backtest_site.py, not chosen: MAE is U-shaped in this value (too few
# = noisy, too many = stale), and 8 was validated across 10x30d / 6x60d / 4x90d
# rather than by taking one run's argmin — a first pass suggested 12 and a
# larger gain, which was the constant overfitting its own scoring windows.
# Roughly two months of history. Keep this in sync with
# backtest_site.RECENT_WINDOW_OCCURRENCES, which is what validates it.
RECENT_WEEKDAY_OCCURRENCES = 8

# Occurrences used for the displayed IQR band. Deliberately larger than the
# mean's window: a quartile estimated from 8 points is noise. Not backtested —
# the backtest scored point predictions, and there is no equivalent measurement
# for interval width here, so this is a stability judgement, not a result.
BAND_WINDOW_OCCURRENCES = 26


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def load_stable_branch_days(path: str = IALC_BASELINE_PATH) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["day_of_week"] = frame["date"].dt.dayofweek
    return frame[frame["total_attendances"] >= MIN_ATTENDANCES_FOR_STABLE_MEAN].copy()


def annotate_holidays(frame: pd.DataFrame) -> pd.DataFrame:
    """Tags each branch-day as a holiday closure, a bridge day, or a first-day-back.

    National holidays never appear in IALC-M at all (a closed branch-day is an
    absent row, verified 2026-08-01), so `is_holiday_closure` is expected to
    match almost nothing — it is kept as a live assertion that the assumption
    still holds, not as a filter doing real work. A nonzero count means either
    a branch traded through a holiday or the holiday table drifted; both are
    worth surfacing rather than silently averaging in.

    The flags that do real work are the neighbours. Measured across this
    corpus, normalized per branch: bridge days run +26% over a branch's own
    baseline and first-days-back +24%. Leaving them in makes every "typical
    Tuesday" figure quietly pessimistic, which for a page whose whole job is
    "when should I go" is the wrong direction to be wrong in.
    """
    frame = frame.copy()
    frame["is_holiday_closure"] = holiday_closure_mask(frame, date_column="date")
    # add_holiday_features works off a timestamp column; `date` is already
    # midnight-stamped, so reusing it needs no extra normalization.
    annotated = add_holiday_features(frame, timestamp_column="date", branch_column="branch_id")
    frame["is_bridge_day"] = annotated["is_bridge_day"].astype(bool)
    frame["is_post_holiday"] = annotated["is_post_holiday"].astype(bool)
    return frame


def resolve_municipal_dates(
    municipal: dict[str, dict[str, Any]], frame: pd.DataFrame
) -> dict[str, list[list[int]]]:
    """branch_id -> [[month, day], ...] for the years a visitor can plan into.

    Movable rules (11 of 76 branches) resolve to a different date each year, so
    the payload carries resolved dates rather than a rule the page would have
    to evaluate — keeping the Easter arithmetic in Python, where it is tested,
    instead of reimplementing it in browser JavaScript.
    """
    current_year = int(pd.to_datetime(frame["date"]).max().year)
    years = range(current_year, current_year + PLANNABLE_YEARS)
    resolved: dict[str, list[list[int]]] = {}
    for branch_id, rule in municipal.items():
        dates = [resolve_rule(rule, year) for year in years]
        pairs = sorted({(d.month, d.day) for d in dates if d is not None})
        if pairs:
            resolved[branch_id] = [[month, day] for month, day in pairs]
    return resolved


def typical_days(frame: pd.DataFrame) -> pd.DataFrame:
    """The subset a citizen planning an ordinary visit should be shown."""
    return frame[~(frame["is_holiday_closure"] | frame["is_bridge_day"] | frame["is_post_holiday"])]


def branches_reporting_wait_times(frame: pd.DataFrame) -> set[str]:
    """Branch ids that actually record wait times.

    Found 2026-07-31: Águeda reports `avg_wait_minutes == 0.0` on every one of
    its 42 stable branch-days (1,426 real attendances), as does Palmela Móvel
    over 7 days. Zero variance at exactly 0.0 across dozens of days with real
    attendance is the signature of a branch that doesn't report the field --
    a genuinely fast branch would still vary day to day.

    This matters because such a branch would otherwise surface as an
    unbeatable "0 min" alternative that always wins the reroute comparison --
    the same trap api/service.py's find_smart_reroute already guards against
    for closed branches, arriving here by a different route. Their own page
    shows "sem dados" rather than a fabricated zero.
    """
    observed_any_wait = frame.groupby("branch_id")["avg_wait_minutes"].max()
    return {str(branch_id) for branch_id, peak in observed_any_wait.items() if peak > 0}


def branches_with_colliding_coordinates() -> set[str]:
    """Branch ids whose geocoded point is shared with another branch.

    Found 2026-07-31: five pairs sit on byte-identical coordinates --
    Barreiro/Setúbal, Queluz/Cacém, Porto/Vila Nova de Gaia, Leiria/Ansião,
    Viseu/Sátão. Nominatim resolved both members of each pair to the same
    point (most are `geocode_precision == "municipality"` fallbacks), so any
    distance computed from them is provably wrong -- Barreiro and Setúbal are
    ~20 km apart in reality yet both report 9.8 km from Saldanha.

    Such branches are excluded from *being offered* as alternatives, since a
    reroute suggestion is only as good as its distance. They keep their own
    page: their wait-time data is real and unaffected.
    """
    seen: dict[tuple[float, float], list[str]] = {}
    for branch_id, branch in BRANCHES_BY_ID.items():
        key = (round(branch.latitude, 4), round(branch.longitude, 4))
        seen.setdefault(key, []).append(branch_id)
    colliding: set[str] = set()
    for ids in seen.values():
        if len(ids) > 1:
            colliding.update(ids)
    return colliding


# Portuguese particles stay lowercase inside a place name ("Viana do Castelo",
# never "Viana Do Castelo") unless they open it.
_LOWERCASE_PARTICLES = frozenset({"de", "do", "da", "dos", "das", "e"})


def normalize_district(district: str) -> str:
    """The registry carries both "Vila Real" and "Vila real"; without this the
    branch picker renders two separate groups for one district."""
    words = district.strip().split()
    return " ".join(
        word.lower() if index and word.lower() in _LOWERCASE_PARTICLES else word.capitalize()
        for index, word in enumerate(words)
    )


def build_branch_payload(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """One compact record per branch. Keys are deliberately short -- this file
    is downloaded by every visitor, and the page is the only consumer."""
    # `reporting` is judged on the FULL frame: whether a branch ever records a
    # wait time is a property of the branch, and must not become an artifact
    # of which days the typical-day filter happened to keep.
    reporting = branches_reporting_wait_times(frame)
    colliding = branches_with_colliding_coordinates()
    municipal = load_municipal_holidays()
    resolved_municipal = resolve_municipal_dates(municipal, frame)

    typical = typical_days(frame)
    # The reroute comparison uses the SAME recent basis as the headline
    # numbers. Ranking alternatives on a three-year mean while showing a
    # recent-window mean would let the page recommend a branch on evidence it
    # does not display — and the whole point of the recent window is that the
    # older number can be stale.
    weekday_frame = typical[typical["day_of_week"] < 5].sort_values("date")
    branch_mean_wait = weekday_frame.groupby("branch_id")["avg_wait_minutes"].apply(
        lambda s: s.tail(RECENT_WEEKDAY_OCCURRENCES * 5).mean()
    )

    # Uplift on holiday-adjacent days, per branch — the number the page warns
    # with. Computed against the branch's own typical mean so it is a real
    # multiplier for that branch, not a national average applied blindly.
    adjacent = frame[frame["is_bridge_day"] | frame["is_post_holiday"]]
    adjacent_mean = adjacent.groupby("branch_id")["avg_wait_minutes"].mean()
    typical_mean = typical.groupby("branch_id")["avg_wait_minutes"].mean()

    payload: dict[str, dict[str, Any]] = {}
    for branch_id, group in typical.groupby("branch_id"):
        branch = BRANCHES_BY_ID.get(str(branch_id))
        if branch is None:
            continue

        weekday_frame_branch = group[group["day_of_week"] < 5].sort_values("date")
        # RECENT window, not all history — measured, not assumed. A
        # rolling-origin backtest over 10 windows x 30 days
        # (pipeline/backtest_site.py) found the last-N-occurrences mean beats a
        # three-year mean 6.244 vs 6.797 MAE, winning 9 of 10 windows. Waits
        # drift; a three-year average partly describes a branch that no longer
        # exists. N validated across three window configurations rather than
        # taken from one run's argmin — see RECENT_WEEKDAY_OCCURRENCES.
        weekday_means = weekday_frame_branch.groupby("day_of_week")["avg_wait_minutes"].apply(
            lambda s: s.tail(RECENT_WEEKDAY_OCCURRENCES).mean()
        )
        # Interquartile range, not p10-p90: measured across the corpus, the
        # IQR is ~50% of the mean (Saldanha on a Friday: 36-47 min against a
        # 41 min mean) while p10-p90 is ~97%, wide enough to read as "we
        # don't know". Half of real days land inside the IQR, which is a
        # claim the data supports and a citizen can act on.
        #
        # The BAND uses a longer window than the mean, deliberately. Quartiles
        # from 8 points are noise — you cannot estimate a spread from the same
        # handful of observations that barely pin down a centre. The backtest
        # validated the MEAN only; no equivalent evidence exists for the band,
        # so it takes the widest window that is still recent rather than
        # inheriting a number tuned for a different quantity.
        band_group = weekday_frame_branch.groupby("day_of_week")["avg_wait_minutes"]
        weekday_q1 = band_group.apply(lambda s: s.tail(BAND_WINDOW_OCCURRENCES).quantile(0.25))
        weekday_q3 = band_group.apply(lambda s: s.tail(BAND_WINDOW_OCCURRENCES).quantile(0.75))
        monthly_means = group.groupby(group["date"].dt.month)["avg_wait_minutes"].mean()
        weekly_trend = (
            group.set_index("date")["avg_wait_minutes"].resample("W").mean().dropna().tail(SPARKLINE_WEEKS)
        )
        give_up_rate = (group["total_desistencias"] / group["total_senhas"].replace(0, pd.NA)).mean()

        alternatives = []
        for other_id, other in BRANCHES_BY_ID.items():
            if other_id == branch_id or other_id not in branch_mean_wait.index:
                continue
            # A branch that never reports a wait would always look like the
            # fastest option available -- see branches_reporting_wait_times.
            if other_id not in reporting:
                continue
            # Its distance would be a fabrication -- see the docstring.
            if other_id in colliding:
                continue
            distance_km = haversine_km(branch.latitude, branch.longitude, other.latitude, other.longitude)
            if distance_km > REROUTE_RADIUS_KM:
                continue
            alternatives.append(
                {"id": other_id, "km": round(distance_km, 1), "w": round(float(branch_mean_wait[other_id]))}
            )
        alternatives.sort(key=lambda a: a["w"])

        payload[str(branch_id)] = {
            "n": branch.name,
            "d": normalize_district(branch.district),
            # False => this branch reports no wait times at all; the page must
            # say "sem dados" instead of rendering a fabricated 0 min.
            "ok": str(branch_id) in reporting,
            # Mon..Fri. 0 means "no stable data for that weekday at this branch".
            "wk": [int(round(weekday_means.get(i, 0))) for i in range(5)],
            # Mon..Fri, each [q1, q3] — the range the page actually displays.
            "wq": [
                [int(round(weekday_q1.get(i, 0))), int(round(weekday_q3.get(i, 0)))]
                for i in range(5)
            ],
            "mo": [int(round(monthly_means.get(i, 0))) for i in range(1, 13)],
            "sp": [int(round(v)) for v in weekly_trend.tolist()],
            "gu": round(100 * float(give_up_rate), 1),
            "alt": alternatives[:MAX_ALTERNATIVES],
            # This branch's feriado municipal, RESOLVED to concrete [month, day]
            # pairs for the years a visitor might plan into — 11 of 76 are
            # movable (Ascensão, Segunda-feira de Páscoa, Pentecostes, Monday
            # after the nth Sunday), so a single [month, day] would be wrong for
            # them every year. Omitted, never guessed, where neither the corpus
            # nor the published list could establish it; the page must treat a
            # missing key as "unknown", not as "no municipal holiday".
            **(
                {"hol": resolved_municipal[str(branch_id)]}
                if str(branch_id) in resolved_municipal
                else {}
            ),
            # Multiplier on holiday-adjacent days (pontes, first day back).
            # Emitted only where enough such days exist to be a mean rather
            # than an anecdote.
            **(
                {"pon": round(float(adjacent_mean[branch_id] / typical_mean[branch_id]), 2)}
                if branch_id in adjacent_mean.index
                and branch_id in typical_mean.index
                and typical_mean[branch_id] > 0
                and len(adjacent[adjacent["branch_id"] == branch_id]) >= MIN_ADJACENT_DAYS_FOR_UPLIFT
                else {}
            ),
        }
    return payload


def build_corpus_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """The credibility line's numbers. Only real measured data is counted here
    -- never the ~6.9M historical_derived_proxy training labels, which are
    formula-derived and would misrepresent the corpus if quoted as records."""
    return {
        "attendances": int(frame["total_attendances"].sum()),
        "branch_days": int(len(frame)),
        "branches": int(frame["branch_id"].nunique()),
        "districts": int(frame["district"].nunique()),
        "give_up_national": round(100 * float((frame["total_desistencias"] / frame["total_senhas"].replace(0, pd.NA)).mean()), 1),
        "from": str(frame["date"].min().date()),
        "to": str(frame["date"].max().date()),
    }


def main() -> None:
    frame = annotate_holidays(load_stable_branch_days())

    closures = int(frame["is_holiday_closure"].sum())
    if closures:
        # Expected to be zero — see annotate_holidays' docstring. Loud rather
        # than silent, because the alternative is a wrong holiday table that
        # nothing ever contradicts.
        print(f"warning: {closures} branch-days fall on a holiday this branch should have been closed for")
    print(
        f"holiday-adjacent days excluded from typical means: "
        f"{int((frame['is_bridge_day'] | frame['is_post_holiday']).sum())} of {len(frame)}"
    )

    # The corpus summary counts everything really measured, holiday-adjacent
    # days included — it is a statement about the record, not about a typical
    # visit, so filtering it would understate the evidence base.
    document = {"branches": build_branch_payload(frame), "corpus": build_corpus_summary(frame)}

    blob = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    output = Path(OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"window.LOJA_DATA={blob};\n", encoding="utf-8")

    print(f"wrote {output} - {len(document['branches'])} branches, {len(blob) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
