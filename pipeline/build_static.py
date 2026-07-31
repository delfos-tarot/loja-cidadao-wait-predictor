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

IALC_BASELINE_PATH = "data/cleaned_ialc_baseline.parquet"
OUTPUT_PATH = "site/data.js"

# A branch-day whose "average" rests on very few attendances is a single noisy
# reading, not a stable mean -- the same concern config.py documents for
# HISTORICAL_REAL_AVG_REFERENCE_SAMPLE_SIZE. 30 reuses that threshold rather
# than introducing an unrelated number.
MIN_ATTENDANCES_FOR_STABLE_MEAN = 30

SPARKLINE_WEEKS = 12
MAX_ALTERNATIVES = 3


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


def build_branch_payload(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """One compact record per branch. Keys are deliberately short -- this file
    is downloaded by every visitor, and the page is the only consumer."""
    reporting = branches_reporting_wait_times(frame)
    weekday_frame = frame[frame["day_of_week"] < 5]
    branch_mean_wait = weekday_frame.groupby("branch_id")["avg_wait_minutes"].mean()

    payload: dict[str, dict[str, Any]] = {}
    for branch_id, group in frame.groupby("branch_id"):
        branch = BRANCHES_BY_ID.get(str(branch_id))
        if branch is None:
            continue

        weekday_means = group[group["day_of_week"] < 5].groupby("day_of_week")["avg_wait_minutes"].mean()
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
            distance_km = haversine_km(branch.latitude, branch.longitude, other.latitude, other.longitude)
            if distance_km > REROUTE_RADIUS_KM:
                continue
            alternatives.append(
                {"id": other_id, "km": round(distance_km, 1), "w": round(float(branch_mean_wait[other_id]))}
            )
        alternatives.sort(key=lambda a: a["w"])

        payload[str(branch_id)] = {
            "n": branch.name,
            "d": branch.district,
            # False => this branch reports no wait times at all; the page must
            # say "sem dados" instead of rendering a fabricated 0 min.
            "ok": str(branch_id) in reporting,
            # Mon..Fri. 0 means "no stable data for that weekday at this branch".
            "wk": [int(round(weekday_means.get(i, 0))) for i in range(5)],
            "mo": [int(round(monthly_means.get(i, 0))) for i in range(1, 13)],
            "sp": [int(round(v)) for v in weekly_trend.tolist()],
            "gu": round(100 * float(give_up_rate), 1),
            "alt": alternatives[:MAX_ALTERNATIVES],
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
    frame = load_stable_branch_days()
    document = {"branches": build_branch_payload(frame), "corpus": build_corpus_summary(frame)}

    blob = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    output = Path(OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"window.LOJA_DATA={blob};\n", encoding="utf-8")

    print(f"wrote {output} - {len(document['branches'])} branches, {len(blob) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
