"""Emits the per-service live view — the one thing historical data can never support.

SLC-M and IALC-M are both daily and branch-level, so "how is the IRN desk at
Laranjeiras, specifically" is unanswerable from three years of history. The SIGA
live corpus answers it from its first week. This module turns that corpus into a
static payload, the same way pipeline/build_static.py does for the day-level page.

SHOWS QUEUE LENGTH, NEVER A WAIT IN MINUTES. `tempoRealEspera` is unusable as a
duration and no filtering rescues it: in the subset where BOTH live signals agree
a queue exists (the cleanest 6.9% of readings), it reports a median 162 minutes
against a median of ONE person waiting — 162 minutes to serve one citizen,
against IALC-M's measured 7.2 min service time. Wrong by ~22x. Meanwhile the two
signals contradict each other on 29.5% of open-desk readings (26.0% report a wait
with nobody there). `people_waiting` is a plain count, it is internally coherent,
and median 1 / mean 2.3 when queued is plausible. So this view answers "will
there be a line, and how long is it in PEOPLE" — the questions the data can
actually support.

LEADS WITH WALK-IN PROBABILITY, and that is the whole point. Over 74,466
open-desk readings, 93% show nobody waiting; when there IS a line its median is
one person. The experience is bimodal — walk straight up, or join a queue — so a
single average would describe almost nobody, which is exactly the flaw the
day-level page inherits from IALC-M averages. No reason to rebuild it here.

`pipeline/backtest_live.py` measured which parts are actually predictable
(3 rolling folds, target = queue length in people):

    dimensions        brier (walk-in)   MAE (people | queued)   coverage
    global            0.0805            1.756                   1.00
    combo             0.0673            1.605                   0.99   <- enabled
    combo+hour        0.0812            1.604                   0.95
    combo+dow         0.0868            1.774                   0.54
    combo+dow+hour    0.0828            1.760                   0.17

Knowing the branch and service cuts the walk-in Brier score 16% and the queue
-length error 8.6% — it earns its place on BOTH targets, which is the bar.

DIMENSIONS ARE GATED, NOT CHOSEN. Hour and weekday are deliberately absent — not
because they are uninteresting, but because 8 days is not enough: hour is flat on
one target and worse on the other, and weekday collapses coverage to 54% (adding
hour on top takes it to 17%, i.e. 83% of test rows have no matching cell). Re-run
the gate as the scraper accumulates; when a dimension passes, extend GROUPING
here to match.

Run:
    python -m pipeline.build_live_services
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from config import BRANCHES_BY_ID
from pipeline.backtest_live import load_live

logger = logging.getLogger(__name__)

OUTPUT_PATH = "site/live_services.js"

# Dimensions the gate has approved. Keep in sync with backtest_live.decide().
GROUPING = ["branch_id", "desk_service_id"]

# A combo needs this many open-desk readings before it is shown at all. With
# ~24 sweeps/day a real desk clears this in under a week; below it, a
# "walk-in 100%" claim would rest on a handful of glances.
MIN_READINGS = 40

# Below this many QUEUED readings, the queue-length median is one or two numbers
# wearing a statistic's clothes. The combo still ships (its walk-in rate is the
# useful part) but the length is omitted rather than guessed.
MIN_QUEUED_FOR_LENGTH = 10


def build_payload(frame: pd.DataFrame) -> dict[str, Any]:
    grouped = frame.groupby(GROUPING)
    branches: dict[str, dict[str, Any]] = {}
    kept = dropped = 0

    for (branch_id, service), group in grouped:
        branch = BRANCHES_BY_ID.get(str(branch_id))
        if branch is None or len(group) < MIN_READINGS:
            dropped += 1
            continue
        queued = group[group["has_queue"] == 1]
        record: dict[str, Any] = {
            # Share of open-desk sightings with no queue at all — the headline.
            "wi": round(100 * float(1 - group["has_queue"].mean())),
            "n": int(len(group)),
        }
        if len(queued) >= MIN_QUEUED_FOR_LENGTH:
            # MEDIAN people in line, not mean: queue length is right-skewed and
            # a mean would be dragged by the occasional bad afternoon.
            record["ql"] = int(round(float(queued["people"].median())))
            record["qn"] = int(len(queued))
        entry = branches.setdefault(str(branch_id), {"n": branch.name, "d": branch.district, "s": {}})
        entry["s"][str(service)] = record
        kept += 1

    logger.info("Kept %d (branch, service) combos, dropped %d below %d readings", kept, dropped, MIN_READINGS)
    return branches


def build_corpus(frame: pd.DataFrame) -> dict[str, Any]:
    """Provenance the page can show. Deliberately states the window and the
    bimodality, because both change how the numbers should be read."""
    return {
        "readings": int(len(frame)),
        "days": int(frame["date"].nunique()),
        "from": str(frame["date"].min()),
        "to": str(frame["date"].max()),
        "branches": int(frame["branch_id"].nunique()),
        "walk_in_national": round(100 * float(1 - frame["has_queue"].mean())),
        # None, not 0, when nothing was ever queued — a fresh corpus or a
        # quiet filtered subset must not crash the build, and must not claim
        # "median queue: 0 people", which reads as a measurement rather than an
        # absence of one.
        "queue_median_national": (
            int(round(float(queued["people"].median())))
            if (queued := frame.loc[frame["has_queue"] == 1]) is not None and len(queued)
            else None
        ),
        "coherent_pct": round(100 * float(frame["coherent"].mean()), 1),
        "dimensions": GROUPING,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    frame = load_live()
    document = {"branches": build_payload(frame), "corpus": build_corpus(frame)}

    blob = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"window.LOJA_LIVE={blob};\n", encoding="utf-8")

    corpus = document["corpus"]
    print(f"wrote {output} - {len(document['branches'])} branches, {len(blob) / 1024:.1f} KB")
    print(f"  {corpus['readings']:,} readings over {corpus['days']} days ({corpus['from']} -> {corpus['to']})")
    queue_note = (f"median {corpus['queue_median_national']} people"
                  if corpus["queue_median_national"] is not None else "no queued readings yet")
    print(f"  nationally: {corpus['walk_in_national']}% of open desks have no queue; "
          f"when there is one, {queue_note}")
    print(f"  live signal coherence: {corpus['coherent_pct']}% of readings have both fields agreeing")


if __name__ == "__main__":
    main()
