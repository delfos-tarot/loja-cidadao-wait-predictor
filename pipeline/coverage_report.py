"""Reports real siga_live data coverage per (branch_id, desk_service_id) combo.

Answers the concrete question underlying pipeline/train.py's per-combo
proxy-decay weighting: which combos actually have real coverage right now,
and how much? A single "N live rows total" number hides that coverage is
combo-by-combo, not global — this makes it visible, combo by combo, with the
same (branch_id, desk_service_id) keys compute_sample_weights groups by.

Also flags whether each observed SIGA desk_service_id string exists verbatim
in config.DESK_SERVICES (the dados.gov.pt-derived vocabulary the proxy rows
use) — if not, that combo's proxy rows will never see their weight decay
from this live data, since compute_sample_weights joins on the raw string,
and a SIGA name like "Geral" vs a dados.gov.pt name like "Atendimento Geral"
are different keys even though they're the same real-world service. Checked
2026-07-27: 186/212 (88%) of observed SIGA service names already match
exactly, so this affects a real but minority slice of combos, not most.

Usage:
    python -m pipeline.coverage_report
    python -m pipeline.coverage_report --min-samples 30 --top 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from config import BRANCHES, DEFAULT_DB_PATH, DESK_SERVICES, PROXY_WEIGHT_DECAY_ALPHA
from pipeline.db import load_all_samples

logger = logging.getLogger(__name__)

COVERAGE_REPORT_PATH = "data/siga_coverage_report.csv"


def build_coverage_report(db_path: str = DEFAULT_DB_PATH, alpha: float = PROXY_WEIGHT_DECAY_ALPHA) -> pd.DataFrame:
    frame = load_all_samples(db_path)
    live = frame[frame["source"] == "siga_live"]

    columns = ["branch_id", "desk_service_id", "live_count", "first_seen", "last_seen", "proxy_weight", "matches_proxy_vocabulary"]
    if live.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        live.groupby(["branch_id", "desk_service_id"])
        .agg(live_count=("sampled_at", "size"), first_seen=("sampled_at", "min"), last_seen=("sampled_at", "max"))
        .reset_index()
    )
    summary["proxy_weight"] = 1.0 / (1.0 + alpha * summary["live_count"])
    summary["matches_proxy_vocabulary"] = summary["desk_service_id"].isin(DESK_SERVICES)
    return summary.sort_values("live_count", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report real siga_live coverage per (branch, service) combo")
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--min-samples", type=int, default=30, help="Threshold to call a combo 'meaningfully covered'")
    parser.add_argument("--out", default=COVERAGE_REPORT_PATH)
    parser.add_argument("--top", type=int, default=15, help="How many top combos to print to console")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    report = build_coverage_report(args.db)

    total_registry_branches = len(BRANCHES)
    total_registry_combos = sum(len(b.desk_service_ids) for b in BRANCHES)

    if report.empty:
        print(f"No real siga_live samples yet in {args.db}.")
        print(f"Registry scope: {total_registry_branches} branches, {total_registry_combos} known (branch, service) combos.")
        print("Run the scraper (python -m scrapers.siga_scraper --once) to start collecting real data.")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)

    total_combos_observed = len(report)
    total_live_rows = int(report["live_count"].sum())
    meaningfully_covered = int((report["live_count"] >= args.min_samples).sum())
    vocabulary_matched = int(report["matches_proxy_vocabulary"].sum())
    orphaned = total_combos_observed - vocabulary_matched

    print(f"Registry scope: {total_registry_branches} branches, {total_registry_combos} known (branch, service) combos")
    print(f"Real (branch, service) combos observed so far: {total_combos_observed}")
    print(f"Total real siga_live rows: {total_live_rows}")
    print(f"Combos with >= {args.min_samples} real samples: {meaningfully_covered}")
    print()
    print(
        f"Combos whose SIGA service name matches the dados.gov.pt vocabulary: "
        f"{vocabulary_matched}/{total_combos_observed} — these are the ones where compute_sample_weights' "
        f"proxy-decay actually applies."
    )
    if orphaned:
        print(
            f"{orphaned} combo(s) use a SIGA-only service name with no dados.gov.pt match — their proxy rows "
            f"(if any exist under the matching dados.gov.pt name) will never decay from this live data until "
            f"a service-name reconciliation step exists (see pipeline/reconcile_siga_branches.py's docstring "
            f"for the analogous branch-name problem)."
        )
    print()
    print(f"Top {min(args.top, len(report))} combos by real sample count:")
    for _, row in report.head(args.top).iterrows():
        vocab_flag = "" if row["matches_proxy_vocabulary"] else "  [ORPHANED — no proxy-vocabulary match]"
        print(
            f"  {row['branch_id']:35s} {row['desk_service_id']:35s} n={int(row['live_count']):>5d}  "
            f"first={row['first_seen']}  last={row['last_seen']}  weight={row['proxy_weight']:.3f}{vocab_flag}"
        )
    print()
    print(f"Full per-combo report saved to {args.out}")


if __name__ == "__main__":
    main()
