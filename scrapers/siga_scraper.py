"""Live SIGA queue-status poller — the real, verified public API at
siga.marcacaodeatendimento.pt (see pipeline/siga_client.py for how this was
confirmed: a bare `requests.post`, no browser session, no auth).

Polls only the (distrito, entidade, senha) queries known to cover real Loja
de Cidadão locations — data/siga_relevant_queries.json, built once by
pipeline/siga_discovery.py — not the full national SIGA network (which also
serves many unrelated entities: utility companies, municipal offices, AIMA
immigration desks, etc.). Each result is reconciled to this project's branch
registry via data/siga_branch_crosswalk.json (pipeline/reconcile_siga_branches.py)
and inserted into queue_samples tagged source='siga_live'.

Every poll — open AND closed — is recorded with its real is_open state; a
closed reading has no people_waiting/wait_time (that would be a meaningless
zero), but the is_open observation itself is useful; enough of them build a
real per-branch schedule over time; see pipeline/feature_engineering.py's
estimate_is_open_heuristic, which this data should eventually replace.

KNOWN LIMITATION: SIGA's own service names ("Geral", "Tesouraria") don't
always match dados.gov.pt's fuller service names ("Atendimento Geral") used
elsewhere in this pipeline — stored as-is here (source SIGA `servico.nome`).
Where the names differ, live rows accumulate under a different
desk_service_id than the historical proxy rows for the same real service, so
pipeline/train.py's per-combo proxy-decay weighting won't fully kick in for
those services until a service-name reconciliation step is built (analogous
to pipeline/reconcile_siga_branches.py, not done yet for services).

KNOWN DATA QUALITY ISSUE: real `tempoRealEspera` readings are sometimes
wildly implausible — found 2026-07-27, 46% of "open" readings from one
scheduled scrape showed 180-13,366 minutes, uncorrelated with the actual
people_waiting count. See config.REAL_DATA_MAX_PLAUSIBLE_WAIT_MINUTES for the
full writeup. `estimated_wait_minutes` is filtered to plausible values only;
the untouched value is always kept in `raw_wait_time_minutes` for later
investigation.

Usage:
    python -m scrapers.siga_scraper --once
    python -m scrapers.siga_scraper                 # loops forever, every 15 min
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from config import DEFAULT_DB_PATH, REAL_DATA_MAX_PLAUSIBLE_WAIT_MINUTES
from pipeline.db import get_connection, insert_queue_samples
from pipeline.siga_client import SigaClient
from schemas import QueueReading

logger = logging.getLogger(__name__)

RELEVANT_QUERIES_PATH = "data/siga_relevant_queries.json"
CROSSWALK_PATH = "data/siga_branch_crosswalk.json"
LIVE_SAMPLES_CSV_PATH = "data/live_samples.csv"
POLL_INTERVAL_SECONDS = 15 * 60
REQUEST_DELAY_SECONDS = 0.3
CLOSED_STATE = "FECHADO"  # the only state string actually confirmed by testing

CSV_FIELDS = [
    "branch_id", "desk_service_id", "sampled_at", "people_waiting",
    "last_ticket_called", "estimated_wait_minutes", "source", "is_open",
    "raw_wait_time_minutes",
    # Appended 2026-08-05 — _migrate_csv_header_if_needed rewrites an existing
    # file's header and pads old rows, so this is a safe pure append.
    "opening_hours", "service_state", "reported_service_minutes", "web_ticketing",
]


def load_relevant_queries(path: str = RELEVANT_QUERIES_PATH) -> list[dict]:
    if not Path(path).exists():
        raise SystemExit(f"{path} not found — run `python -m pipeline.siga_discovery` first")
    return json.loads(Path(path).read_text())


def load_crosswalk(path: str = CROSSWALK_PATH) -> dict[int, str]:
    """Returns {siga_location_id: branch_id}."""
    if not Path(path).exists():
        raise SystemExit(f"{path} not found — run `python -m pipeline.reconcile_siga_branches` first")
    entries = json.loads(Path(path).read_text())
    return {entry["siga_location_id"]: entry["branch_id"] for entry in entries}


def _dedupe_readings(readings: list[QueueReading]) -> list[QueueReading]:
    """The same physical (branch, service) can be reachable through more than
    one (entidade, senha) query path in a single poll — e.g. duplicate
    entidade entries in SIGA's own data (two separate "AIMA" ids seen during
    discovery). Found via pipeline/coverage_report.py showing combos with
    N>1 real samples that all share the exact same timestamp down to the
    microsecond — not real repeated observation over time, just the same
    instant counted multiple times. Keep one reading per combo per poll,
    preferring an open/informative reading over a closed duplicate if they
    disagree.
    """
    deduped: dict[tuple[str, str], QueueReading] = {}
    for reading in readings:
        key = (reading.branch_id, reading.desk_service_id)
        existing = deduped.get(key)
        if existing is None or (reading.is_open and not existing.is_open):
            deduped[key] = reading
    return list(deduped.values())


def poll_once(
    client: SigaClient, relevant_queries: list[dict], location_to_branch: dict[int, str]
) -> list[QueueReading]:
    readings: list[QueueReading] = []
    sampled_at = datetime.now(timezone.utc)

    for query in relevant_queries:
        try:
            locais = client.get_locais(
                query["distrito_id"], query["entidade_id"], query["senha_id"], query.get("id_instituicao", 0)
            )
        except Exception:
            logger.exception(
                "Failed to poll distrito=%s entidade=%s senha=%s",
                query["distrito_id"], query["entidade_id"], query["senha_id"],
            )
            time.sleep(REQUEST_DELAY_SECONDS)
            continue
        time.sleep(REQUEST_DELAY_SECONDS)

        for location in locais:
            branch_id = location_to_branch.get(location["id"])
            if branch_id is None:
                continue  # not one of our reconciled branches — out of scope

            servico = location.get("servico", {})
            state = servico.get("estado")
            is_open = state != CLOSED_STATE

            raw_wait_minutes = servico.get("tempoRealEspera") if is_open else None
            # See module + config docstrings: real tempoRealEspera readings
            # are sometimes wildly implausible (up to 13,366 minutes seen,
            # uncorrelated with actual queue size). raw_wait_minutes always
            # keeps the untouched value; estimated_wait_minutes is the
            # filtered one every other consumer (training, API) reads.
            plausible_wait_minutes = (
                raw_wait_minutes
                if raw_wait_minutes is not None and raw_wait_minutes <= REAL_DATA_MAX_PLAUSIBLE_WAIT_MINUTES
                else None
            )

            readings.append(
                QueueReading(
                    branch_id=branch_id,
                    desk_service_id=servico.get("nome", query["senha_nome"]),
                    sampled_at=sampled_at,
                    people_waiting=servico.get("utentesEmEspera") if is_open else None,
                    last_ticket_called=None,
                    estimated_wait_minutes=plausible_wait_minutes,
                    source="siga_live",
                    is_open=is_open,
                    raw_wait_time_minutes=raw_wait_minutes,
                    # Captured regardless of open/closed: `horario` is a
                    # schedule, so it is exactly as true at 22:00 as at noon,
                    # and it is the field that replaces config.py's hardcoded
                    # Mon-Fri 9-17 assumption. See schemas.QueueReading.
                    opening_hours=servico.get("horario") or None,
                    service_state=state,
                    reported_service_minutes=servico.get("tempoMedAtendimento") if is_open else None,
                    web_ticketing=servico.get("senhaWeb"),
                )
            )

    return _dedupe_readings(readings)


def _migrate_csv_header_if_needed(path: Path) -> None:
    """Rewrites the CSV in place if its on-disk header predates CSV_FIELDS.

    Found 2026-07-28: `raw_wait_time_minutes` was appended to CSV_FIELDS
    after live_samples.csv already existed with an 8-column header. The
    header is only ever written for a brand-new file, so every row written
    since then silently carried 9 fields under an 8-field header --
    corrupting the file for any strict reader (pandas.read_csv errors
    outright on the field-count mismatch; found via import_live_samples).
    Existing short rows are padded with '' for the new trailing column(s)
    rather than dropped, matching how other optional fields (e.g.
    last_ticket_called) already represent "not captured" as blank.
    """
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] == CSV_FIELDS:
        return
    old_header = rows[0]
    if not set(old_header).issubset(CSV_FIELDS):
        raise SystemExit(
            f"{path} header {old_header} has fields not in current CSV_FIELDS "
            f"{CSV_FIELDS} -- this isn't a pure append, needs a manual fix"
        )
    width = len(old_header)
    fixed_rows = [CSV_FIELDS]
    for row in rows[1:]:
        if len(row) == len(CSV_FIELDS):
            fixed_rows.append(row)
        elif len(row) == width:
            fixed_rows.append(row + [""] * (len(CSV_FIELDS) - width))
        else:
            raise SystemExit(f"{path} has a row with unexpected field count {len(row)}: {row}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(fixed_rows)


def append_readings_to_csv(readings: list[QueueReading], csv_path: str = LIVE_SAMPLES_CSV_PATH) -> int:
    """Appends readings to a plain CSV rather than the main SQLite DB.

    The main queue_history.db is dominated by ~1.7M formula-derived proxy
    rows and is 400MB+ — far past GitHub's 100MB per-file limit, and a poor
    fit for git regardless (binary SQLite diffs don't compress incrementally,
    so every commit would bloat history by the whole file's size). A plain,
    append-only CSV holding only real siga_live rows is small, and git can
    diff/delta-compress text efficiently, so repeated commits stay cheap —
    this is what a CI workflow (e.g. GitHub Actions) should write to; merge
    it into your local queue_history.db with pipeline/import_live_samples.py.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _migrate_csv_header_if_needed(path)
    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for reading in readings:
            writer.writerow(
                {
                    "branch_id": reading.branch_id,
                    "desk_service_id": reading.desk_service_id,
                    "sampled_at": reading.sampled_at.isoformat(),
                    "people_waiting": reading.people_waiting,
                    "last_ticket_called": reading.last_ticket_called,
                    "estimated_wait_minutes": reading.estimated_wait_minutes,
                    "source": reading.source,
                    "is_open": reading.is_open,
                    "raw_wait_time_minutes": reading.raw_wait_time_minutes,
                }
            )
    return len(readings)


def run_once(db_path: str = DEFAULT_DB_PATH, csv_path: str | None = None) -> int:
    client = SigaClient()
    relevant_queries = load_relevant_queries()
    location_to_branch = load_crosswalk()

    readings = poll_once(client, relevant_queries, location_to_branch)

    if csv_path is not None:
        stored = append_readings_to_csv(readings, csv_path)
        logger.info("Appended %d live siga_live samples to %s", stored, csv_path)
    else:
        with get_connection(db_path) as connection:
            stored = insert_queue_samples(connection, readings)
        logger.info("Stored %d live siga_live samples in %s", stored, db_path)
    return stored


def run_forever(db_path: str = DEFAULT_DB_PATH, interval_seconds: int = POLL_INTERVAL_SECONDS) -> None:
    while True:
        try:
            run_once(db_path)
        except Exception:
            logger.exception("Polling cycle failed")
        time.sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll live SIGA queue status for reconciled Loja de Cidadao branches")
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--once", action="store_true", help="Run a single polling cycle instead of looping every 15 minutes")
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Append to this CSV instead of writing to --db (for CI: small, git-friendly, merge locally with pipeline/import_live_samples.py)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.once:
        run_once(args.db, csv_path=args.csv_out)
    else:
        if args.csv_out is not None:
            raise SystemExit("--csv-out is only supported with --once (CI runs one cycle per invocation)")
        run_forever(args.db)


if __name__ == "__main__":
    main()
