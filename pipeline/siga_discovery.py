"""One-time (well, occasional) discovery crawl of the real SIGA API.

A full national crawl is a genuine 3-level nest — district x entidade x
senha, confirmed by testing that GetLocais requires both IdEntidade and
IdSenha (no bulk "everything in this district" shortcut exists). Across the
18 mainland districts this is on the order of a few thousand GetSenhas/
GetLocais calls, so it's run as its own occasional step (minutes, not
seconds) rather than something scrapers/siga_scraper.py redoes on every
15-minute poll.

This script filters to locations whose name matches "Loja de Cidadão" (this
project's actual scope — the same SIGA platform also serves many other
unrelated entities/locations: utility companies, municipal offices, AIMA
immigration desks, etc.) and records:
  - data/siga_relevant_queries.json — the (distrito, entidade, senha) triples
    that returned at least one matching location, for scrapers/siga_scraper.py
    to poll going forward (no need to redo the full crawl each time).
  - data/siga_discovered_locations.json — the raw matched location records,
    for pipeline/reconcile_siga_branches.py to join against the existing
    (dados.gov.pt-derived) branch registry.

Checkpointed incrementally (writes after each district) so an interrupted
run doesn't lose progress; already-covered districts are skipped on restart
unless --force is passed.

Usage:
    python -m pipeline.siga_discovery
    python -m pipeline.siga_discovery --distritos 11,13
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from pipeline.siga_client import MAINLAND_DISTRITOS, SigaClient

logger = logging.getLogger(__name__)

RELEVANT_QUERIES_PATH = "data/siga_relevant_queries.json"
DISCOVERED_LOCATIONS_PATH = "data/siga_discovered_locations.json"
REQUEST_DELAY_SECONDS = 0.3
NAME_FILTER = "loja de cidadão"


def _load_json(path: str) -> list:
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    return []


def _save_json(path: str, data: list) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))


def crawl_district(client: SigaClient, id_distrito: int) -> tuple[list[dict], list[dict]]:
    """Returns (relevant_queries, matched_locations) for one district."""
    relevant_queries: list[dict] = []
    matched_locations: list[dict] = []

    try:
        entidades = client.get_entidades(id_distrito)
    except Exception:
        logger.exception("Failed to list entidades for distrito %d", id_distrito)
        return relevant_queries, matched_locations
    time.sleep(REQUEST_DELAY_SECONDS)

    logger.info("Distrito %d (%s): %d entidades", id_distrito, MAINLAND_DISTRITOS.get(id_distrito), len(entidades))

    for entidade in entidades:
        try:
            senhas = client.get_senhas(id_distrito, entidade["id"], entidade.get("idInstituicao", 0))
        except Exception:
            logger.exception("Failed to list senhas for distrito=%d entidade=%d", id_distrito, entidade["id"])
            continue
        time.sleep(REQUEST_DELAY_SECONDS)

        for senha in senhas:
            try:
                locais = client.get_locais(id_distrito, entidade["id"], senha["id"], entidade.get("idInstituicao", 0))
            except Exception:
                logger.exception(
                    "Failed to list locais for distrito=%d entidade=%d senha=%d", id_distrito, entidade["id"], senha["id"]
                )
                continue
            time.sleep(REQUEST_DELAY_SECONDS)

            matches = [loc for loc in locais if NAME_FILTER in loc.get("nome", "").lower()]
            if matches:
                relevant_queries.append(
                    {
                        "distrito_id": id_distrito,
                        "distrito_nome": MAINLAND_DISTRITOS.get(id_distrito),
                        "entidade_id": entidade["id"],
                        "entidade_nome": entidade["nome"],
                        "id_instituicao": entidade.get("idInstituicao", 0),
                        "senha_id": senha["id"],
                        "senha_nome": senha["nome"],
                    }
                )
                matched_locations.extend(matches)

    logger.info(
        "Distrito %d done: %d relevant (entidade,senha) queries, %d matched locations",
        id_distrito, len(relevant_queries), len(matched_locations),
    )
    return relevant_queries, matched_locations


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover which SIGA (distrito, entidade, senha) queries cover real Loja de Cidadão locations")
    parser.add_argument("--distritos", default=None, help="Comma-separated distrito ids to crawl (default: all mainland)")
    parser.add_argument("--force", action="store_true", help="Re-crawl districts already present in the output files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    target_distritos = (
        [int(d) for d in args.distritos.split(",")] if args.distritos else list(MAINLAND_DISTRITOS.keys())
    )

    relevant_queries = _load_json(RELEVANT_QUERIES_PATH)
    discovered_locations = _load_json(DISCOVERED_LOCATIONS_PATH)
    already_covered = {q["distrito_id"] for q in relevant_queries}

    client = SigaClient()
    for id_distrito in target_distritos:
        if id_distrito in already_covered and not args.force:
            logger.info("Skipping distrito %d (already covered; use --force to re-crawl)", id_distrito)
            continue

        district_queries, district_locations = crawl_district(client, id_distrito)
        relevant_queries = [q for q in relevant_queries if q["distrito_id"] != id_distrito] + district_queries
        discovered_locations = (
            [loc for loc in discovered_locations if loc.get("distritoId") != id_distrito] + district_locations
        )

        _save_json(RELEVANT_QUERIES_PATH, relevant_queries)
        _save_json(DISCOVERED_LOCATIONS_PATH, discovered_locations)

    logger.info(
        "Discovery complete: %d relevant queries, %d matched locations across %d districts",
        len(relevant_queries), len(discovered_locations), len(target_distritos),
    )


if __name__ == "__main__":
    main()
