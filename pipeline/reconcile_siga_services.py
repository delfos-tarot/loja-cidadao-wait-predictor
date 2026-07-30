"""Matches real SIGA desk-service names (servico.nome in
data/siga_discovered_locations.json) against config.DESK_SERVICES, the
dados.gov.pt-derived vocabulary proxy rows use.

The analogous problem to pipeline/reconcile_siga_branches.py, but for
service names instead of branch names -- documented in
pipeline/coverage_report.py's module docstring: a SIGA name like "Geral"
and a dados.gov.pt name like "Atendimento Geral" refer to the same
real-world desk, but compute_sample_weights joins on the raw string, so
that combo's proxy rows never see their weight decay from real siga_live
coverage until the two names are reconciled. Checked 2026-07-27: 186/212
(88%) of observed SIGA service names already match exactly -- this fixes
the remaining minority, not most combos.

No coordinate/distance signal exists for services the way it does for
branches, so this uses a single name-similarity threshold rather than the
two-signal (name + distance) cross-check reconcile_siga_branches.py can
do -- set higher than that script's floor (MIN_NAME_SIMILARITY) to
compensate for having only one signal to trust instead of two.

Output: data/siga_desk_service_crosswalk.json -- a list of
{siga_service_name, canonical_service_name, match_confidence}. Applied at
load time (pipeline/db.py's apply_service_crosswalk, wired into
load_all_samples), not by rewriting scraped data -- same reasoning as
pipeline/train.py's clean_siga_live_readings: keeps the raw scraped record
and the CI pipeline untouched, and applies retroactively to already-
collected rows automatically on every load rather than needing a one-off
migration.

Usage:
    python -m pipeline.reconcile_siga_services
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import unicodedata
from pathlib import Path

from config import DESK_SERVICES

logger = logging.getLogger(__name__)

DISCOVERED_LOCATIONS_PATH = "data/siga_discovered_locations.json"
SERVICE_CROSSWALK_PATH = "data/siga_desk_service_crosswalk.json"

# Higher than reconcile_siga_branches.py's MIN_NAME_SIMILARITY (0.55) --
# that script can fall back to a coordinate-distance cross-check for a
# mediocre name match; there's no equivalent second signal for services, so
# name similarity alone needs a higher bar to avoid a false match silently
# merging two different real desk services under one vocabulary entry.
MIN_NAME_SIMILARITY = 0.65


def normalize_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name.strip()).encode("ascii", "ignore").decode("ascii")
    return normalized.lower().strip()


def name_similarity(a: str, b: str) -> float:
    norm_a, norm_b = normalize_name(a), normalize_name(b)
    if norm_a == norm_b:
        return 1.0
    if norm_a in norm_b or norm_b in norm_a:
        # A short SIGA abbreviation being a clean substring of the fuller
        # dados.gov.pt name (or vice versa) is a strong signal on its own --
        # SequenceMatcher.ratio() length-normalizes over BOTH strings
        # combined, so it undervalues exactly this pattern. Found while
        # testing: the flagship real case this script exists for, "Geral"
        # (SIGA) vs "Atendimento Geral" (dados.gov.pt), scores only ~0.43 by
        # plain ratio() alone despite being a clean, unambiguous substring
        # match -- below any threshold that would also reject real false
        # positives. Floor of 0.7 (comfortably above MIN_NAME_SIMILARITY for
        # any substring match), rising toward 1.0 as the two lengths converge.
        shorter, longer = sorted((norm_a, norm_b), key=len)
        return 0.7 + 0.3 * (len(shorter) / len(longer))
    return difflib.SequenceMatcher(None, norm_a, norm_b).ratio()


def extract_siga_service_names(siga_locations: list[dict]) -> list[str]:
    names: dict[str, None] = {}  # dict, not set -- preserves first-seen order for stable/reproducible output
    for location in siga_locations:
        name = location.get("servico", {}).get("nome")
        if name:
            names[name] = None
    return list(names.keys())


def match_services(siga_service_names: list[str], vocabulary: tuple[str, ...] = DESK_SERVICES) -> list[dict]:
    """Only accepts structurally safe matches -- a normalized-exact match, or
    a clean substring containment. A merely "similar-looking" name is
    rejected outright, no matter how high its ratio() score.

    Found 2026-07-30 by inspecting the first real run's output: pure string
    similarity cannot distinguish "an abbreviation of the same service" from
    "a different service that happens to read similarly", and it produced
    four genuinely wrong mappings at 0.667-0.909 confidence -- 'Atendimento
    email' -> 'Atendimento EMEL' (an email desk vs. Lisbon's parking
    authority), 'Chamadas efetuadas' -> 'Chamadas recebidas' (outgoing vs.
    incoming calls, semantic opposites), 'Município de Tondela' -> 'Câmara
    Municipal de Odivelas' (two different municipalities), and
    'Licenciamento' -> 'Licenciamento de Festas' (generic licensing
    narrowed to party licensing). Silently merging two distinct real
    services under one key is far worse than leaving a combo unreconciled:
    an unreconciled combo just doesn't get proxy-decay (the status quo,
    a missed improvement), while a wrong merge actively pollutes a real
    service's training data with another service's readings.

    No threshold tuning fixes this -- 'Atendimento email'/'Atendimento
    EMEL' scored 0.909, higher than several correct matches -- so the rule
    is structural rather than score-based. Substring containment is the
    safe case because it means one name is literally the other plus
    qualifiers ('Câmara Municipal - Atendimento Geral' contains
    'Atendimento Geral'), which is exactly the SIGA-prefixes-a-department
    pattern this script exists to reconcile.
    """
    crosswalk: list[dict] = []

    for siga_name in siga_service_names:
        if siga_name in vocabulary:
            continue  # already an exact match -- no crosswalk entry needed

        normalized_siga = normalize_name(siga_name)
        best_match, best_similarity = None, 0.0

        for canonical_name in vocabulary:
            normalized_canonical = normalize_name(canonical_name)
            is_safe = normalized_siga == normalized_canonical or normalized_canonical in normalized_siga
            if not is_safe:
                continue
            similarity = name_similarity(siga_name, canonical_name)
            if similarity > best_similarity:
                best_match, best_similarity = canonical_name, similarity

        if best_match is None or best_similarity < MIN_NAME_SIMILARITY:
            logger.debug("No safe vocabulary match for SIGA service '%s'", siga_name)
            continue

        crosswalk.append(
            {
                "siga_service_name": siga_name,
                "canonical_service_name": best_match,
                "match_confidence": round(best_similarity, 3),
            }
        )

    return crosswalk


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile SIGA desk-service names against the dados.gov.pt vocabulary")
    parser.add_argument("--discovered", default=DISCOVERED_LOCATIONS_PATH)
    parser.add_argument("--out", default=SERVICE_CROSSWALK_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    siga_locations = json.loads(Path(args.discovered).read_text())
    siga_service_names = extract_siga_service_names(siga_locations)
    crosswalk = match_services(siga_service_names)

    Path(args.out).write_text(json.dumps(crosswalk, indent=2, ensure_ascii=False))

    already_exact = sum(1 for name in siga_service_names if name in DESK_SERVICES)
    unmatched = [
        name for name in siga_service_names if name not in DESK_SERVICES and name not in {e["siga_service_name"] for e in crosswalk}
    ]
    logger.info(
        "%d SIGA service names: %d already exact matches, %d newly reconciled, %d still unmatched",
        len(siga_service_names), already_exact, len(crosswalk), len(unmatched),
    )
    if unmatched:
        logger.info("Unmatched SIGA service names (proxy-decay still won't apply): %s", unmatched)


if __name__ == "__main__":
    main()
