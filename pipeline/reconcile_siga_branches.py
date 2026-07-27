"""Matches real SIGA locations (data/siga_discovered_locations.json, from
pipeline/siga_discovery.py) against this project's existing branch registry
(config.BRANCHES, geocoded from the *different* dados.gov.pt dataset).

The two sources don't share a naming convention — confirmed concretely, e.g.
dados.gov.pt calls one branch "Loja de Cidadão das Laranjeiras" while SIGA
calls the same physical location "Loja de Cidadão Laranjeiras" (no "das").
So this is a fuzzy match (name similarity after stripping the common
"Loja de Cidadão ..." prefix, cross-checked against distance when both sides
have real coordinates), not a trivial slug/exact-string join.

Output: data/siga_branch_crosswalk.json — a list of
{branch_id, siga_location_id, siga_name, match_confidence, distance_km}.
Every match is scored, not asserted — pipeline/siga_scraper.py should still
log which branches have no (or only low-confidence) SIGA match, since those
can't get live data yet.

Usage:
    python -m pipeline.reconcile_siga_branches
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import unicodedata
from pathlib import Path

from api.service import haversine_km
from config import BRANCHES

logger = logging.getLogger(__name__)

DISCOVERED_LOCATIONS_PATH = "data/siga_discovered_locations.json"
CROSSWALK_PATH = "data/siga_branch_crosswalk.json"

# Loose bounding box (mainland + islands) for sanity-checking SIGA-provided
# coordinates. Found by testing, not assumption: at least one real SIGA
# record has latitude/longitude transposed (Barreiro: {"latitude": -9.07,
# "longitude": 38.66} — clearly swapped, since -9.07 is a real Portuguese
# longitude and 38.66 a real Portuguese latitude), which produces a
# nonsensical multi-thousand-km "distance" if trusted at face value.
PORTUGAL_LAT_RANGE = (32.0, 43.0)
PORTUGAL_LON_RANGE = (-32.0, -6.0)

# Below this fuzzy name-similarity score, a candidate is never considered.
MIN_NAME_SIMILARITY = 0.55
# At or above this similarity, trust the name match even if coordinates
# disagree — an (almost) exact name match is far more likely to mean our own
# geocoded coordinate is imprecise (e.g. municipality-level fallback) than a
# coincidental name clash.
HIGH_CONFIDENCE_NAME_SIMILARITY = 0.85
# Below HIGH_CONFIDENCE_NAME_SIMILARITY, a name match this far from our own
# coordinate is rejected outright — mediocre name similarity plus a wildly
# wrong distance is a false positive (e.g. two different branches both named
# "Vila ..."), not a real match with a coordinate quirk.
MAX_TRUSTED_DISTANCE_KM = 10.0

_PREFIX_PATTERN = re.compile(r"^loja\s+de\s+cidad[ãa]o\s+(d[ao]s?\s+)?", re.IGNORECASE)


def normalize_name(name: str) -> str:
    stripped = _PREFIX_PATTERN.sub("", name.strip())
    normalized = unicodedata.normalize("NFKD", stripped).encode("ascii", "ignore").decode("ascii")
    return normalized.lower().strip()


def name_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()


def dedupe_locations(locations: list[dict]) -> list[dict]:
    """The same physical SIGA location appears once per (entidade, senha) it
    offers, so dedupe by SIGA's own numeric location id before matching."""
    seen: dict[int, dict] = {}
    for location in locations:
        seen[location["id"]] = location
    return list(seen.values())


def _has_plausible_coords(location: dict) -> bool:
    lat, lon = location.get("latitude"), location.get("longitude")
    if not lat or not lon:
        return False
    return PORTUGAL_LAT_RANGE[0] <= lat <= PORTUGAL_LAT_RANGE[1] and PORTUGAL_LON_RANGE[0] <= lon <= PORTUGAL_LON_RANGE[1]


def _distance_km(branch, location: dict) -> float | None:
    if not _has_plausible_coords(location):
        return None
    return round(haversine_km(branch.latitude, branch.longitude, location["latitude"], location["longitude"]), 2)


def match_branches(siga_locations: list[dict], branches=BRANCHES) -> list[dict]:
    crosswalk: list[dict] = []
    unique_locations = dedupe_locations(siga_locations)

    for branch in branches:
        candidates = sorted(
            ((name_similarity(branch.name, loc["nome"]), loc) for loc in unique_locations),
            key=lambda pair: pair[0],
            reverse=True,
        )

        accepted: dict | None = None
        accepted_similarity = 0.0
        accepted_distance: float | None = None

        for similarity, location in candidates:
            if similarity < MIN_NAME_SIMILARITY:
                break  # sorted descending — nothing further will pass either

            distance_km = _distance_km(branch, location)

            if similarity >= HIGH_CONFIDENCE_NAME_SIMILARITY:
                accepted, accepted_similarity, accepted_distance = location, similarity, distance_km
                break

            if distance_km is None or distance_km <= MAX_TRUSTED_DISTANCE_KM:
                accepted, accepted_similarity, accepted_distance = location, similarity, distance_km
                break

            logger.debug(
                "Rejecting candidate '%s' for branch '%s': similarity %.2f but %.1fkm apart",
                location["nome"], branch.name, similarity, distance_km,
            )

        if accepted is None:
            logger.warning("No confident SIGA match for branch '%s'", branch.name)
            continue

        if accepted_distance is not None and accepted_distance > MAX_TRUSTED_DISTANCE_KM:
            logger.info(
                "Branch '%s' matched SIGA '%s' by near-exact name (similarity %.2f) despite %.1fkm coordinate "
                "discrepancy — likely our own geocoded coordinate is imprecise, not a wrong match",
                branch.name, accepted["nome"], accepted_similarity, accepted_distance,
            )

        coords_plausible = _has_plausible_coords(accepted)
        crosswalk.append(
            {
                "branch_id": branch.branch_id,
                "branch_name": branch.name,
                "siga_location_id": accepted["id"],
                "siga_name": accepted["nome"],
                "siga_morada": accepted.get("morada"),
                # Null'd out (not just passed through) when outside a
                # plausible Portugal bounding box, e.g. the Barreiro record's
                # transposed lat/lon — never hand a bogus coordinate to a
                # downstream consumer as if it were trustworthy.
                "siga_latitude": accepted.get("latitude") if coords_plausible else None,
                "siga_longitude": accepted.get("longitude") if coords_plausible else None,
                "match_confidence": round(accepted_similarity, 3),
                "distance_km": accepted_distance,
            }
        )

    return crosswalk


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile SIGA locations against the existing branch registry")
    parser.add_argument("--discovered", default=DISCOVERED_LOCATIONS_PATH)
    parser.add_argument("--out", default=CROSSWALK_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    siga_locations = json.loads(Path(args.discovered).read_text())
    crosswalk = match_branches(siga_locations)

    Path(args.out).write_text(json.dumps(crosswalk, indent=2, ensure_ascii=False))

    matched_branch_ids = {entry["branch_id"] for entry in crosswalk}
    unmatched = [b for b in BRANCHES if b.branch_id not in matched_branch_ids]
    flagged_distance = [entry for entry in crosswalk if entry["distance_km"] and entry["distance_km"] > MAX_TRUSTED_DISTANCE_KM]

    logger.info(
        "Matched %d/%d branches to a SIGA location (%d flagged for distance mismatch, %d unmatched)",
        len(crosswalk), len(BRANCHES), len(flagged_distance), len(unmatched),
    )
    if unmatched:
        logger.info("Unmatched branches (no live data possible yet): %s", [b.name for b in unmatched])


if __name__ == "__main__":
    main()
