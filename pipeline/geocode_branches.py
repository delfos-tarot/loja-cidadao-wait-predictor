"""Builds the real branch registry (data/branches_registry.json) from the 78
real store names discovered in the dados.gov.pt attendance dataset
(data/known_stores.csv, written by pipeline/load_historical.py).

Geocodes each store via the free OpenStreetMap Nominatim API, respecting its
usage policy: max ~1 request/second, a descriptive User-Agent, and aggressive
on-disk caching (data/geocode_cache.json) so a store is never re-geocoded
once resolved. If a specific store address doesn't resolve, falls back to
geocoding its municipality center — a coarser but still real approximation,
recorded as such in the `geocode_precision` field.

Each branch's desk_service_ids come from the real (store_name, service_type)
pairs observed in data/cleaned_historical_baseline.parquet, not a guess.

This is a one-off/periodic build step, not something the API runs at
request time. config.py loads its output at import time if present.

Usage:
    python -m pipeline.geocode_branches
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

KNOWN_STORES_PATH = "data/known_stores.csv"
CLEANED_BASELINE_PATH = "data/cleaned_historical_baseline.parquet"
GEOCODE_CACHE_PATH = "data/geocode_cache.json"
BRANCH_REGISTRY_PATH = "data/branches_registry.json"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = "loja-cidadao-wait-predictor/0.1 (research project; no production contact configured)"
NOMINATIM_MIN_INTERVAL_SECONDS = 1.1
REQUEST_TIMEOUT_SECONDS = 15


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")


def _load_cache(cache_path: str) -> dict:
    path = Path(cache_path)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_cache(cache_path: str, cache: dict) -> None:
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cache_path).write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def _nominatim_search(query: str) -> tuple[float, float] | None:
    response = requests.get(
        NOMINATIM_URL,
        params={"q": query, "format": "jsonv2", "limit": 1, "countrycodes": "pt"},
        headers={"User-Agent": NOMINATIM_USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        return None
    return float(results[0]["lat"]), float(results[0]["lon"])


def geocode_store(store_name: str, municipality: str, district: str, cache: dict) -> dict:
    """Returns {latitude, longitude, precision} using the cache when possible."""
    cache_key = f"{store_name}|{municipality}|{district}"
    if cache_key in cache:
        return cache[cache_key]

    result = None
    precision = "failed"
    try:
        coords = _nominatim_search(f"{store_name}, {municipality}, {district}, Portugal")
        time.sleep(NOMINATIM_MIN_INTERVAL_SECONDS)
        if coords is not None:
            result, precision = coords, "store"
        else:
            coords = _nominatim_search(f"{municipality}, {district}, Portugal")
            time.sleep(NOMINATIM_MIN_INTERVAL_SECONDS)
            if coords is not None:
                result, precision = coords, "municipality"
    except Exception:
        logger.exception("Geocoding failed for %s", store_name)

    entry = (
        {"latitude": result[0], "longitude": result[1], "precision": precision}
        if result is not None
        else {"latitude": None, "longitude": None, "precision": precision}
    )
    cache[cache_key] = entry
    return entry


def load_services_by_store(baseline_path: str = CLEANED_BASELINE_PATH) -> dict[str, list[str]]:
    frame = pd.read_parquet(baseline_path)
    services_by_store: dict[str, list[str]] = (
        frame.groupby("store_name")["service_type"].unique().apply(lambda values: sorted(set(values))).to_dict()
    )
    return services_by_store


def build_registry(
    known_stores_path: str = KNOWN_STORES_PATH,
    baseline_path: str = CLEANED_BASELINE_PATH,
    cache_path: str = GEOCODE_CACHE_PATH,
) -> list[dict]:
    stores = pd.read_csv(known_stores_path)
    services_by_store = load_services_by_store(baseline_path)
    cache = _load_cache(cache_path)

    branches: list[dict] = []
    seen_branch_ids: set[str] = set()
    for _, row in stores.iterrows():
        store_name, municipality, district = row["store_name"], row["municipality"], row["district"]
        branch_id = slugify(store_name)
        if branch_id in seen_branch_ids:
            logger.warning("Duplicate branch_id '%s' after slugify for '%s'; skipping", branch_id, store_name)
            continue
        seen_branch_ids.add(branch_id)

        geocode = geocode_store(store_name, municipality, district, cache)
        if geocode["latitude"] is None:
            logger.warning("Could not geocode '%s' at any precision; omitting from registry", store_name)
            continue

        branches.append(
            {
                "branch_id": branch_id,
                "name": store_name,
                "district": district,
                "municipality": municipality,
                "latitude": geocode["latitude"],
                "longitude": geocode["longitude"],
                "geocode_precision": geocode["precision"],
                "desk_service_ids": services_by_store.get(store_name, []),
            }
        )

    _save_cache(cache_path, cache)
    return branches


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the real branch registry from dados.gov.pt store names")
    parser.add_argument("--known-stores", default=KNOWN_STORES_PATH)
    parser.add_argument("--baseline", default=CLEANED_BASELINE_PATH)
    parser.add_argument("--cache", default=GEOCODE_CACHE_PATH)
    parser.add_argument("--out", default=BRANCH_REGISTRY_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    branches = build_registry(args.known_stores, args.baseline, args.cache)
    Path(args.out).write_text(json.dumps(branches, indent=2, ensure_ascii=False))

    store_precision = sum(1 for b in branches if b["geocode_precision"] == "store")
    municipality_precision = sum(1 for b in branches if b["geocode_precision"] == "municipality")
    logger.info(
        "Wrote %d branches to %s (%d geocoded at store precision, %d at municipality precision)",
        len(branches), args.out, store_precision, municipality_precision,
    )


if __name__ == "__main__":
    main()
