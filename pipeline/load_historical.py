"""Ingests the real "Serviços das Lojas de Cidadão - Mensal" open dataset from
dados.gov.pt: lists available monthly resources via the dados.gov.pt REST API,
downloads the most recent N months as .xlsx, and cleans them into a single
canonical DataFrame.

Verified against the live API (2026-07-26): dataset id
`servicos-das-lojas-de-cidadao-mensal`, 115+ monthly .xlsx resources back to
2017, columns Data/Distrito/Concelho/Loja/Servico/Atendimentos, ~19k rows per
month across 77 stores and ~180 distinct service labels.

IMPORTANT: this dataset reports *daily attendance counts* (Atendimentos) per
store/service — it has no queue length or wait-time field, and no
intra-day timestamp. It is not a substitute for live SIGA polling as a
ground-truth wait_time_minutes source. `pipeline/demand_baseline.py` uses the
cleaned output from this module two ways: (a) as a genuine historical
demand-baseline feature, and (b) to derive an approximate, clearly-tagged
wait-time proxy for bootstrap training until real measured wait times
accumulate from live scraping.

Usage:
    python -m pipeline.load_historical                 # fetch latest 24 months from the real API
    python -m pipeline.load_historical --months 12
    python -m pipeline.load_historical --skip-download  # only re-parse whatever is already in data/historical_raw
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATASET_API_URL = "https://dados.gov.pt/api/1/datasets/servicos-das-lojas-de-cidadao-mensal"
HISTORICAL_RAW_DIR = "data/historical_raw"
CLEANED_OUTPUT_PATH = "data/cleaned_historical_baseline.parquet"
KNOWN_STORES_PATH = "data/known_stores.csv"
KNOWN_SERVICES_PATH = "data/known_desk_services.csv"
REQUEST_TIMEOUT_SECONDS = 30

RESOURCE_MONTH_PATTERN = re.compile(r"(\d{4})(\d{2})")

CANONICAL_COLUMNS = ["date", "district", "municipality", "store_name", "service_type", "total_attendances"]

# The dataset has used both accented and unaccented headers across years.
RAW_COLUMN_MAP = {
    "Data": "date",
    "Distrito": "district",
    "Concelho": "municipality",
    "Loja": "store_name",
    "Servico": "service_type",
    "Serviço": "service_type",
    "Atendimentos": "total_attendances",
}


def list_available_resources() -> list[dict]:
    """Returns dataset resources sorted newest-first, each as
    {year, month, url, title}. Resources whose title doesn't contain a
    YYYYMM stamp are skipped (defensive against unrelated attachments)."""
    response = requests.get(DATASET_API_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    resources = response.json().get("resources", [])

    parsed = []
    for resource in resources:
        match = RESOURCE_MONTH_PATTERN.search(resource.get("title", ""))
        if not match:
            continue
        year, month = int(match.group(1)), int(match.group(2))
        parsed.append({"year": year, "month": month, "url": resource["url"], "title": resource["title"]})

    parsed.sort(key=lambda item: (item["year"], item["month"]), reverse=True)
    return parsed


def download_resource(resource: dict, raw_dir: str) -> Path:
    """Downloads one monthly resource, skipping the request if already cached
    locally — except for the current calendar month, which the source
    updates daily and so is always re-fetched."""
    dest = Path(raw_dir) / resource["title"]
    dest.parent.mkdir(parents=True, exist_ok=True)

    now = pd.Timestamp.now()
    is_current_month = resource["year"] == now.year and resource["month"] == now.month
    if dest.exists() and not is_current_month:
        return dest

    response = requests.get(resource["url"], timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    dest.write_bytes(response.content)
    logger.info("Downloaded %s (%d bytes)", dest.name, len(response.content))
    return dest


def fetch_latest_months(months: int, raw_dir: str = HISTORICAL_RAW_DIR) -> list[Path]:
    resources = list_available_resources()[:months]
    if not resources:
        logger.warning("No resources found on dados.gov.pt for this dataset")
        return []

    paths = []
    for resource in resources:
        try:
            paths.append(download_resource(resource, raw_dir))
        except Exception:
            logger.exception("Failed to download %s", resource.get("title"))
    return paths


def parse_and_clean(xlsx_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in xlsx_paths:
        try:
            raw = pd.read_excel(path)
        except Exception:
            logger.exception("Failed to read %s", path)
            continue

        cleaned = raw.rename(columns=RAW_COLUMN_MAP)
        missing = set(CANONICAL_COLUMNS) - set(cleaned.columns)
        if missing:
            logger.warning("Skipping %s: missing columns %s", path.name, missing)
            continue
        frames.append(cleaned[CANONICAL_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.date
    for text_column in ("district", "municipality", "store_name", "service_type"):
        combined[text_column] = combined[text_column].astype(str).str.strip()
    combined["total_attendances"] = pd.to_numeric(combined["total_attendances"], errors="coerce").fillna(0).astype(int)
    combined = combined.drop_duplicates(subset=["date", "store_name", "service_type"])
    return combined


def save_cleaned(frame: pd.DataFrame, output_path: str = CLEANED_OUTPUT_PATH) -> str:
    try:
        frame.to_parquet(output_path, index=False)
        return output_path
    except Exception:
        logger.warning("Parquet write failed (missing pyarrow/fastparquet?); falling back to CSV", exc_info=True)
        csv_path = str(Path(output_path).with_suffix(".csv"))
        frame.to_csv(csv_path, index=False)
        return csv_path


def save_known_dimensions(frame: pd.DataFrame) -> None:
    """Writes the unique store and service-type dimensions seen in the
    cleaned data — inputs to pipeline/geocode_branches.py (branch registry)
    and the desk_service_id taxonomy in config.py."""
    stores = frame[["store_name", "municipality", "district"]].drop_duplicates().sort_values("store_name")
    stores.to_csv(KNOWN_STORES_PATH, index=False)
    logger.info("Wrote %d unique stores to %s", len(stores), KNOWN_STORES_PATH)

    services = pd.DataFrame({"service_type": sorted(frame["service_type"].unique())})
    services.to_csv(KNOWN_SERVICES_PATH, index=False)
    logger.info("Wrote %d unique service types to %s", len(services), KNOWN_SERVICES_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the real dados.gov.pt Lojas de Cidadao attendance dataset")
    parser.add_argument("--months", type=int, default=24, help="Number of most recent monthly files to fetch")
    parser.add_argument("--raw-dir", default=HISTORICAL_RAW_DIR)
    parser.add_argument(
        "--skip-download", action="store_true", help="Only re-parse whatever .xlsx files already exist in --raw-dir"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.skip_download:
        xlsx_paths = sorted(Path(args.raw_dir).glob("*.xlsx"))
    else:
        xlsx_paths = fetch_latest_months(args.months, args.raw_dir)

    if not xlsx_paths:
        raise SystemExit(f"No .xlsx files available in {args.raw_dir}; check network access or drop files there manually")

    cleaned = parse_and_clean(xlsx_paths)
    if cleaned.empty:
        raise SystemExit("Parsed 0 usable rows from the downloaded files")
    logger.info("Cleaned dataset: %d rows spanning %s to %s", len(cleaned), cleaned["date"].min(), cleaned["date"].max())

    output_path = save_cleaned(cleaned)
    logger.info("Saved cleaned historical baseline to %s", output_path)

    save_known_dimensions(cleaned)


if __name__ == "__main__":
    main()
