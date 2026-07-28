"""Ingests the real "Indicadores dos Atendimentos das Lojas de Cidadão - Mensal"
(IALC-M) open dataset from dados.gov.pt: lists available monthly resources via
the dados.gov.pt REST API, downloads the most recent N months as .xlsx, and
cleans them into a single canonical DataFrame.

Verified against the live API (2026-07-27): dataset id
`indicadores-dos-atendimentos-das-lojas-de-cidadao-mensal`, 115 monthly .xlsx
resources spanning 2017-01 through the current month, columns
Data/Loja/Distrito/Concelho/Total_Senhas/Total_Atendimentos/Total_Desistencias/
Tempo_Medio_Espera_Min/Tempo_Medio_Atendimento_Min.

IMPORTANT — how this relates to pipeline/load_historical.py's SLC-M dataset:
these are two *different* datasets from the same source. SLC-M has
attendance counts per (store, service, day) but no wait-time field at all.
IALC-M has **real measured** `Tempo_Medio_Espera_Min` (avg wait until service
starts) and `Tempo_Medio_Atendimento_Min` (avg service duration) per
(store, day) — but no per-service breakdown. Cross-checked 2026-07-27: for
the 24-month overlap, summing SLC-M's per-service Atendimentos by (branch,
day) matches IALC-M's Total_Atendimentos for the same (branch, day) at
100.00% exact agreement (35,773/35,773 rows, correlation 1.0) — same
underlying system, same counting rules, branch names reconcile 1:1 via the
same slugify() used elsewhere. So the two are safe to combine.

Real data caveat found during the same audit: many rows carry very low
`Total_Atendimentos` (some branch-days have exactly 1), making that day's
"average" wait time a single noisy raw observation rather than a smoothed
mean. Downstream consumers should weight rows by their own Total_Atendimentos
rather than trust every row equally — see pipeline/calibrate_constants.py and
pipeline/demand_baseline.py's per-combo weighting for how this is applied.

WHY 36 MONTHS, NOT THE FULL 115 AVAILABLE (2026-07-27): considered and
rejected as overkill — see pipeline/load_historical.py's module docstring
for the full reasoning (COVID-era contamination in 2020-2021, diminishing
value beyond 2-3 annual cycles, and this isn't even the project's real
bottleneck — siga_live's live coverage is). Must be pulled for the same
range as load_historical.py's SLC-M — pipeline/ingest_real_wait_times.py
inner-joins the two, so extending one without the other gains nothing.

Usage:
    python -m pipeline.load_ialc                 # fetch latest 36 months from the real API
    python -m pipeline.load_ialc --months 115     # fetch full 2017-2026 history (not recommended, see above)
    python -m pipeline.load_ialc --skip-download  # only re-parse whatever is already in data/ialc_raw
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
import requests

from pipeline.geocode_branches import slugify

logger = logging.getLogger(__name__)

DATASET_API_URL = "https://dados.gov.pt/api/1/datasets/indicadores-dos-atendimentos-das-lojas-de-cidadao-mensal"
IALC_RAW_DIR = "data/ialc_raw"
CLEANED_OUTPUT_PATH = "data/cleaned_ialc_baseline.parquet"
REQUEST_TIMEOUT_SECONDS = 30

RESOURCE_MONTH_PATTERN = re.compile(r"(\d{4})(\d{2})")

CANONICAL_COLUMNS = [
    "date",
    "branch_id",
    "store_name",
    "district",
    "municipality",
    "total_senhas",
    "total_attendances",
    "total_desistencias",
    "avg_wait_minutes",
    "avg_service_minutes",
]

RAW_COLUMN_MAP = {
    "Data": "date",
    "Loja": "store_name",
    "Distrito": "district",
    "Concelho": "municipality",
    "Total_Senhas": "total_senhas",
    "Total_Atendimentos": "total_attendances",
    "Total_Desistencias": "total_desistencias",
    "Tempo_Medio_Espera_Min": "avg_wait_minutes",
    "Tempo_Medio_Atendimento_Min": "avg_service_minutes",
}


def list_available_resources() -> list[dict]:
    """Returns dataset resources sorted newest-first, each as
    {year, month, url, title}. Mirrors pipeline/load_historical.py's approach
    since both datasets are served by the same dados.gov.pt platform."""
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


DOWNLOAD_RETRY_ATTEMPTS = 3


def download_resource(resource: dict, raw_dir: str) -> Path:
    """Downloads one monthly resource, skipping the request if already cached
    locally — except for the current calendar month, which the source
    updates daily and so is always re-fetched.

    dados.gov.pt intermittently read-times-out on an otherwise-valid request
    (observed 2026-07-27: 4 of 24 requests in one run) — retried transiently
    rather than silently dropping that month from the cleaned output, which
    would otherwise pass silently as a smaller-than-expected row count.
    """
    dest = Path(raw_dir) / resource["title"]
    dest.parent.mkdir(parents=True, exist_ok=True)

    now = pd.Timestamp.now()
    is_current_month = resource["year"] == now.year and resource["month"] == now.month
    if dest.exists() and not is_current_month:
        return dest

    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_RETRY_ATTEMPTS + 1):
        try:
            response = requests.get(resource["url"], timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            dest.write_bytes(response.content)
            logger.info("Downloaded %s (%d bytes)", dest.name, len(response.content))
            return dest
        except requests.exceptions.RequestException as error:
            last_error = error
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, DOWNLOAD_RETRY_ATTEMPTS, resource["title"], error)

    raise RuntimeError(f"Failed to download {resource['title']} after {DOWNLOAD_RETRY_ATTEMPTS} attempts") from last_error


def fetch_latest_months(months: int, raw_dir: str = IALC_RAW_DIR) -> list[Path]:
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
        missing = set(RAW_COLUMN_MAP.values()) - set(cleaned.columns)
        if missing:
            logger.warning("Skipping %s: missing columns %s", path.name, missing)
            continue
        frames.append(cleaned[list(RAW_COLUMN_MAP.values())])

    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.date
    for text_column in ("store_name", "district", "municipality"):
        combined[text_column] = combined[text_column].astype(str).str.strip()
    combined["branch_id"] = combined["store_name"].apply(slugify)

    numeric_columns = ["total_senhas", "total_attendances", "total_desistencias", "avg_wait_minutes", "avg_service_minutes"]
    for column in numeric_columns:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")

    combined = combined.dropna(subset=["avg_wait_minutes", "avg_service_minutes"])
    combined = combined.drop_duplicates(subset=["date", "branch_id"])
    return combined[CANONICAL_COLUMNS]


def save_cleaned(frame: pd.DataFrame, output_path: str = CLEANED_OUTPUT_PATH) -> str:
    try:
        frame.to_parquet(output_path, index=False)
        return output_path
    except Exception:
        logger.warning("Parquet write failed (missing pyarrow/fastparquet?); falling back to CSV", exc_info=True)
        csv_path = str(Path(output_path).with_suffix(".csv"))
        frame.to_csv(csv_path, index=False)
        return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the real dados.gov.pt IALC-M wait-time indicators dataset")
    parser.add_argument("--months", type=int, default=36, help="Number of most recent monthly files to fetch")
    parser.add_argument("--raw-dir", default=IALC_RAW_DIR)
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
    logger.info(
        "Cleaned IALC-M dataset: %d rows spanning %s to %s across %d branches",
        len(cleaned),
        cleaned["date"].min(),
        cleaned["date"].max(),
        cleaned["branch_id"].nunique(),
    )

    output_path = save_cleaned(cleaned)
    logger.info("Saved cleaned IALC-M baseline to %s", output_path)


if __name__ == "__main__":
    main()
