# Loja do Cidadão — Wait Time Predictor

Predicts wait times at Portugal's "Loja do Cidadão" branches and public service
desks using a tabular XGBoost regression model. No LLM calls at inference time —
predictions are deterministic, sub-50ms, and served from a `.joblib` artifact.

This project is self-contained under `loja-cidadao/` and is unrelated to the
SNS health-data website living at the repo root — do not mix files between them.

## Architecture

- **Real historical ingestion**: `pipeline/load_historical.py` pulls the real
  "Serviços das Lojas de Cidadão - Mensal" dataset from the dados.gov.pt REST
  API (daily attendance counts per store/service, not wait times — see
  "Key data sources" below). `pipeline/geocode_branches.py` turns its 78 real
  store names into a geocoded branch registry. `pipeline/demand_baseline.py`
  turns the cleaned attendance data into (a) a real demand-baseline feature
  and (b) an approximate, clearly-tagged wait-time proxy label for bootstrap
  training.
- **Live scraper**: `scrapers/siga_scraper.py` polls the real, verified public
  SIGA API (`siga.marcacaodeatendimento.pt`) — see `pipeline/siga_client.py`.
  A full national crawl of that API is a genuine 3-level nest (district x
  entidade x senha), so `pipeline/siga_discovery.py` finds which queries cover
  real Loja de Cidadão locations once, and `pipeline/reconcile_siga_branches.py`
  fuzzy-matches those locations against the existing branch registry (the two
  data sources don't share a naming convention). The scraper itself only
  polls that reconciled, reduced query set.
- **Pipeline**: pandas feature engineering (calendar, tax deadlines,
  Open-Meteo weather, real demand baseline, `is_open`, lag/rolling wait-time
  averages) feeding an `XGBRegressor`.
- **API**: FastAPI + Uvicorn loads the trained `.joblib` model once at startup
  and serves predictions with sub-50ms latency. A closed branch short-circuits
  to a deterministic zero-wait response (`surge_level="closed"`) without
  running the model, and reroute suggestions exclude closed candidates.
- **Tests**: pytest covers API schema, the real ingestion/cleaning logic, and
  graceful-degradation fallback paths.

## Directory layout

```
loja-cidadao/
├── data/                    # raw downloads, cleaned baseline, queue_history.db
│   ├── historical_raw/           # downloaded dados.gov.pt .xlsx files
│   ├── branches_registry.json    # geocoded real branch registry (generated)
│   ├── known_stores.csv          # unique real store names (generated)
│   ├── known_desk_services.csv   # unique real service labels (generated)
│   ├── cleaned_historical_baseline.parquet  # cleaned attendance data (generated)
│   ├── siga_relevant_queries.json     # (distrito,entidade,senha) triples to poll (generated)
│   ├── siga_discovered_locations.json # raw matched SIGA locations (generated)
│   └── siga_branch_crosswalk.json     # branch_id <-> siga_location_id (generated)
├── scrapers/                # live SIGA polling script
├── pipeline/                # ingestion, discovery, feature engineering, training
├── models/                  # serialized .joblib model artifacts
├── api/                     # FastAPI backend routes
├── tests/                   # pytest coverage
├── config.py                # branch registry, desk services, constants
├── schemas.py                # shared dataclasses
├── requirements.txt
└── CLAUDE.md
```

`config.py` loads `data/branches_registry.json` and `data/known_desk_services.csv`
at import time if present, falling back to a tiny 3-branch placeholder set
otherwise (e.g. on a fresh checkout before the real pipeline has run).

## Environment setup

```bash
cd loja-cidadao
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

macOS only: XGBoost needs the OpenMP runtime, which isn't bundled.
If `import xgboost` fails with `Library not loaded: @rpath/libomp.dylib`, run
`brew install libomp` once.

Note: the original spec's `open-meteo-sdk` package does not exist on PyPI.
`requirements.txt` uses the actual official client, `openmeteo-requests`
(plus its companions `requests-cache` and `retry-requests`), which is what
`pipeline/feature_engineering.py`'s `WeatherClient` wraps.

Scripts are run as modules (`python -m pipeline.train`, not
`python pipeline/train.py`) so `pipeline/`, `scrapers/`, and `api/` can all
import the shared top-level `config.py` / `schemas.py` regardless of which
file is the entry point. `pytest.ini` sets `pythonpath = .` so tests get the
same resolution automatically.

## Build / run scripts

```bash
# 1. Real-data pipeline (run in this order; each step's output feeds the next)
python -m pipeline.load_historical --months 24   # fetch+clean real dados.gov.pt attendance data
python -m pipeline.geocode_branches              # geocode real store names -> branch registry (~1 req/sec, cached)
python -m pipeline.demand_baseline               # build demand-baseline feature + proxy wait-time labels

# Alternative cold-start: pure synthetic data, no network, if you just want
# to exercise the pipeline/API without the real dataset:
python -m pipeline.synthetic_bootstrap --days 30

# 2. Live SIGA scraper setup (one-off/occasional — run again only if new
#    branches appear or the API changes shape):
python -m pipeline.siga_discovery                # ~15-20 min: crawl all 18 mainland districts
python -m pipeline.reconcile_siga_branches        # match discovered locations to the branch registry

# Then poll live queue data. Two ways to run it:
python -m scrapers.siga_scraper --once                     # writes straight to data/queue_history.db (local use)
python -m scrapers.siga_scraper                             # loops forever, every 15 min

# CI (e.g. GitHub Actions, see .github/workflows/siga_scraper.yml) writes to a
# small git-tracked CSV instead of the 400MB+ local DB:
python -m scrapers.siga_scraper --once --csv-out data/live_samples.csv
git pull && python -m pipeline.import_live_samples          # merge CI's new rows into your local DB

# Check real data coverage per (branch, service) combo before trusting metrics:
python -m pipeline.coverage_report

# 3. Train the model (time-series split, prints MAE/RMSE/R²)
python -m pipeline.train

# 4. Serve the API
uvicorn api.main:app --reload --port 8000

# 5. Run tests
pytest -v
```

## Architectural rules (do not violate)

1. **No GenAI/LLM at runtime.** All `/predict` responses must come strictly
   from the exported XGBoost `.joblib` model — no LLM calls in the request path.
2. **Graceful degradation.** If weather, live queue state, or the historical
   demand baseline is unavailable, fall back to statistical baseline averages
   rather than failing the request. Never let a missing external signal 500
   the API.
3. **Modular, typed, testable.** All functions carry type hints; logic that can
   be unit-tested (feature transforms, reroute scoring) lives outside route
   handlers so it can be tested without spinning up the app.
4. **Time-series discipline.** Never shuffle temporal data for
   train/test splits — always split chronologically.
5. **Response contract.** `/predict` must always return:
   `predicted_wait_minutes`, `surge_level`, `confidence_score`,
   `recommended_alternative_store`. A closed branch reuses `surge_level`
   (value `"closed"`, `config.SURGE_CLOSED_LABEL`) rather than changing the
   response shape — never add new top-level fields for this.

## Key data sources

- **Real historical attendance**: dados.gov.pt dataset
  `servicos-das-lojas-de-cidadao-mensal` — daily attendance counts
  (`Atendimentos`) per store/service, back to 2017/2018, updated daily.
  **This is demand volume, not a wait-time measurement** — there is no queue
  length or wait-time field in this dataset, and no intra-day timestamp. It
  feeds two things: a real `historical_avg_attendances` feature, and an
  approximate `wait_time_minutes` proxy label (tagged
  `source='historical_derived_proxy'` in `queue_samples`) derived via an
  M/M/1-style queueing formula documented in `pipeline/demand_baseline.py`,
  expanded into `config.DIURNAL_SNAPSHOTS` daypart snapshots (not one flat
  timestamp) so `hour_of_day` carries real variance, with desk count scaled
  to real volume (`estimate_desks_for_volume`) rather than one flat constant.
  `pipeline/train.py` downweights these proxy rows per-(branch, service) as
  real `siga_live` coverage grows for that specific combo (see
  `compute_sample_weights` — a global cutover would blind whichever combos
  haven't accumulated live data yet, given ~2,000 distinct real combos).
  Run `python -m pipeline.coverage_report` to see real coverage per combo
  before trusting any headline metric — with a single scrape, every combo
  sits at n=1; the blended R^2 stays dominated by proxy rows until real
  coverage actually accumulates over real elapsed time, and `train.py`
  already reports metrics segmented by source (`metrics_by_source` in the
  saved artifact) so this isn't hidden behind one blended number.
- **Branch registry**: 78 real store names/districts/municipalities from the
  same dataset, geocoded via Nominatim (`pipeline/geocode_branches.py`). Most
  resolved at store-name precision; some fell back to municipality-center
  precision — see each branch's `geocode_precision` field.
- **Live queue counters**: the real, verified public SIGA API
  (`siga.marcacaodeatendimento.pt`, `POST /Senhas/GetLocais` and friends —
  see `pipeline/siga_client.py`'s docstring for exactly how this was
  confirmed). Returns real `tempoRealEspera` (wait minutes), `utentesEmEspera`
  (people waiting), and `estado` (open/closed) per location/service — no
  estimation needed for current live state. Two real data-source quirks to
  know about:
  - SIGA's own service names ("Geral", "Tesouraria") don't always match
    dados.gov.pt's fuller names ("Atendimento Geral") — stored as-is;
    `pipeline/reconcile_siga_branches.py`'s docstring covers the analogous
    branch-name mismatch this hasn't been extended to for services yet.
  - At least one SIGA location record has latitude/longitude transposed
    (Barreiro) — `pipeline/reconcile_siga_branches.py` sanity-bounds
    coordinates against a Portugal bounding box before trusting them.
  - `tempoRealEspera` is sometimes wildly implausible — found 2026-07-27,
    46% of "open" readings from one scheduled scrape showed 180-13,366
    minutes, uncorrelated with the actual `utentesEmEspera` count (many
    showed 0 people waiting alongside a 5-figure "wait"). Every reading's
    untouched value is kept in `raw_wait_time_minutes`; `estimated_wait_minutes`
    (the field training/API actually read) is filtered to
    `config.REAL_DATA_MAX_PLAUSIBLE_WAIT_MINUTES` (480 min = one business
    day). Nothing is lost — just not trusted by default until the pattern
    is understood.
- **Weather**: Open-Meteo Forecast API (rainfall `rain_mm`) by branch
  coordinates. Its `start_date`/`end_date` params only cover roughly the last
  ~110 days plus forecast — for training rows older than that (most of our
  2-year historical span), `rain_mm` silently falls back to
  `config.BASELINE_RAIN_MM` via the same graceful-degradation path used at
  inference time. To get real historical rain further back, switch to
  Open-Meteo's Historical Weather Archive API
  (`archive-api.open-meteo.com`) for dates beyond that window — not done yet.
- **Tax deadlines**: IRS and IMI calendar dates (hardcoded in `config.py`,
  update yearly).
- **is_open**: real observed value from live SIGA polling when available
  (near-now requests only — see `config.NEAR_NOW_WINDOW_MINUTES`); otherwise
  a fixed Mon-Fri/9-17 heuristic (`config.ASSUMED_BUSINESS_*`). Proxy/synthetic
  rows set this explicitly from generation-time evidence, not the heuristic —
  the real dataset includes ~15.7k Saturday rows with real nonzero
  attendance, which the Mon-Fri heuristic alone would have wrongly marked closed.

## Continuous live data collection (GitHub Actions)

`.github/workflows/siga_scraper.yml` runs the scraper on a schedule (weekday
opening/midday/afternoon slots, WEST/WET-approximate) without needing any
machine to stay on. It writes to `data/live_samples.csv`, not
`data/queue_history.db` — the local DB is 400MB+ (dominated by proxy rows)
and gitignored on purpose: GitHub hard-rejects files over 100MB, and a binary
SQLite blob re-committed every run would bloat history regardless, since git
can't delta-compress it the way it can plain text. `live_samples.csv` is
small and append-only, so repeated commits stay cheap.

To fold CI's collected rows into your local training data:
```bash
git pull
python -m pipeline.import_live_samples   # dedupes against what's already local; safe to re-run
```

## Testing

Run `pytest -v` from `loja-cidadao/` before considering any change to
`pipeline/` or `api/` complete. Tests must pass with weather/calendar/demand-
baseline data both present and missing (fallback path). The pytest fixture in
`tests/conftest.py` seeds a small synthetic subset (not the full real
registry) so the suite stays fast and fully offline — it's a no-op once the
real production DB/model already exist, as they do in this project's own
working copy after running the real pipeline above.
