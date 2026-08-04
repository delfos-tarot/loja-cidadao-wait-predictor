# Loja do Cidadão — Wait Time Predictor

Predicts wait times at Portugal's "Loja do Cidadão" branches and public service
desks using a tabular XGBoost regression model. No LLM calls at inference time —
predictions are deterministic, sub-50ms, and served from a `.joblib` artifact.

This project is self-contained under `loja-cidadao/` and is unrelated to the
SNS health-data website living at the repo root — do not mix files between them.

## Architecture

- **Real historical ingestion**: `pipeline/load_historical.py` pulls the real
  "Serviços das Lojas de Cidadão - Mensal" (SLC-M) dataset from the
  dados.gov.pt REST API (daily attendance counts per store/service, not wait
  times — see "Key data sources" below). `pipeline/geocode_branches.py` turns
  its 78 real store names into a geocoded branch registry. A second, separate
  dados.gov.pt dataset, `pipeline/load_ialc.py` (IALC-M), has real measured
  daily wait/service-duration averages per branch (no per-service breakdown)
  — see "Key data sources". `pipeline/calibrate_constants.py` fits the M/M/1
  proxy formula's `avg_service_minutes` constants against that real data, and
  `pipeline/demand_baseline.py` turns the cleaned attendance data into (a) a
  real demand-baseline feature and (b) an approximate, clearly-tagged
  wait-time proxy label for bootstrap training, using the calibrated
  constants when available. `pipeline/ingest_real_wait_times.py` separately
  inserts the real IALC-M averages themselves into `queue_samples`, tagged
  `source='historical_real_daily_avg'` — real but branch-level/daily, a
  third tier between the formula-derived proxy and `siga_live`.
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
│   ├── cleaned_historical_baseline.parquet  # cleaned SLC-M attendance data (generated)
│   ├── ialc_raw/                     # downloaded dados.gov.pt IALC-M .xlsx files
│   ├── cleaned_ialc_baseline.parquet # cleaned real wait/duration data (generated)
│   ├── calibrated_service_constants.json  # fitted avg_service_minutes per category (generated)
│   ├── siga_relevant_queries.json     # (distrito,entidade,senha) triples to poll (generated)
│   ├── siga_discovered_locations.json # raw matched SIGA locations (generated)
│   ├── siga_branch_crosswalk.json     # branch_id <-> siga_location_id (generated)
│   ├── siga_desk_service_crosswalk.json  # SIGA service name -> dados.gov.pt name (generated)
│   ├── municipal_holidays_seed.json  # published feriados municipais — a CANDIDATE list; the corpus vetoes it (hand-maintained)
│   └── municipal_holidays.json       # per-branch feriado municipal, corpus-derived + seed-reconciled (generated)
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
python -m pipeline.load_historical --months 36   # fetch+clean real dados.gov.pt SLC-M attendance data (36mo default: past COVID era, 2-3 real annual cycles -- see module docstring)
python -m pipeline.geocode_branches              # geocode real store names -> branch registry (~1 req/sec, cached)
python -m pipeline.load_ialc --months 36         # fetch+clean real dados.gov.pt IALC-M wait-time data -- must match load_historical's range, see load_ialc.py docstring
python -m pipeline.calibrate_constants           # fit avg_service_minutes per category against real IALC-M waits (optional but recommended)
python -m pipeline.demand_baseline               # build demand-baseline feature + proxy wait-time labels (uses calibrated constants if present); safe to re-run, deletes its own prior rows first
python -m pipeline.ingest_real_wait_times        # insert real IALC-M branch-day averages as source='historical_real_daily_avg'; safe to re-run, deletes its own prior rows first
python -m pipeline.derive_municipal_holidays     # mine each branch's feriado municipal from IALC-M closure gaps -> data/municipal_holidays.json; re-run only when the corpus gains a year or a new branch appears

# Alternative cold-start: pure synthetic data, no network, if you just want
# to exercise the pipeline/API without the real dataset:
python -m pipeline.synthetic_bootstrap --days 30

# 2. Live SIGA scraper setup (one-off/occasional — run again only if new
#    branches appear or the API changes shape):
python -m pipeline.siga_discovery                # ~15-20 min: crawl all 18 mainland districts
python -m pipeline.reconcile_siga_branches        # match discovered locations to the branch registry
python -m pipeline.reconcile_siga_services        # match SIGA service names to the dados.gov.pt vocabulary (proxy-decay join)

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
   train/test splits — always split chronologically. Since 2026-07-27 this
   is applied **per source** (`chronological_split_by_source`), not as one
   global cutoff across all rows — a single global cutoff structurally
   excluded `siga_live` from training entirely, since it's a tiny, very
   recent sliver next to ~2 years of historical data, so every live row
   landed in the most-recent-20% test slice and none in train. Splitting
   each source along its own timeline (still fully chronological, still no
   shuffling) gave the model its first real training exposure to live
   data: 1,019 `siga_live` rows in training where there were previously 0.
   Result on retrain: `siga_live` segment R² improved from -0.183 to
   -0.044 — still negative (worse than predicting the mean), but a real,
   measured improvement from a model that had literally never seen a real
   example before. Not declared "fixed" — the live segment is still
   unreliable, and now genuinely small (255 test rows) so treat this
   number itself as high-variance, not a settled verdict.

   **Historical window extended 24 -> 36 months the same day (2023-08
   onward), for a different reason — not to fix siga_live.** Considered
   and rejected pulling the full 2017-2026 history as overkill (COVID-era
   distortion in 2020-2021, diminishing value past 2-3 annual cycles — see
   `pipeline/load_historical.py`'s module docstring). Confirmed
   empirically after retraining on the larger corpus: `siga_live` segment
   R² moved -0.044 -> -0.074 — statistically flat (same 255 test rows,
   noise-level difference), exactly as predicted, because SLC-M/IALC-M
   history and siga_live are different sources — growing one doesn't move
   the other. `historical_real_daily_avg` grew 543K -> 862K rows and its
   own R² held steady (~0.48-0.50). The lesson: this was a real, bounded
   improvement to calibration/demand-baseline robustness, not a fix for
   the actual bottleneck — only sustained siga_live scraper runtime moves
   that number.
5. **Response contract.** `/predict` must always return:
   `predicted_wait_minutes`, `surge_level`, `confidence_score`,
   `recommended_alternative_store`. A closed branch reuses `surge_level`
   (value `"closed"`, `config.SURGE_CLOSED_LABEL`) rather than changing the
   response shape — never add new top-level fields for this.
6. **`confidence_score` reflects real support, not just live-signal
   freshness.** Found 2026-07-27: with only 3 daily snapshot anchors
   (9:30/12:30/15:30), a same-week day-of-week ranking that was correct
   (monotonic with real demand) at 12:30 came out scrambled at 10-11am on
   the identical branch/service — the model had far less real support away
   from those hours. Two changes address this:
   - `config.DIURNAL_SNAPSHOTS` was densified from 3 to 8 anchors (hourly,
     9:30-16:30) and the model retrained — re-checked afterward across all
     8 hours: the day-of-week ranking is now stable and matches real demand
     direction everywhere it produces a non-degenerate (non-zero) value.
   - `api/service.py`'s `minutes_to_nearest_snapshot` still discounts
     `confidence_score` (`config.SNAPSHOT_DISTANCE_CONFIDENCE_PENALTY`) for
     requests far from any anchor — kept as a permanent signal (there will
     always be *some* gap between "hours with real support" and "hours a
     user asks about"), not just a stopgap for the old 3-anchor sparsity.
     **`config.SNAPSHOT_PROXIMITY_MINUTES` must scale with anchor
     spacing** — it was tuned to 30 min against 3-hour-apart anchors, and
     densifying to hourly anchors without lowering it would have made the
     discount fire for literally zero in-business-hours queries (max
     possible distance to an hourly anchor is exactly 30 min, at the
     midpoint) — silently dead code passing all its own tests. Now 10 min
     (same ~1/6-of-spacing ratio as before).

   **OPEN FINDING (2026-07-30) — `DIURNAL_SNAPSHOTS`' volume factors
   appear inverted relative to reality; do not trust hour-of-day
   predictions until resolved.** Those factors were always a documented
   hand-drawn approximation (SLC-M has no intra-day timestamp, so nothing
   in the historical data could ever validate them). Live SIGA
   `people_waiting` counts are the first real intra-day evidence this
   project has had, and they disagree sharply: normalized per (branch,
   service) across 71 branches, the observed peak is **10am local (1.67)**
   and the observed trough is **1pm (0.63)**, while the assumed curve puts
   its peak at 12-13h (1.35) and its lowest weight at 9am (0.70).
   Correlation between assumed and observed: **-0.79**. Practical
   consequence: asked "what's the best hour to go", the model currently
   recommends approximately the busiest real time. `people_waiting` was
   used deliberately rather than `tempoRealEspera` — the latter climbs
   monotonically 36 -> 75 min across the day, the signature of the
   stale-counter pathology, not a real queue.

   Deliberately **not** acted on yet: only ~3 days of live data support
   it, a midday `people_waiting` dip could plausibly reflect staff lunch
   breaks cutting capacity rather than lower demand, and recalibration is
   high-blast-radius (regenerates every proxy label, invalidates
   `calibrate_constants.py`'s fitted output). When revisited, recalibrate
   against real `people_waiting` rather than re-guessing the shape by
   hand. Full numbers in `config.py` above `DIURNAL_SNAPSHOTS`.

   **Found and fixed the same day:** predictions for genuinely far-future
   dates — where `rain_mm`/`people_waiting`/rolling-wait-stats all fall
   back to baseline simultaneously — came out erratic: several anchor
   hours clamped to exactly 0.0 (raw model output went negative there) for
   the same branch/day that looked sane on a date inside the real training
   range. Root cause: `people_waiting`'s fallback was a hardcoded `0`
   regardless of `historical_avg_attendances` — pairing "zero people
   waiting" with a real high-volume branch's busy-midday average is a
   feature combination that never occurs in training (both are derived
   from the same attendance count in `pipeline/demand_baseline.py`, so a
   real busy-midday row always has correspondingly high `people_waiting`
   too), and the model extrapolated erratically off that out-of-distribution
   input. Fixed in `api/service.py`'s `estimate_people_waiting`: the
   fallback now derives an estimate from `historical_avg_attendances` and
   the nearest snapshot's `volume_factor` — the same
   `avg_hourly_attendance * volume_factor` calculation
   `pipeline/demand_baseline.py` uses to build training labels in the
   first place, so the fallback feature vector looks like something the
   model actually saw. Confirmed: the same branch/hours that clamped to
   0.0 now form a smooth, physically plausible bell curve (low morning →
   sharp midday peak → low evening) for every day of the week. This was a
   pure inference-time fallback fix — no retrain needed, the model itself
   didn't change.

## Key data sources

- **Real historical attendance (SLC-M)**: dados.gov.pt dataset
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
  The formula's `avg_service_minutes` constants are fit against real IALC-M
  wait times when available — see the IALC-M bullet below and
  `pipeline/calibrate_constants.py`.
  **These proxy labels were RETIRED from training on 2026-08-01
  (`config.PROXY_LABEL_TRAINING_WEIGHT = 0.0`) — read that constant's comment
  before reinstating them.** They were a bootstrap for a project with no real
  wait data; IALC-M ended that on 2026-07-27. A controlled ablation (one
  snapshot, one split, test set = real rows only, n=181,381) found that
  **removing 8.6x the training data made the model better on real
  measurements**: MAE 7.955 -> 7.804, RMSE 18.41 -> 17.29, R² 0.484 -> 0.545.
  The feared failure mode did not occur — segmented by real-coverage, the
  *thinnest* combos improved most (0-real-row combos 13.17 -> 7.95 MAE, though
  n=154 there, so directional). The "coverage insurance" case for keeping them
  is empty: of 2,303 (branch, service) combos exactly **one** has proxy rows
  without real rows, while 301 have real rows without proxy. Retiring the tier
  also removes `DIURNAL_SNAPSHOTS`' inverted hour curve from training at the
  root — no recalibration, no label regeneration. **Honest cost:** `hour_of_day`
  is now supported almost entirely by `siga_live` (~45k rows over days), since
  `historical_real_daily_avg` spreads across hours by design and teaches no hour
  relationship. Thin-and-real was chosen over abundant-and-inverted.
  Nothing is deleted — rows stay in the DB, `demand_baseline.py` still
  generates them, and the real `historical_demand_baseline` attendance feature
  is untouched. `train.py` logs the exclusion on every run so it cannot go
  silent, and drops the rows *before* feature engineering — which also means
  `rolling_15min_avg_wait`/`rolling_1h_avg_wait` now summarise real
  observations rather than mixing in formula output (MAE 7.806 -> 7.762), and
  cuts a full retrain from ~2m36s to ~55s.

  **CONFIRMED ON UNSEEN DATA (2026-08-04).** The justification above was an
  offline ablation; a forward test then scored both artifacts on the *same*
  27,478 rows that neither had been trained on (both from the later of the two
  cutoffs, 2026-08-01T10:11 — scoring each from its own cutoff would have given
  the older model ~19 extra hours the newer one trained on):

      model                        MAE     RMSE    R^2     pred mean   bias
      current (real-data only)     31.24   63.53   0.480   40.0        +1.1
      backup  (with proxy labels)  31.35   78.15   0.214   13.7        -25.2
      (actual mean 38.9 ± 88.1)

  **The finding is calibration, not MAE.** MAE is tied — it is dominated by the
  many low-wait rows where under-predicting costs little, and it is blind to
  bias by construction. The proxy-trained model **under-predicts by 25 minutes**
  (13.7 against a real 38.9), with prediction std 31.0 against an actual 88.1:
  proxy labels are formula-generated and capped at `MAX_DERIVED_WAIT_MINUTES`,
  so they taught a compressed, systematically low distribution. Removing them
  let the real labels set the scale. R^2 more than doubled and RMSE fell 19%.
  Caveat: a forward test can only ever score `siga_live` (the only source that
  produces new rows between retrains), so this says the retirement fixed a
  serious calibration fault — not that the model is accurate in general.

  The per-combo decay below is retained and still tested, since it is what a
  restore would rely on:
  `pipeline/train.py` downweights these proxy rows per-(branch, service) as
  real `siga_live` coverage grows for that specific combo (see
  `compute_sample_weights` — a global cutover would blind whichever combos
  haven't accumulated live data yet, given ~2,000 distinct real combos).
  Run `python -m pipeline.coverage_report` to see real coverage per combo
  before trusting any headline metric — with a single scrape, every combo
  sits at n=1; the blended R^2 stays dominated by proxy rows until real
  coverage actually accumulates over real elapsed time. **`live_count`
  counts only usable readings (non-null `wait_time_minutes`), not raw
  scrape attempts** — found 2026-07-27 that the report used to count every
  `siga_live` row regardless of validity, silently treating readings
  already filtered out as implausible (see the SIGA quirks below) as if
  they were trustworthy real coverage, for exactly the question this
  report exists to answer. `raw_attempt_count`/`implausible_rate` columns
  now make the gap between "scraped" and "usable" visible instead of
  hiding it — the gap turned out to be large (see below). `train.py`
  already reports metrics segmented by source (`metrics_by_source` in the
  saved artifact) so this isn't hidden behind one blended number.
- **Real historical wait times (IALC-M)**: a *second*, separate dados.gov.pt
  dataset, `indicadores-dos-atendimentos-das-lojas-de-cidadao-mensal`
  (discovered 2026-07-27 — easy to miss, since it's not the same dataset as
  SLC-M above and doesn't come up under the obvious search terms). Real,
  **measured** `Tempo_Medio_Espera_Min` (avg wait until service starts) and
  `Tempo_Medio_Atendimento_Min` (avg service duration) per (branch, day),
  spanning 2017-01 through the current month — but branch-level only, no
  per-service breakdown, and some branch-days carry very low
  `Total_Atendimentos` (occasionally 1), making that day's "average" a
  single noisy raw reading rather than a stable mean. Cross-checked against
  SLC-M (2026-07-27): summing SLC-M's per-service `Atendimentos` by
  (branch, day) matches IALC-M's `Total_Atendimentos` for the same
  (branch, day) at **100.00% exact agreement** over the 24-month overlap
  (35,773/35,773 rows) — same underlying system, safe to combine. Ingested
  by `pipeline/load_ialc.py` into `data/cleaned_ialc_baseline.parquet`.
  Consumed two ways: by `pipeline/calibrate_constants.py` (see below), and
  by `pipeline/ingest_real_wait_times.py`, which inserts these real
  branch-day averages directly into `queue_samples` tagged
  `source='historical_real_daily_avg'` — broadcasting the same real value
  to every `desk_service_id` that SLC-M shows as actually active at that
  branch that day (not a per-service split, which the identifiability check
  ruled out — see `pipeline/service_categories.py`), at a single
  representative timestamp per day rather than one row per
  `config.DIURNAL_SNAPSHOTS` entry (N identical copies of one real number
  would just multiply this source's training weight while teaching a flat,
  wrong hour_of_day relationship). That single timestamp is **not** pinned
  to the same hour every day, either — each (branch, date) deterministically
  rotates across `DIURNAL_SNAPSHOTS`' candidate hours
  (`date.toordinal() % N`), so every individual day still gets exactly one
  honest real observation, but real (not formula-shaped) training signal
  ends up spread across every candidate hour instead of concentrated at
  one. Each row's `sample_size` (the real `Total_Atendimentos` behind that
  day's average) drives its own training weight — see
  `pipeline/train.py`'s `compute_sample_weights` docstring for the full
  four-tier scheme (`siga_live` > `historical_real_daily_avg` >
  `historical_derived_proxy`/`synthetic_bootstrap`).
- **Service-category calibration**: `pipeline/calibrate_constants.py` fits
  `avg_service_minutes` per broad service category (see
  `pipeline/service_categories.py`) against real IALC-M branch-day waits,
  replacing config.py's hand-guessed `SERVICE_AVG_MINUTES`/
  `DEFAULT_SERVICE_AVG_MINUTES`. Only 3 categories are used — **IRN**
  (registries/passport/civil-registry), **GENERAL_OTHER** (the default/triage
  desk — the single largest category, 64.4% of real attendance volume
  nationally, and the "everyday Loja do Cidadão visit" most citizens
  experience, especially in Lisbon), and **OTHER_SPECIALIZED** (everything
  else: tax authority, social security, driving/vehicle registry,
  immigration, private-partner desks) — because an identifiability check
  found finer splits aren't separable from real service-mix variation across
  branches (GENERAL_OTHER/IRN correlate at -0.82 across branches; the
  smaller categories are either mutually collinear or present at too few
  branches to fit independently). Uses a bounded nonlinear least-squares fit
  (`scipy.optimize.least_squares`) with a chronological holdout — fitting and
  evaluating on the same data was exactly the mistake the original blended
  R²=0.9985 made. Desks are precomputed once from the *baseline* constants
  and held fixed during the fit (see the module docstring for why re-deriving
  desks from the trial `mu` at every iteration silently broke the optimizer:
  `estimate_desks_for_volume`'s `math.ceil()` poisoned the finite-difference
  Jacobian). Output written to `data/calibrated_service_constants.json`,
  including holdout RMSE for both the calibrated and baseline constants so
  you can see whether the fit actually helped, not just trust that it ran.
  `pipeline/demand_baseline.py` uses this file if present
  (`load_calibrated_mu`/`avg_service_minutes_for`), falling back to the old
  per-service constants if calibration hasn't been run yet.
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
  - **Confirmed across the full real corpus (2026-07-27, correcting an
    earlier miscount): ~31.7% of desk-open readings are implausible**
    (592 of 1,866 rows where SIGA returned a raw wait value at all — the
    other 7,084 of the 8,950 total `siga_live` rows are closed-desk
    readings with no wait value to evaluate, not implausible ones; an
    earlier version of this note conflated the two and wrongly reported
    85.8%). This matches the original 31.7%/46% single-scrape estimates —
    they were right; the recount confirms it, not revises it upward.
    **Segmented, not just an aggregate rate**: implausible rate rises
    sharply with `people_waiting` — 21.9% at 0 people waiting (n=1,549),
    70.1% at 1-5 (n=204), 96.3% at 6-20 (n=80), 100% at 21+ (n=33). More
    people in line makes SIGA's own wait-time field *less* trustworthy,
    not more — the opposite of what you'd want. Root cause still not
    confirmed (candidates: `tempoRealEspera` computed from a stale
    last-ticket timestamp rather than truly live state; a formula that
    breaks down under real load), but the shape of the problem is now
    well-characterized enough to design a targeted fix around, rather
    than a single global cutoff.
  - **The targeted fix, built 2026-07-30: `pipeline/db.py`'s
    `clean_siga_live_readings()`.** The flat 480-min ceiling was still
    letting contaminated readings through just under it — found by
    inspecting `train.py`'s held-out residuals, where the 20 worst
    `siga_live` misses all had labels of 387-480 min against 0-4 people
    waiting, and `abs(residual)` correlated with the raw wait at 0.745 but
    with `people_waiting` at only 0.169. Comparing each reading to the same
    combo's immediately-preceding poll separated **two distinct failure
    modes**: ~55% of >=300min readings are *frozen* (unchanged despite
    15-30 real minutes elapsed), the rest are *erratic* (deltas up to
    ±18,000 min between polls 20-40 min apart). This also refuted the
    "SIGA measures ticket-queue depth, which `people_waiting` doesn't
    capture" theory — a real ticket queue cannot both freeze and swing by
    18,000 minutes. Frozen repeats are dropped (zero information loss —
    a stuck value adds nothing beyond the already-kept prior reading);
    erratic ones are *clamped, not dropped*, so a corrected point
    survives. Measured on retrain: `siga_live` MAE 36.2 -> 29.4 min
    (-19%), RMSE 75.2 -> 64.6. **R² fell 0.540 -> 0.497 and that is not a
    regression** — cleaning removed extreme values from the target's
    distribution in both train and test, and R² is scored against that
    (now narrower) variance. Trust the minute-denominated MAE/RMSE here;
    R² is only comparable across models scored on an identical target
    distribution.
  - **Service-name reconciliation (`pipeline/reconcile_siga_services.py`,
    2026-07-30)** finally closes the gap `coverage_report.py` had been
    flagging: orphaned combos dropped 229 -> 56. Two hard-won constraints
    are encoded there, both found by inspecting real output rather than
    trusting the score:
    1. **Matching is structural, never score-based.** Pure string
       similarity ranked `'Atendimento email'` -> `'Atendimento EMEL'`
       (an email desk vs. Lisbon's parking authority) at 0.909 — *higher*
       than several correct matches — plus `'Chamadas efetuadas'` ->
       `'Chamadas recebidas'` (semantic opposites) and two others. No
       threshold separates these, so only normalized-exact or
       canonical-contained-in-SIGA matches are accepted. Six names stay
       unreconciled as a result (including `'Geral'`); that is the
       intended trade — an unreconciled combo merely forgoes proxy-decay,
       while a wrong merge pollutes a real service's training data.
    2. **The crosswalk is applied to `compute_sample_weights`' join
       only — never to rename `desk_service_id` on the frame.** Renaming
       was tried first and actively corrupted the data: several branches
       run multiple physically distinct desks whose SIGA names reduce to
       one canonical name (Coimbra has 5), and every desk at a branch
       shares one sweep's `sampled_at`. Renaming collapsed them into a
       zero-gap series, which drove `clean_siga_live_readings`'
       `max_plausible_delta` to zero and clamped away genuine
       between-desk differences (clamps 336 -> 764; MAE 29.5 -> 30.3).
       `tests/test_db.py::test_load_all_samples_does_not_rename_service_names`
       guards this. **Grouping keys for any time-series operation must
       stay one-per-real-desk.**
- **Weather**: Open-Meteo Forecast API (rainfall `rain_mm`) by branch
  coordinates. Its `start_date`/`end_date` params only cover roughly the last
  ~110 days plus forecast — for training rows older than that (most of our
  3-year historical span), `rain_mm` silently falls back to
  `config.BASELINE_RAIN_MM` via the same graceful-degradation path used at
  inference time. To get real historical rain further back, switch to
  Open-Meteo's Historical Weather Archive API
  (`archive-api.open-meteo.com`) for dates beyond that window — not done yet.
- **Tax deadlines**: IRS and IMI calendar dates (hardcoded in `config.py`,
  update yearly).
- **Holidays (`pipeline/holidays_pt.py`, added 2026-08-01)**: national
  holidays are computed (10 fixed + 4 Easter-relative, via a local
  `easter_sunday`), and each branch's **`feriado municipal` is derived from
  data, not recalled** — `pipeline/derive_municipal_holidays.py` mines it from
  three years of IALC-M and writes `data/municipal_holidays.json`.

  **The load-bearing fact, verified 2026-08-01: a closed branch-day is
  ABSENT from IALC-M, not present with zero attendances.** All 15 weekday
  national holidays across 2024-2025 produce zero rows; the full corpus check
  now reports 33 holidays confirmed closed, 0 observed open. Two consequences
  drive the whole design:
  1. **`is_holiday` is deliberately NOT a model feature** — it would be
     constant-0 across every training row (holiday rows don't exist), so it is
     unlearnable, and `is_holiday=1` at inference would be pure extrapolation
     off the training distribution — the same class of bug that made
     far-future predictions erratic via `estimate_people_waiting`. Holidays
     act on **`is_open`** instead, which is also where they change what a
     citizen sees. `tests/test_holidays_pt.py` asserts `is_holiday` stays out
     of `FEATURE_COLUMNS`; read the module docstring before "fixing" that.
  2. **The adjacent days are where the signal is**, and they are fully present
     in the data. Measured across the corpus, normalized per branch:
     bridge days ("pontes") **23.3 min vs 18.8 baseline (+26%, n=1,769)** and
     first-day-back **23.0 vs 18.8 (+24%, n=1,909)** — both larger than
     anything `rain_mm` has ever contributed, and available for every
     historical row with no external API. These became the real features,
     `is_bridge_day` and `is_post_holiday`. They overlap by design (a Friday
     after a Thursday holiday is both) and neither is recoverable from the
     other (a Monday before a Tuesday holiday is a ponte but not a
     first-day-back).

  **MEASURED EFFECT OF THE TWO FEATURES: essentially nil on the blended
  metric, and the reason why is the most useful thing here.** Measured by a
  controlled A/B (one data snapshot, one split, models differing only in
  whether the two columns are present — cross-retrain comparison is
  meaningless here, see `pipeline/forward_test.py`):

      tier                        dMAE     dR2      what it means
      historical_derived_proxy   +0.003  -0.0120   slightly WORSE
      historical_real_daily_avg  -0.027  +0.0021   better (real measurements)
      siga_live                  +0.175  +0.0025   noise (n=9,058, ~6 days)
      blended                    +0.0006 -0.0042   nil (+0.03%)

  The proxy tier gets worse **because its labels cannot contain a holiday
  effect**: `pipeline/demand_baseline.py`'s M/M/1 formula has no holiday term,
  so a ponte's synthetic label is identical to an ordinary day's. The features
  are pure noise to that tier and can only cost it a spurious split. Since
  proxy rows are 88% of the test set, they dominate the blended number — so
  the blended metric moving the wrong way is an artifact of the corpus
  composition, not evidence against the features. The one tier made of real
  measured waits (`historical_real_daily_avg`) improved, which is the tier the
  +26%/+24% effect actually lives in.

  Kept on that basis, but **do not quote this as a win** — the honest summary
  is "no measurable net change, right direction on real data". The blocking
  follow-up is to give the proxy generator a holiday-adjacent term so the
  synthetic tier stops contradicting the real one. That is deliberately NOT
  done here: it regenerates all ~6.9M proxy labels and invalidates
  `calibrate_constants.py`'s fit — the same high-blast-radius category as the
  inverted `DIURNAL_SNAPSHOTS` curve, and the two should be recalibrated in
  one pass rather than separately.

  The unambiguous win in this change is not the features at all — it is
  `is_open` (below). That one needs no retrain and no metric to justify it.

  Municipal holidays were derived rather than hardcoded because a table of 74
  half-recalled dates fails in the worst direction: a wrong date both excludes
  a real trading day and admits a real closure, and nothing downstream would
  contradict it. The derivation reproduced feriados municipais it was never
  given — Guarda 27/11, Leiria 22/05, Santarém 19/03, Coimbra 04/07, Setúbal
  15/09, Batalha 14/08, Santiago do Cacém 25/07 — and, the strongest check,
  **independently clustered branches that share one**: all three Lisboa
  branches landed on 13/06 (Santo António) and Porto/Gaia/Valongo on 24/06
  (São João), which cannot happen by chance. Data alone resolved 45 of 78.

  **A published list was then added as `data/municipal_holidays_seed.json`
  (2026-08-01), and it does NOT override the corpus.** `reconcile()` treats it
  strictly as a candidate: **the seed proposes, the data vetoes.** Outcome —
  67 confirmed, 8 added, 1 contradicted, 1 conflict; coverage 45 -> 76 of 78,
  and the national attendance volume covered went from 74.4% to ~99%. The four
  outcome classes are all reported rather than collapsed, because each means
  something different:
  - **added** is what the seed exists for. Faro (07/09), Viseu (21/09), Sintra
    and Seixal (29/06) were correct all along but *unobservable*: their holiday
    fell on a weekend in two of the three corpus years, so `min_years=2`
    rightly refused to conclude from a single sighting.
  - **contradicted** is the seed being rejected, and it is not necessarily an
    error in the list — **feriados municipais are optional under the Código do
    Trabalho**, so a municipality having one does not oblige this branch to
    close. Castelo Branco traded through Nossa Senhora de Mércoles in all three
    years, so it is left unlisted.
  - **11 of 76 rules are movable**, and that is not an edge case: the two dates
    originally recalled from memory that the corpus contradicted (Castelo
    Branco 08/09, Gondomar 24/06) were *both* wrong because those holidays are
    movable — Easter+9 and the Monday after October's first Sunday. A fixed-date
    table would have been wrong for them every single year. `resolve_rule()`
    handles `fixed`, `easter_offset` (Segunda-feira de Páscoa +1, Mércoles +9,
    Ascensão/Dia da Espiga +39, Pentecostes +50) and `monday_after_nth_sunday`;
    all 14 arithmetic cases are pinned against the published 2026/2027 calendar
    in `tests/test_holidays_pt.py`.
  - **The veto also caught a registry bug.** `loja_de_cidadao_de_pinhal_novo`
    is recorded in `branches_registry.json` with `municipality: "Setúbal"`, but
    Pinhal Novo is a freguesia of **Palmela** — proven here, since the branch
    was OPEN on Setúbal's 15/09 in both observable years and CLOSED on Palmela's
    01/06. Handled via `branch_overrides` in the seed. **Its geocode and
    district are suspect for the same reason and have not been rechecked.**

  Two branches remain unlisted (Castelo Branco, which demonstrably does not
  close; Serpa, absent from the published list) — **left unlisted rather than
  guessed**, and a missing entry means "unknown", never "no holiday".
  Vila Pouca de Aguiar is a standing **conflict**: the corpus shows it closed
  16/05 in two separate years, while the published list gives 22/06 (which fell
  on a weekend in both observable years, so it was never testable). Derived
  wins; the 16/05 closure is real but unexplained.

  Two false-positive classes make the filters in that script non-optional:
  `loja_de_cidadao_de_palmela_movel` is a **mobile unit** (open 30% of
  weekdays, matching ~180 spurious month-days) and Freixo de Espada à Cinta
  reports intermittently (77%). `MIN_WEEKDAY_PRESENCE_RATE` excludes both —
  absence only carries information for a branch that is reliably open.

  **Carnival is conditional and the data says so.** It is `tolerância de
  ponto`, granted yearly at the government's discretion, so it is flagged
  `statutory=False`: closed in 2024/2025, and in 2026 it left exactly one
  branch reporting one attendance against a 75-branch/198-attendance norm.
  Treated as a holiday, but `verify_against_corpus()` re-checks it rather than
  assuming it recurs.

  **In the static site build** (`pipeline/build_static.py`): holiday-adjacent
  branch-days are excluded from the weekday/month/sparkline means via
  `typical_days()`, since a page answering "when should I go" should describe
  an ordinary visit — leaving them in made every typical-day figure quietly
  pessimistic. 2,805 of 51,206 branch-days (5.5%) are excluded, and this
  **changed the recommended best weekday for 8 of 78 branches**, so it is a
  real correction rather than a cosmetic one. The rows are filtered from the
  means, not dropped from the frame: `build_corpus_summary` still counts them,
  because that number is a statement about the evidence base, not about a
  typical visit. Two new payload keys: `hol` ([month, day] of the branch's
  feriado municipal, **omitted when unknown** — the page must treat a missing
  key as unknown, not as "no holiday") and `pon` (that branch's own
  holiday-adjacent multiplier, median 1.21, emitted only above
  `MIN_ADJACENT_DAYS_FOR_UPLIFT`). `is_holiday_closure` is carried as a live
  assertion that closed days stay absent — it should always sum to zero, and
  `main()` prints a warning if it ever doesn't.
- **is_open**: real observed value from live SIGA polling when available
  (near-now requests only — see `config.NEAR_NOW_WINDOW_MINUTES`); otherwise
  a fixed Mon-Fri/9-17 heuristic (`config.ASSUMED_BUSINESS_*`) **now also
  suppressed by national and municipal holidays** (2026-08-01). A real
  observed value still wins over the calendar — if live SIGA reports a desk
  open on a holiday, that is a measurement and this is a guess; the holiday
  mask is folded into the heuristic *before* the real value fills in, and
  `tests/test_holidays_pt.py` guards that ordering. Passing `branch_id` into
  `estimate_is_open_heuristic` is what makes it municipality-aware; without it
  a Porto branch reports open on São João and could then be offered as a fast
  reroute target — the exact failure `get_is_open` exists to prevent, arriving
  via the calendar rather than via live state. Proxy/synthetic
  rows set this explicitly from generation-time evidence, not the heuristic —
  the real dataset includes ~15.7k Saturday rows with real nonzero
  attendance, which the Mon-Fri heuristic alone would have wrongly marked closed.

## THE TWO WAIT-TIME SOURCES DO NOT MEASURE THE SAME QUANTITY (2026-08-02)

**Read this before touching sample weighting, `hour_of_day`, or the
`DIURNAL_SNAPSHOTS` curve.** It invalidates the obvious next moves, all three
of which were tried and measured here rather than reasoned about.

`wait_time_minutes` is one column, but it means different things in
`historical_real_daily_avg` (IALC-M `Tempo_Medio_Espera_Min`) and in
`siga_live` (SIGA `tempoRealEspera`). Compared across 72 branches with >=100
usable live readings each:

  - live readings average **2.45x** IALC-M (median 2.04x)
  - per-branch ratios: Murça 0.8x, Odivelas 1.3x, Laranjeiras 2.1x,
    Porto 2.6x, Coimbra 3.7x, Braga 3.7x, **Viseu 5.8x**
  - **correlation between the two measures across branches: 0.193**

A genuinely slow branch should be slow under both. It is not. Known partial
explanations — no temporal overlap (IALC-M ends 2026-07-25, siga_live starts
2026-07-26), live sampling only business hours while IALC-M averages whole
days, live being per-service against IALC-M's branch-level — account for some
of the gap but cannot produce 5.8x at Viseu or a 0.193 correlation. **The root
cause is not established.** A plausible unverified candidate is that IALC-M
measures the realised wait of people who were *served* while `tempoRealEspera`
estimates the current queue depth for someone joining now, but the give-up
test below did not confirm it.

**Consequence: reweighting is the wrong lever.** It does not add hour
resolution to a shared target; it slides the target's *definition* toward
live's, per branch, by an inconsistent factor.

### Measured: upweighting siga_live makes everything worse
Training weight mass is 594,957 (daily avg) : 36,231 (live), ~16:1, which is
why the model cannot learn an hour effect. Multiplying live's weight:

    live_x   MAE     R^2     daily_avg MAE   siga_live MAE   hour range   hour gain%
    1        7.762   0.546   6.456           32.613          0.51 min     2.28%
    5        7.890   0.537   6.574           32.920          0.37         3.09%
    20       8.295   0.526   6.973           33.449          7.24         3.98%
    50       9.300   0.484   8.038           33.314          11.78        4.89%

The hour dimension *does* unlock (range 0.51 -> 11.78 min), confirming that
row-count is what suppresses it. **But `siga_live` MAE gets WORSE too** — if
the hour signal were real, upweighting the rows carrying it would improve the
fit on those rows. It does not. The model is fitting ~7 days of scatter.

### Consequently downgraded: the `DIURNAL_SNAPSHOTS` "inverted curve" finding
`config.py`'s open finding (assumed vs observed hour curve, r = -0.79) was
computed from `people_waiting`, and its standing recommendation was to
recalibrate against that field. **Do not act on that plan.** `people_waiting`
has now failed two separate checks: it reads 0.2-0.8 people at branches
serving 1,000+/day alongside 100-minute waits (incoherent unless it counts
desk-assigned users rather than the ticket queue), and it runs *opposite* to
wait across the day (r = -0.587). Its blast radius is now small anyway — after
the proxy retirement the hand-drawn `volume_factor` only reaches
`estimate_people_waiting`'s inference fallback.

### The hourly wait climb is REAL at busy branches — earlier note was wrong
An earlier reading of "`tempoRealEspera` climbs monotonically 36 -> 75 min
across the day = stale-counter pathology" was reached from a pooled,
per-branch-normalized average that mixed Laranjeiras in with 70-attendance-a-
day branches. Segmented by branch volume it reverses cleanly:

    segment   daily attendance   wait 9h -> 16h    change
    low       70                 19.3 -> 3.4       -15.8   (drains)
    mid       280                33.1 -> 42.9      +9.8
    HIGH      1058               50.9 -> 80.5      +29.7   (builds)

A broken counter does not know how busy a branch is. Quiet branches drain,
busy ones accumulate backlog — ordinary transient queueing. Laranjeiras runs
55 -> 117 min. **The separately-documented *implausible* readings (5-figure
waits, frozen repeats, ±18,000 min swings) remain genuinely broken and
`clean_siga_live_readings` still earns its place** — that is a different claim
and the two were conflated.

### Give-ups: measured, and NOT fixable by feature engineering
15.5% of all tickets nationally are abandoned (3,495,897 of 22,553,347), 25.4%
at Laranjeiras. None appear in any wait label, so `avg_wait_minutes` is
"wait among those who stayed" — censored hardest exactly where waits are worst.
Two fixes were tried; **both failed**:

    A status quo                          MAE 7.762
    B weight by (1 - give_up_rate)        MAE 7.781   (+0.019, worse)
    C give_up_rate as a feature           MAE 7.542   (-0.220, but LEAKY)
    E prior-history give-up as a feature  MAE 7.820   (+0.058, worse)

C's gain was **contemporaneous leakage** — a day's give-up rate is an outcome
of that day, unknown at planning time, and driven by the same congestion that
made the wait long. The deployable version (E, expanding mean of the branch's
prior days) is worse than doing nothing, because **74.2% of give-up variance is
between branches rather than within them** — `branch_id` already encodes it.
B fails because down-weighting high-give-up days discards the busiest branches,
where most signal lives. Note the weighting *does* correlate with censoring
(weight vs give-up rate r = +0.438; `sample_size` is `total_attendances`, i.e.
people served), but correcting it costs accuracy without recovering anything.
The missing observations cannot be reconstructed from this corpus.

### What would actually move any of this
  1. **Capture throughput.** `queue_samples.last_ticket_called` exists and is
     **100% empty** (0 of 150,342 rows) — the scraper writes the column but
     never populates it, and `siga_client.py` documents no ticket-number field.
     Ticket numbers advancing between polls would give a hard count of people
     served: independent of `tempoRealEspera`'s reliability and immune to
     survivorship. Probe a raw SIGA response before anything else here.
  2. **Two-stage, not reweighted.** Predict the day level from historical data
     (where the model is strong, MAE 6.46), then apply a *relative* hour
     profile learned from live. Relative profiles are scale-free, so the 2.45x
     mismatch cancels — live's hour information without live's units. Branches
     with no live coverage keep a flat profile rather than a fabricated curve.
  3. **Predict give-up rate as its own output.** Measurable across three years
     for every branch, no survivorship problem (give-ups are counted, not
     lost), no hour resolution needed, no dependency on scraper accumulation.

## Forward-test reference points

`pipeline/forward_test.py` is the only leakage-free measure of a deployed
artifact. Record every run here — the whole point of a *fixed* reference is
that runs are comparable to each other, which held-out metrics never are.

    date         model                    cutoff             window   n        MAE     RMSE    R^2     bias
    2026-08-04   real-data only (17f)     2026-08-01 10:11   75.2h    27,478   31.24   63.53   0.480   +1.1
    2026-08-04   pre-retirement (15f)     2026-08-01 10:11   75.2h    27,478   31.35   78.15   0.214   -25.2

Reading these honestly:
  - **Live-tier only.** IALC-M arrives monthly, so between retrains every new
    row is `siga_live`. A forward test therefore measures prediction of SIGA's
    `tempoRealEspera` — a different quantity from the IALC-M waits that make up
    95% of training (see the source-mismatch section). It is a real,
    leakage-free number for a narrower thing than "wait time accuracy".
  - **Quote the window composition**, always. The 2026-08-04 window spans a
    Saturday afternoon, a Sunday gap (cron is Mon-Sat), and two weekdays.
  - **Check the coverage split, not just the aggregate.** On the run above,
    sparse combos scored MAE 47.0 against 21.7 for well-covered ones. A change
    that helps covered combos while hurting sparse ones is invisible in the
    headline — which is why that split exists.

**`--model` reads the ARTIFACT's own `feature_columns`, never the module-level
`FEATURE_COLUMNS`** (fixed 2026-08-04). Using the current constant made it
impossible to score any model built before a feature change — precisely the
comparison a forward test exists for; the 15-feature backup raised a
`feature_names mismatch` against the 17-feature codebase. `api/service.py` had
always done this correctly.

Comparing two artifacts with different cutoffs requires scoring both from the
LATER cutoff, or the older model is handed rows the newer one trained on.
`forward_test.py` has no flag for this yet — the 2026-08-04 comparison was run
with a short script against `build_forward_test_frame`.

## Backtesting the static site as a forecaster

`pipeline/backtest_site.py` scores the page's own aggregates as predictions,
rolling-origin, using only data already on disk. The page always made a
forecast ("the typical Tuesday is 24 min" predicts next Tuesday); nothing had
ever checked it. Result over 10 windows x 30 days:

    recent_weekday (last 8)              MAE 6.244   <- best, wins 9 of 10 windows
    recent_weekday + holiday uplift      MAE 6.288
    weekday_typical (SITE TODAY)         MAE 6.797
    weekday_all_days (site pre-08-01)    MAE 6.821
    weekday_typical + month              MAE 7.025
    branch_mean                          MAE 7.060
    naive_persistence                    MAE 8.566

Three findings that should temper what the page claims:
  - **Recent history beats all history.** Averaging the last 8 occurrences of a
    weekday beats a three-year mean by ~0.4-0.55 min. `build_static.py` has NOT
    been switched to this yet. `RECENT_WINDOW_OCCURRENCES` was validated across
    10x30d / 6x60d / 4x90d rather than taking the argmin of one run — that
    first pass suggested 12 and a 0.85 min gain, which was the constant
    overfitting its own scoring windows.
  - **The holiday-adjacent exclusion does not improve forecast accuracy**
    (6.797 vs 6.821 — noise). It changes which weekday is recommended for 8 of
    78 branches and is defensible as "describe an ordinary visit", but it is a
    positioning argument, not an accuracy one. Do not claim otherwise.
  - **The month adjustment actively hurts** (7.025 vs 6.797), and the weekday
    split itself buys only 0.26 min over ignoring the calendar entirely. The
    site's most prominent feature is worth ~15 seconds of accuracy; keep it as
    a UX choice, not a precision claim.

Hour-of-day and per-service are deliberately out of scope here: neither SLC-M
nor IALC-M has ever had a time field (both are daily), so no backtest over this
corpus can speak to them.

## Continuous live data collection (GitHub Actions)

`.github/workflows/siga_scraper.yml` runs the scraper on a schedule (weekday
opening/midday/afternoon slots, WEST/WET-approximate) without needing any
machine to stay on. It writes to `data/live_samples.csv`, not
`data/queue_history.db` — the local DB is 400MB+ (dominated by proxy rows)
and gitignored on purpose: GitHub hard-rejects files over 100MB, and a binary
SQLite blob re-committed every run would bloat history regardless, since git
can't delta-compress it the way it can plain text. `live_samples.csv` is
small and append-only, so repeated commits stay cheap.

**Scheduled runs aren't guaranteed to fire on time — found 2026-07-27,
day one of running this:** only 3 of the 5 daily cron slots produced a
run, each 24-99 min late (checked via `gh run list`). Root cause: the
cron fired at exact on-the-hour minutes (`:00`), and GitHub's own docs
warn that's the highest-congestion moment for the shared Actions
scheduler — every repo with an on-the-hour cron collides there, and
scheduled runs can be delayed or silently dropped under that load. Fixed
by offsetting the cron to `:17` instead of `:00` — doesn't guarantee
on-time firing, just avoids the specific congestion peak GitHub calls
out. If gaps recur, check `gh run list --workflow=siga_scraper.yml`
before assuming the scraper or SIGA API itself is broken.

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
