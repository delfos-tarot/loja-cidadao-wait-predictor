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
│   └── siga_desk_service_crosswalk.json  # SIGA service name -> dados.gov.pt name (generated)
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
