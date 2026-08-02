"""Train an XGBRegressor to predict wait_time_minutes and export it to joblib.

Uses a chronological (non-shuffled) train/test split, since queue wait times
are a time series and random shuffling would leak future information into
the training set — but applied per-source (chronological_split_by_source),
not as one global cutoff across all rows. A single global cutoff would
structurally exclude siga_live from training entirely, since it's a tiny,
very recent sliver next to ~2 years of historical data — every live row
would land in the most-recent 20% (test) and none in training. Splitting
each source along its own timeline means siga_live gets its own 80/20 split,
so the model actually trains on some real measurements instead of only
being tested against data it never learned from. Prints MAE, RMSE, and R^2
on the held-out slice, then saves the fitted model bundled with its feature
column order and evaluation metrics to `models/xgboost_wait_time.joblib`.

Rows are not all trusted equally — four source tiers, weighted differently
(see `compute_sample_weights`):
  - `siga_live`: real, per-service, live-instant — weight 1.0.
  - `historical_real_daily_avg`: real but coarser (branch-level, daily
    average from IALC-M — see pipeline/ingest_real_wait_times.py), weighted
    per-row by how many real attendances backed that specific branch-day
    average (some are as low as 1).
  - `historical_derived_proxy` / `synthetic_bootstrap`: formula-derived, not
    measured — **RETIRED from training on 2026-08-01**
    (`config.PROXY_LABEL_TRAINING_WEIGHT = 0.0`). A controlled ablation showed
    that removing them IMPROVED accuracy on real measurements (MAE 7.955 ->
    7.804, R^2 0.484 -> 0.545 on a real-rows-only test set), and that the
    thinnest-coverage combos — the ones the tier existed to serve — improved
    most. They also carry DIURNAL_SNAPSHOTS' hand-drawn hour curve, which live
    data contradicts at r = -0.79. Nothing is deleted: the rows stay in the DB
    and the per-combo decay below still works; restoring is one constant. See
    config.PROXY_LABEL_TRAINING_WEIGHT for the full numbers and the honest cost
    (hour_of_day is now supported almost entirely by siga_live).

Usage:
    python -m pipeline.train
    python -m pipeline.train --test-fraction 0.2
    python -m pipeline.train --proxy-decay-alpha 0.1
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from config import (
    DEFAULT_DB_PATH,
    DEFAULT_MODEL_PATH,
    HISTORICAL_REAL_AVG_REFERENCE_SAMPLE_SIZE,
    PROXY_LABEL_TRAINING_WEIGHT,
    PROXY_WEIGHT_DECAY_ALPHA,
)
from pipeline.db import clean_siga_live_readings, load_all_samples, load_service_crosswalk
from pipeline.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN, QueueFeatureTransformer

logger = logging.getLogger(__name__)


def load_training_frame(db_path: str) -> pd.DataFrame:
    frame = load_all_samples(db_path)
    frame = clean_siga_live_readings(frame)
    frame = frame.dropna(subset=[TARGET_COLUMN])
    frame = frame.sort_values("sampled_at").reset_index(drop=True)
    return frame


def chronological_split(frame: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(frame) * (1 - test_fraction))
    train_frame = frame.iloc[:split_index]
    test_frame = frame.iloc[split_index:]
    return train_frame, test_frame


def chronological_split_by_source(frame: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applies chronological_split independently within each `source`, then
    recombines — still fully chronological (no shuffling, no future leakage;
    "time-series discipline" per CLAUDE.md), just anchored to each source's
    own timeline instead of one global cutoff.

    Found 2026-07-27: a single global chronological cutoff across all
    sources structurally starves siga_live of any training exposure at
    all. siga_live only exists for the last few weeks, next to ~2 years of
    historical_derived_proxy/historical_real_daily_avg — so under one
    global 80/20 split by row position, literally every siga_live row
    lands in the most-recent 20% (test), and zero land in training. The
    "test" of live performance was really "how does a model with zero
    exposure to live data generalize to it", not "how well does the model
    learn from live data" — two different questions. Splitting per source
    means siga_live gets its own 80/20 split along its own (short) timeline,
    so the model actually gets to train on real measurements for the first
    time, while historical_derived_proxy/historical_real_daily_avg keep
    the same global-style split they already had (their timelines already
    span ~2 years, so a global cutover was never a problem for them).
    """
    train_parts, test_parts = [], []
    for _, subset in frame.groupby("source"):
        train_subset, test_subset = chronological_split(subset.sort_values("sampled_at"), test_fraction)
        train_parts.append(train_subset)
        test_parts.append(test_subset)
    train_frame = pd.concat(train_parts).sort_values("sampled_at").reset_index(drop=True)
    test_frame = pd.concat(test_parts).sort_values("sampled_at").reset_index(drop=True)
    return train_frame, test_frame


def compute_sample_weights(
    frame: pd.DataFrame,
    alpha: float = PROXY_WEIGHT_DECAY_ALPHA,
    real_avg_reference_sample_size: float = HISTORICAL_REAL_AVG_REFERENCE_SAMPLE_SIZE,
    proxy_label_training_weight: float = PROXY_LABEL_TRAINING_WEIGHT,
) -> np.ndarray:
    """Four-tier weighting by source:

    - `siga_live`: weight 1.0 (real, per-service, live-instant — the
      strongest ground truth).
    - `historical_real_daily_avg`: weight sample_size / (sample_size +
      real_avg_reference_sample_size), using that row's OWN sample_size (how
      many real attendances backed that specific branch-day average — see
      pipeline/ingest_real_wait_times.py). Real, but coarser (branch-level,
      daily) than siga_live, and some branch-days rest on as few as 1
      attendance, so this is a per-row reliability weight, not a per-combo
      decay — there's no "combo" to decay against, just "how much do we
      trust this one number".
    - `historical_derived_proxy` / `synthetic_bootstrap`: 1 / (1 + alpha *
      live_count), where live_count is the number of siga_live rows that
      specific (branch_id, desk_service_id) already has — so a combo's
      formula-derived proxy fades out only as that combo earns real
      siga_live coverage, without penalizing combos that haven't yet.

    Live rows' service names are mapped through the SIGA service crosswalk
    (pipeline/reconcile_siga_services.py) *for this count only*, so real
    coverage recorded under a SIGA-specific name ("Câmara - Atendimento
    Geral") still decays the proxy rows filed under the dados.gov.pt name
    ("Atendimento Geral"). Note several such SIGA names can map to one
    canonical name at the same branch — that's intended here, since they
    are all real coverage of that service and their counts should sum.
    The mapping is deliberately confined to this join: see
    pipeline/db.py's load_service_crosswalk docstring for how renaming the
    frame itself instead corrupted the time-series grouping.
    """
    live_rows = frame.loc[frame["source"] == "siga_live", ["branch_id", "desk_service_id"]]
    crosswalk = load_service_crosswalk()
    if crosswalk:
        live_rows = live_rows.assign(desk_service_id=live_rows["desk_service_id"].replace(crosswalk))
    live_counts = live_rows.value_counts().rename("live_count").reset_index()
    merge_columns = ["branch_id", "desk_service_id", "source"]
    if "sample_size" in frame.columns:
        merge_columns.append("sample_size")
    merged = frame[merge_columns].merge(live_counts, on=["branch_id", "desk_service_id"], how="left")
    merged["live_count"] = merged["live_count"].fillna(0)

    is_live = (merged["source"] == "siga_live").to_numpy()
    is_real_daily_avg = (merged["source"] == "historical_real_daily_avg").to_numpy()

    proxy_weight = 1.0 / (1.0 + alpha * merged["live_count"].to_numpy())

    sample_size = merged["sample_size"].fillna(0).to_numpy() if "sample_size" in merged.columns else np.zeros(len(merged))
    real_daily_avg_weight = sample_size / (sample_size + real_avg_reference_sample_size)

    # Retired to 0.0 on 2026-08-01 — see config.PROXY_LABEL_TRAINING_WEIGHT for
    # the ablation that justified it. Applied as a multiplier on top of the
    # decay rather than replacing it, so restoring the constant to 1.0 brings
    # back exactly the previous behaviour.
    proxy_weight = proxy_weight * proxy_label_training_weight

    weights = proxy_weight
    weights = np.where(is_real_daily_avg, real_daily_avg_weight, weights)
    weights = np.where(is_live, 1.0, weights)
    return weights


def drop_zero_weight_rows(frame: pd.DataFrame, weights: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Removes rows XGBoost would ignore anyway.

    A zero-weight row contributes nothing to the loss, so this changes no
    result — it just avoids carrying ~6.9M retired proxy rows through the
    split, the fit, and the metrics. Training drops from ~4 minutes to under
    one, which is the difference between running an experiment and not
    bothering.

    Done BEFORE the chronological split on purpose: leaving proxy rows in the
    TEST set while excluding them from training would produce a blended metric
    dominated by rows the model was never shown — a meaningless number that
    looks like a catastrophic regression.
    """
    keep = weights > 0
    return frame.loc[keep].reset_index(drop=True), weights[keep]


def train_model(X_train: pd.DataFrame, y_train: pd.Series, sample_weight: np.ndarray | None = None) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        enable_categorical=True,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def evaluate(model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2)}


def evaluate_by_source(
    model: XGBRegressor, test_frame: pd.DataFrame, feature_columns: list[str], target_column: str
) -> dict[str, dict]:
    """Segments held-out evaluation by row source.

    A single blended R^2 is misleading here: >99.9% of rows are
    historical_derived_proxy, a label the model can reconstruct almost
    exactly from redundant inputs (historical_avg_attendances, hour_of_day,
    branch/service categoricals) it's also given — that's curve-fitting a
    known formula, not evidence of real predictive accuracy. This surfaces
    the real (siga_live) slice's performance separately instead of letting
    the huge proxy majority average it away.
    """
    results: dict[str, dict] = {}
    for source_name, subset in test_frame.groupby("source"):
        predictions = model.predict(subset[feature_columns])
        actual = subset[target_column]
        mae = float(mean_absolute_error(actual, predictions))
        rmse = float(np.sqrt(mean_squared_error(actual, predictions)))

        r2: float | None = None
        note: str | None = None
        if len(subset) < 2:
            note = "too few rows for R^2"
        elif actual.nunique() == 1:
            note = "constant target in this slice — R^2 undefined, ignore"
        else:
            r2 = float(r2_score(actual, predictions))

        results[source_name] = {"n": int(len(subset)), "mae": mae, "rmse": rmse, "r2": r2, "note": note}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the XGBoost wait-time model")
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--model-out", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--proxy-decay-alpha", type=float, default=PROXY_WEIGHT_DECAY_ALPHA)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    frame = load_training_frame(args.db)
    if len(frame) < 50:
        raise SystemExit(
            f"Only {len(frame)} labeled rows available; need historical data or a synthetic "
            "bootstrap (python -m pipeline.synthetic_bootstrap --days 30, or the real pipeline: "
            "pipeline.load_historical -> pipeline.geocode_branches -> pipeline.demand_baseline) before training."
        )
    logger.info("Loaded %d labeled rows spanning %s to %s", len(frame), frame["sampled_at"].min(), frame["sampled_at"].max())

    # Drop retired (zero-weight) rows BEFORE feature engineering, not after:
    # the transform is the expensive step (~2 min over 7.8M rows, vs 4s for the
    # fit itself), and there is no point enriching rows the loss will ignore.
    # Weighting only needs source/branch/service/sample_size, all present on the
    # raw frame. Removing them before the SPLIT also matters for correctness --
    # leaving proxy rows in TEST while excluding them from TRAIN would produce a
    # blended metric dominated by rows the model was never shown.
    #
    # Filtering here rather than after the transform is not purely an
    # optimisation: add_lag_rolling_features computes rolling wait averages over
    # whatever rows are present, so doing it first means rolling_15min_avg_wait /
    # rolling_1h_avg_wait now summarise REAL observations instead of blending
    # measured waits with formula output. Measured on retrain: MAE 7.806 ->
    # 7.762, R^2 0.5432 -> 0.5459. Small, but in the right direction and for the
    # right reason.
    pre_weights = compute_sample_weights(frame, alpha=args.proxy_decay_alpha)
    kept, _ = drop_zero_weight_rows(frame, pre_weights)
    if len(kept) < len(frame):
        logger.warning(
            "EXCLUDING %d zero-weight rows from train AND test (%d -> %d). "
            "config.PROXY_LABEL_TRAINING_WEIGHT=%.3f -- set to 1.0 to restore proxy labels.",
            len(frame) - len(kept), len(frame), len(kept), PROXY_LABEL_TRAINING_WEIGHT,
        )
        for source_name, count in frame["source"].value_counts().items():
            logger.info("  %-28s %8d -> %8d", source_name, count, int((kept["source"] == source_name).sum()))
    frame = kept

    transformer = QueueFeatureTransformer()
    features = transformer.transform(frame)
    # Assign as Series (not .to_numpy()) so pandas aligns by index label
    # rather than row position, which protects against any upstream step
    # reordering rows (e.g. the per-group rolling-window computation).
    features[TARGET_COLUMN] = frame[TARGET_COLUMN]
    features["sampled_at"] = frame["sampled_at"]
    features["source"] = frame["source"]
    features["sample_size"] = frame["sample_size"]
    features = features.sort_values("sampled_at").reset_index(drop=True)

    train_frame, test_frame = chronological_split_by_source(features, args.test_fraction)
    logger.info("Train rows: %d | Test rows: %d", len(train_frame), len(test_frame))

    X_train, y_train = train_frame[FEATURE_COLUMNS], train_frame[TARGET_COLUMN]
    X_test, y_test = test_frame[FEATURE_COLUMNS], test_frame[TARGET_COLUMN]

    train_weights = compute_sample_weights(train_frame, alpha=args.proxy_decay_alpha)
    live_row_count = int((train_frame["source"] == "siga_live").sum())
    real_daily_avg_row_count = int((train_frame["source"] == "historical_real_daily_avg").sum())
    logger.info(
        "Sample weighting: %d siga_live rows, %d historical_real_daily_avg rows in training set (proxy-decay alpha=%.3f)",
        live_row_count, real_daily_avg_row_count, args.proxy_decay_alpha,
    )

    model = train_model(X_train, y_train, sample_weight=train_weights)
    metrics = evaluate(model, X_test, y_test)
    metrics_by_source = evaluate_by_source(model, test_frame, FEATURE_COLUMNS, TARGET_COLUMN)

    print("Evaluation metrics (held-out, most recent time slice, ALL sources blended):")
    print(f"  MAE  : {metrics['mae']:.3f} minutes")
    print(f"  RMSE : {metrics['rmse']:.3f} minutes")
    print(f"  R^2  : {metrics['r2']:.4f}")
    print()
    print("Same test set, segmented by row source — a blended number hides how")
    print("much of the fit is formula-derived proxy rows vs real measurements:")
    for source_name, m in metrics_by_source.items():
        r2_display = f"{m['r2']:.4f}" if m["r2"] is not None else "N/A"
        note_display = f"  ({m['note']})" if m["note"] else ""
        print(f"  {source_name:26s} n={m['n']:>8d}  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  R^2={r2_display}{note_display}")

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        # The newest sampled_at this model actually learned from. Distinct
        # from trained_at (wall clock) and the field pipeline/forward_test.py
        # MUST filter on: rows can be scraped while training runs, so
        # trained_at would wrongly mark genuinely-unseen rows as seen, and a
        # slow import would wrongly mark trained-on rows as unseen -- either
        # way silently scoring the model against its own training data.
        "data_cutoff": frame["sampled_at"].max().isoformat(),
        "metrics": metrics,
        "metrics_by_source": metrics_by_source,
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
    }
    joblib.dump(artifact, args.model_out)
    logger.info("Saved model artifact to %s", args.model_out)


if __name__ == "__main__":
    main()
