"""Train an XGBRegressor to predict wait_time_minutes and export it to joblib.

Uses a chronological (non-shuffled) train/test split, since queue wait times
are a time series and random shuffling would leak future information into
the training set. Prints MAE, RMSE, and R^2 on the held-out (most recent)
slice, then saves the fitted model bundled with its feature column order and
evaluation metrics to `models/xgboost_wait_time.joblib`.

Rows are not all trusted equally: real `siga_live` measurements get full
weight; `historical_derived_proxy` / `synthetic_bootstrap` rows get a weight
that decays per-(branch_id, desk_service_id) as real live coverage grows for
that specific combo (see `compute_sample_weights`). This is deliberately not
a global "drop all proxy data once N live rows exist" cutover: real coverage
will be extremely uneven across ~2,000 distinct combos, and a global
threshold would blind whichever combos haven't accumulated live data yet.
Today, with zero siga_live rows anywhere, every weight is 1.0 — this is a
no-op until live scraping actually produces data.

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

from config import DEFAULT_DB_PATH, DEFAULT_MODEL_PATH, PROXY_WEIGHT_DECAY_ALPHA
from pipeline.db import load_all_samples
from pipeline.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN, QueueFeatureTransformer

logger = logging.getLogger(__name__)


def load_training_frame(db_path: str) -> pd.DataFrame:
    frame = load_all_samples(db_path)
    frame = frame.dropna(subset=[TARGET_COLUMN])
    frame = frame.sort_values("sampled_at").reset_index(drop=True)
    return frame


def chronological_split(frame: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(frame) * (1 - test_fraction))
    train_frame = frame.iloc[:split_index]
    test_frame = frame.iloc[split_index:]
    return train_frame, test_frame


def compute_sample_weights(frame: pd.DataFrame, alpha: float = PROXY_WEIGHT_DECAY_ALPHA) -> np.ndarray:
    """Real `siga_live` rows get weight 1.0. Proxy/synthetic rows get
    1 / (1 + alpha * live_count), where live_count is the number of
    siga_live rows that specific (branch_id, desk_service_id) already has —
    so a combo's proxy data fades out only as that combo earns real
    coverage, without penalizing combos that haven't yet.
    """
    live_counts = (
        frame.loc[frame["source"] == "siga_live", ["branch_id", "desk_service_id"]]
        .value_counts()
        .rename("live_count")
        .reset_index()
    )
    merged = frame[["branch_id", "desk_service_id", "source"]].merge(
        live_counts, on=["branch_id", "desk_service_id"], how="left"
    )
    merged["live_count"] = merged["live_count"].fillna(0)

    is_live = (merged["source"] == "siga_live").to_numpy()
    proxy_weight = 1.0 / (1.0 + alpha * merged["live_count"].to_numpy())
    return np.where(is_live, 1.0, proxy_weight)


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

    transformer = QueueFeatureTransformer()
    features = transformer.transform(frame)
    # Assign as Series (not .to_numpy()) so pandas aligns by index label
    # rather than row position, which protects against any upstream step
    # reordering rows (e.g. the per-group rolling-window computation).
    features[TARGET_COLUMN] = frame[TARGET_COLUMN]
    features["sampled_at"] = frame["sampled_at"]
    features["source"] = frame["source"]
    features = features.sort_values("sampled_at").reset_index(drop=True)

    train_frame, test_frame = chronological_split(features, args.test_fraction)
    logger.info("Train rows: %d | Test rows: %d", len(train_frame), len(test_frame))

    X_train, y_train = train_frame[FEATURE_COLUMNS], train_frame[TARGET_COLUMN]
    X_test, y_test = test_frame[FEATURE_COLUMNS], test_frame[TARGET_COLUMN]

    train_weights = compute_sample_weights(train_frame, alpha=args.proxy_decay_alpha)
    live_row_count = int((train_frame["source"] == "siga_live").sum())
    logger.info(
        "Sample weighting: %d live rows in training set (proxy-decay alpha=%.3f)",
        live_row_count, args.proxy_decay_alpha,
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
        "metrics": metrics,
        "metrics_by_source": metrics_by_source,
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
    }
    joblib.dump(artifact, args.model_out)
    logger.info("Saved model artifact to %s", args.model_out)


if __name__ == "__main__":
    main()
