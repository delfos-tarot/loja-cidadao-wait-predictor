"""Scores the saved model against data that arrived AFTER it was trained.

The honest counterpart to pipeline/train.py's held-out metrics. That split
takes the most recent 20% per source, so with only a few days of siga_live
the slice is roughly "the last day" -- and it MOVES every retrain. Measured
2026-07-31: two retrains hours apart were scored on test sets spanning
different days with target means of 60.4 and 46.9 minutes, making their
numbers incomparable. Any "is it improving?" question asked across retrains
was therefore unanswerable.

A forward test fixes the reference point instead of the percentile: train on
everything up to a cutoff, then score against rows that genuinely did not
exist at training time. No leakage is possible by construction, and repeated
runs against the same artifact are comparable to each other.

WHAT THIS STILL CANNOT DO -- read before trusting a number:
  - The window is one contiguous slice of real time, so it inherits that
    slice's composition. The first real run covered 11:02-13:01 local, which
    spans the busy mid-morning and the lunch trough (see config.py's
    DIURNAL_SNAPSHOTS note), so its target mean says as much about those two
    hours as about the model. `window_hours`, and the actual vs predicted
    mean/std, are reported alongside every score precisely so that shift is
    visible rather than silent -- do not quote MAE without them.
  - Below MIN_TRUSTWORTHY_WINDOW_HOURS the headline is suppressed entirely
    rather than shown with a caveat nobody reads.

COVERAGE SEGMENTATION IS THE POINT, not a bonus. Aggregate live-tier scores
are computed only over combos that HAVE live data -- i.e. exactly the combos
that least need the historical_derived_proxy tier. A change that helps those
while quietly degrading the ~1,500 low-coverage combos (whose only signal is
the proxy tier, and which a citizen is just as likely to query) would look
like a clean win in aggregate. Splitting by coverage is what makes that
failure mode observable, so any weighting change must be checked here.

Usage:
    python -m pipeline.forward_test
    python -m pipeline.forward_test --min-live-samples 30
"""

from __future__ import annotations

import argparse
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import DEFAULT_DB_PATH, DEFAULT_MODEL_PATH
from pipeline.db import clean_siga_live_readings, load_all_samples
from pipeline.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN, QueueFeatureTransformer

logger = logging.getLogger(__name__)

# Below this, a headline score is noise: it would cover only part of one
# business day, so hour-of-day composition alone can swing it more than any
# realistic model change. One full day is the minimum that spans a whole
# open-to-close cycle.
MIN_TRUSTWORTHY_WINDOW_HOURS = 24.0

# A combo needs this much real coverage before its proxy rows meaningfully
# decay (pipeline/train.py's compute_sample_weights) -- the same threshold
# pipeline/coverage_report.py reports on, reused so both tools agree on what
# "has live coverage" means.
DEFAULT_MIN_LIVE_SAMPLES = 30


def score(actual: pd.Series, predicted: np.ndarray) -> dict[str, float | int | None]:
    """MAE/RMSE/R^2 plus the distribution stats needed to interpret them.

    actual_mean/std and predicted_mean/std are not decoration: a model can
    post a respectable MAE while systematically under-predicting (found
    2026-07-31: predicted mean 18.1 against an actual mean of 41.4, with
    predicted std about a third of actual). Only the distribution stats make
    that visible.
    """
    if len(actual) == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None,
                "actual_mean": None, "actual_std": None, "predicted_mean": None, "predicted_std": None}
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        # R^2 needs at least two distinct target values to be defined at all.
        "r2": float(r2_score(actual, predicted)) if actual.nunique() > 1 else None,
        "actual_mean": float(actual.mean()),
        "actual_std": float(actual.std()),
        "predicted_mean": float(np.mean(predicted)),
        "predicted_std": float(np.std(predicted)),
    }


def build_forward_test_frame(db_path: str, data_cutoff: str) -> pd.DataFrame:
    """Rows strictly newer than `data_cutoff`, with features built.

    Features are built over the FULL history and only then filtered, not
    built over the unseen slice alone: rolling wait features look backwards,
    so a row's features legitimately depend on readings that preceded it.
    Cutting first would starve the earliest unseen rows of that context and
    make them look artificially unlike serving conditions. This is not
    leakage -- every input still predates the row being scored.
    """
    frame = load_all_samples(db_path)
    frame = clean_siga_live_readings(frame)
    frame = frame.dropna(subset=[TARGET_COLUMN]).sort_values("sampled_at").reset_index(drop=True)

    features = QueueFeatureTransformer(db_path=db_path).transform(frame)
    features[TARGET_COLUMN] = frame[TARGET_COLUMN]
    features["sampled_at"] = frame["sampled_at"]
    features["source"] = frame["source"]
    features["branch_id_raw"] = frame["branch_id"]
    features["desk_service_id_raw"] = frame["desk_service_id"]

    cutoff = pd.Timestamp(data_cutoff)
    return features[features["sampled_at"] > cutoff]


def live_coverage_counts(db_path: str, data_cutoff: str) -> pd.Series:
    """Usable live samples per (branch, service) AS OF the training cutoff.

    Deliberately counts only rows the model could have learned from, not
    rows in the unseen window -- the question is "did this combo have real
    coverage when the model was built", so counting the window itself would
    let a combo be graded as well-covered on the strength of data that
    arrived afterwards.
    """
    frame = load_all_samples(db_path)
    frame = clean_siga_live_readings(frame)
    live = frame[(frame["source"] == "siga_live") & frame[TARGET_COLUMN].notna()]
    live = live[live["sampled_at"] <= pd.Timestamp(data_cutoff)]
    return live.groupby(["branch_id", "desk_service_id"]).size()


def run_forward_test(
    db_path: str = DEFAULT_DB_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    min_live_samples: int = DEFAULT_MIN_LIVE_SAMPLES,
) -> dict:
    artifact = joblib.load(model_path)
    data_cutoff = artifact.get("data_cutoff")
    if data_cutoff is None:
        raise SystemExit(
            f"{model_path} predates the data_cutoff field (added 2026-07-31) — retrain before "
            "forward-testing. Falling back to trained_at would silently score the model against "
            "its own training rows; see pipeline/train.py's data_cutoff comment."
        )

    unseen = build_forward_test_frame(db_path, data_cutoff)
    if unseen.empty:
        return {"data_cutoff": data_cutoff, "window_hours": 0.0, "overall": score(pd.Series(dtype=float), np.array([])),
                "by_source": {}, "by_coverage": {}, "trustworthy": False}

    model = artifact["model"]
    # The ARTIFACT's own feature list, never the module-level FEATURE_COLUMNS.
    # Using the current constant makes it impossible to score a model built
    # before a feature change — which is exactly the comparison a forward test
    # exists for. Found 2026-08-04 trying to score the pre-holiday backup
    # (15 features) against a 17-feature codebase: XGBoost raised a
    # feature_names mismatch and the head-to-head was unrunnable.
    # api/service.py has always read the artifact's list; this now matches it.
    feature_columns = artifact.get("feature_columns", FEATURE_COLUMNS)
    missing = [column for column in feature_columns if column not in unseen.columns]
    if missing:
        raise SystemExit(
            f"{model_path} needs feature(s) the current pipeline no longer produces: {missing}. "
            "Scoring it would require the feature engineering it was trained with."
        )
    predictions = model.predict(unseen[feature_columns])
    unseen = unseen.assign(_prediction=predictions)

    window_hours = float((unseen["sampled_at"].max() - unseen["sampled_at"].min()).total_seconds() / 3600)

    by_source = {
        name: score(subset[TARGET_COLUMN], subset["_prediction"].to_numpy())
        for name, subset in unseen.groupby("source")
    }

    # Coverage split applies to live rows only: proxy/daily-avg rows are not
    # per-combo live observations, so grading them by live coverage would
    # conflate two different questions.
    live = unseen[unseen["source"] == "siga_live"]
    by_coverage: dict[str, dict] = {}
    if not live.empty:
        counts = live_coverage_counts(db_path, data_cutoff)
        keys = list(zip(live["branch_id_raw"], live["desk_service_id_raw"]))
        covered = pd.Series([counts.get(k, 0) >= min_live_samples for k in keys], index=live.index)
        by_coverage = {
            f"live_covered_ge_{min_live_samples}": score(live.loc[covered, TARGET_COLUMN], live.loc[covered, "_prediction"].to_numpy()),
            f"live_sparse_lt_{min_live_samples}": score(live.loc[~covered, TARGET_COLUMN], live.loc[~covered, "_prediction"].to_numpy()),
        }

    return {
        "data_cutoff": data_cutoff,
        "window_start": unseen["sampled_at"].min().isoformat(),
        "window_end": unseen["sampled_at"].max().isoformat(),
        "window_hours": round(window_hours, 2),
        "trustworthy": window_hours >= MIN_TRUSTWORTHY_WINDOW_HOURS,
        "overall": score(unseen[TARGET_COLUMN], unseen["_prediction"].to_numpy()),
        "by_source": by_source,
        "by_coverage": by_coverage,
    }


def _format(label: str, m: dict) -> str:
    if not m["n"]:
        return f"  {label:34s} (no rows)"
    r2 = f"{m['r2']:.4f}" if m["r2"] is not None else "N/A"
    return (
        f"  {label:34s} n={m['n']:>7d}  MAE={m['mae']:7.2f}  RMSE={m['rmse']:7.2f}  R^2={r2:>8s}"
        f"   actual {m['actual_mean']:6.1f}±{m['actual_std']:<6.1f} pred {m['predicted_mean']:6.1f}±{m['predicted_std']:<6.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the saved model on data newer than it")
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--min-live-samples", type=int, default=DEFAULT_MIN_LIVE_SAMPLES)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_forward_test(args.db, args.model, args.min_live_samples)

    if not result["overall"]["n"]:
        print(f"No data newer than the model's cutoff ({result['data_cutoff']}) — nothing to forward-test yet.")
        return

    print(f"Forward test — rows that did not exist when this model was trained")
    print(f"  model data cutoff : {result['data_cutoff']}")
    print(f"  unseen window     : {result['window_start']} -> {result['window_end']}  ({result['window_hours']}h)")
    print()
    if not result["trustworthy"]:
        print(f"  !! Window is under {MIN_TRUSTWORTHY_WINDOW_HOURS:.0f}h, so these are INDICATIVE ONLY — a partial day is")
        print(f"     dominated by which hours it happens to cover, not by model quality.")
        print()
    print(_format("ALL unseen rows", result["overall"]))
    print()
    print("By source:")
    for name, m in result["by_source"].items():
        print(_format(name, m))
    if result["by_coverage"]:
        print()
        print("Live rows by that combo's coverage at training time:")
        print("  (a change helping covered combos while hurting sparse ones is the")
        print("   failure mode aggregate metrics cannot see — compare these two)")
        for name, m in result["by_coverage"].items():
            print(_format(name, m))


if __name__ == "__main__":
    main()
