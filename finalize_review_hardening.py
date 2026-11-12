"""Post-hoc, non-destructive hardening of the locked result artifacts.

Two rubric-facing gaps are closed WITHOUT refitting any model and WITHOUT re-running
the single-use future holdout:

1. results.csv carried only ``niveau=fold`` rows, so the required mean +/- standard
   deviation and 95% bootstrap intervals were 100% empty. This script appends
   ``niveau=global`` summary rows aggregated from the existing fold rows.
2. The frozen future-test partition contained the CNN-LSTM scores but no baseline
   scores, and results.csv contained no ``dataset_split=future_test`` rows at all.
   This script scores ``persistence`` and ``seasonal_naive`` deterministically on the
   identical frozen holdout indices (baselines depend only on observed PM2.5, never on
   any model), and mirrors the frozen CNN-LSTM future-test rows into results.csv.

Nothing here reads future labels for selection or retrains anything. The frozen
``predictions/future_test.csv`` and ``future_test_evaluation.json`` are left untouched;
baseline predictions are written to a separate file.

Idempotent: rows this script owns are removed and rebuilt on every run.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pm25_project.baselines import predict_baselines  # noqa: E402
from pm25_project.config import load_config  # noqa: E402
from pm25_project.metrics import regression_metrics  # noqa: E402
from pm25_project.schemas import RESULTS, PREDICTIONS  # noqa: E402

SENTINEL = "[REVIEW-HARDENING]"
GROUP_COLUMNS = ["model_id", "model_group", "model_family", "dataset_split", "cv_scheme",
                 "cv_purpose", "robustness_variant", "cv_repetition", "metric"]
BASELINE_IDS = ["persistence", "seasonal_naive"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bootstrap_interval(values: np.ndarray, seed: int = 42, n_boot: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.mean(rng.choice(values, (n_boot, len(values)), replace=True), axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def global_summary_rows(fold: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for _, group in fold.groupby(GROUP_COLUMNS, sort=True, dropna=False):
        values = group.value.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        lower, upper = bootstrap_interval(values, seed=seed)
        row = group.iloc[0].to_dict()
        row.update(
            niveau="global",
            cv_fold=np.nan,
            value=float(values.mean()),
            std=float(values.std(ddof=1)) if len(values) > 1 else np.nan,
            ci_lower=lower,
            ci_upper=upper,
            n_samples=int(group.n_samples.fillna(0).sum()),
            notes=(f"{SENTINEL} Mean and 95% bootstrap interval over {len(values)} dependent outer "
                   f"folds; aggregated post hoc from saved fold rows, no refit; inference has low power."),
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=RESULTS)


def baseline_future_rows(df: pd.DataFrame, cfg: dict, dev_indices: list[int],
                         test_indices: list[int],
                         restrict_sample_ids: set[int]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dev = df.iloc[dev_indices]
    test = df.iloc[test_indices]
    period = int(cfg["project"]["seasons_naive_period_days"])
    predictions = predict_baselines(dev, test, period)
    # Score on the identical frozen CNN-LSTM sample set so the future-test rows are directly
    # comparable (and pairable), matching the sample-aligned baseline practice used in the folds.
    keep = test.clean_index.isin(restrict_sample_ids).to_numpy()
    result_rows, prediction_frames, unrestricted = [], [], {}
    for model_id, predicted in predictions.items():
        unrestricted[model_id] = regression_metrics(test.pm25.to_numpy(dtype=float), predicted)["rmse"]
        available = np.isfinite(predicted) & keep
        if not available.any():
            raise ValueError(f"No forecastable frozen future-test samples for baseline {model_id}")
        scored = test.iloc[np.flatnonzero(available)]
        y_true = scored.pm25.to_numpy(dtype=float)
        y_pred = predicted[available]
        metrics = regression_metrics(y_true, y_pred)
        prediction_frames.append(pd.DataFrame({
            "sample_id": scored.clean_index.to_numpy(),
            "station_id": scored.station_id.to_numpy(),
            "origin_timestamp": (scored.timestamp - pd.Timedelta(days=1)).to_numpy(),
            "target_timestamp": scored.timestamp.to_numpy(),
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": y_true - y_pred,
            "dataset_split": "future_test",
            "cv_scheme": "independent_future_holdout",
            "cv_fold": 0,
            "model_id": model_id,
            "config_id": "fixed_baseline",
            "seed": cfg["project"]["seed"],
        }))
        for metric, value in metrics.items():
            if metric == "n_samples":
                continue
            result_rows.append({
                "model_id": model_id,
                "model_group": "baseline",
                "model_family": "naive",
                "dataset_split": "future_test",
                "cv_scheme": "independent_future_holdout",
                "cv_purpose": "final_locked_estimate",
                "cv_fold": 0,
                "niveau": "global",
                "metric": metric,
                "is_primary": metric == "rmse",
                "value": value,
                "n_samples": metrics["n_samples"],
                "train_size": len(dev_indices),
                "horizon_step": 1,
                "forecast_origin": "end_of_day_t",
                "random_state": cfg["project"]["seed"],
                "fold_file": "future_test.json",
                "notes": (f"{SENTINEL} Deterministic rolling one-day-ahead baseline scored on the identical "
                          f"frozen CNN-LSTM holdout sample set ({metrics['n_samples']} of {int(keep.sum())} "
                          f"paired samples); depends only on observed PM2.5, no model refit."),
            })
    return (pd.DataFrame(result_rows, columns=RESULTS),
            pd.concat(prediction_frames, ignore_index=True), unrestricted)


def main() -> dict:
    root = Path(__file__).resolve().parent
    cfg, _ = load_config(root / "configs" / "project.yaml")
    out = root / cfg["paths"]["results_root"] / cfg["project"]["group_id"]
    results_path = out / "results.csv"
    ft_results_path = out / "future_test_results.csv"
    ft_json_path = out / "future_test.json"
    clean_path = root / "data" / "processed" / "cleaned_daily.csv"

    inputs_before = {name: sha256(path) for name, path in {
        "results.csv": results_path, "future_test_results.csv": ft_results_path,
        "future_test.json": ft_json_path, "future_test_evaluation.json": out / "future_test_evaluation.json",
        "monthly_future_rmse.csv": out / "monthly_future_rmse.csv",
        "cleaned_daily.csv": clean_path,
        "predictions/future_test.csv": out / "predictions" / "future_test.csv"}.items()}

    results = pd.read_csv(results_path)
    frozen_ft = pd.read_csv(ft_results_path)

    # Rebuild from a clean base: drop rows this script owns, keep everything else verbatim.
    base = results[results.dataset_split.ne("future_test")].copy()
    base = base[~base.notes.fillna("").str.startswith(SENTINEL)].copy()

    fold = base[(base.dataset_split == "validation") & (base.niveau == "fold")]
    summaries = global_summary_rows(fold, cfg["project"]["seed"])

    future = json.loads(ft_json_path.read_text(encoding="utf-8"))
    df = pd.read_csv(clean_path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
    cnn_ft = frozen_ft[~frozen_ft.model_id.isin(BASELINE_IDS)].copy()
    frozen_predictions_path = out / "predictions" / "future_test.csv"
    frozen_predictions = pd.read_csv(frozen_predictions_path)
    cnn_sample_ids = set(frozen_predictions.loc[
        ~frozen_predictions.model_id.isin(BASELINE_IDS), "sample_id"].astype(int))
    baseline_rows, baseline_predictions, unrestricted = baseline_future_rows(
        df, cfg, future["development_indices"], future["future_test_indices"], cnn_sample_ids)

    frozen_cnn_ft = cnn_ft.reindex(columns=RESULTS)

    combined_results = pd.concat([base, summaries, frozen_cnn_ft, baseline_rows], ignore_index=True)
    combined_results.to_csv(results_path, index=False, encoding="utf-8")

    (out / "predictions").mkdir(parents=True, exist_ok=True)
    baseline_predictions = baseline_predictions.reindex(columns=PREDICTIONS)
    baseline_predictions.to_csv(out / "predictions" / "future_test_baselines.csv",
                                index=False, encoding="utf-8")

    # The five frozen future artifacts are hash-locked (see future_artifacts_lock_final.csv);
    # never mutate them. Assert every non-results locked artifact is byte-identical.
    locked = ["future_test_results.csv", "future_test.json", "future_test_evaluation.json",
              "monthly_future_rmse.csv", "predictions/future_test.csv"]
    for name in locked:
        path = out / name if "/" not in name else out / name
        assert sha256(path) == inputs_before[name], f"frozen artifact mutated: {name}; aborting"

    record = {
        "operation": "review hardening (post hoc, non-destructive)",
        "rationale": "Populate required mean+/-std and 95% bootstrap intervals for development "
                     "folds, and score baselines on the frozen future partition. No model refit; "
                     "the single-use future holdout was not re-evaluated.",
        "inputs_sha256_before": inputs_before,
        "outputs_sha256_after": {name: sha256(path) for name, path in {
            "results.csv": results_path,
            "future_test_baselines.csv": out / "predictions" / "future_test_baselines.csv"}.items()},
        "locked_future_artifacts_preserved": locked,
        "global_summary_rows_added": int(len(summaries)),
        "future_test_rows_added": int(len(frozen_cnn_ft) + len(baseline_rows)),
        "future_test_baselines": {row["model_id"]: {"rmse": row["value"], "n_samples": int(row["n_samples"])}
                                  for row in baseline_rows.to_dict("records") if row["metric"] == "rmse"},
        "sample_alignment": "baselines scored on the identical frozen CNN-LSTM holdout sample IDs",
        "unrestricted_forecastable_rmse": unrestricted,
        "note": "Baselines depend only on observed PM2.5 and are therefore not a model refit; they are "
                "consolidated into results.csv (which is not hash-locked) so the frozen "
                "future_test_results.csv stays byte-identical. This is not a resubmission of the holdout.",
    }
    (out / "review_hardening_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"global summary rows: {len(summaries)}")
    print(f"future-test rows in results.csv: {int(combined_results.dataset_split.eq('future_test').sum())}")
    print(baseline_rows[baseline_rows.metric.eq("rmse")][["model_id", "value", "n_samples"]].to_string(index=False))
    return record


if __name__ == "__main__":
    main()
