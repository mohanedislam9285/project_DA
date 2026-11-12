"""Reconcile saved temporal OOF predictions with inputs, splits, checkpoints, and baselines.

This is a development-only audit. It does not score or read future-test targets.
It writes new audit outputs and does not overwrite the existing evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pm25_project.config import load_config
from pm25_project.features import build_origin_features
from pm25_project.sequences import make_sequences


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "16"
AUDIT = OUT / "reproducibility_audit"
AUDIT.mkdir(parents=True, exist_ok=True)
cfg, _ = load_config(ROOT / "configs" / "project.yaml")
df = pd.read_csv(ROOT / "data" / "processed" / "cleaned_daily.csv", parse_dates=["timestamp"])
df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
df["clean_index"] = pd.to_numeric(df.clean_index, errors="raise").astype(int)
reservation = json.loads((OUT / "future_test.json").read_text(encoding="utf-8"))
dev_indices = np.asarray(reservation["development_indices"], dtype=int)
test_indices = set(map(int, reservation["future_test_indices"]))
assert not (set(dev_indices.tolist()) & test_indices)
dev = df.iloc[dev_indices].copy()
cutoff = pd.to_datetime(reservation["development_end"], utc=True)
assert dev.timestamp.max() <= cutoff

feature_table = build_origin_features(dev, cfg)
columns = list(cfg["features"].get("predictors", [])) + list(cfg["features"].get("calendar", []))
if cfg["features"].get("include_target_lags", True):
    columns.append("pm25_lag1")
if cfg["features"].get("include_past_rolling", True):
    columns.extend(f"pm25_roll{w}_mean" for w in cfg["features"].get("rolling_windows_days", [3, 7]))
columns = list(dict.fromkeys(columns))
lookback = int(cfg["project"].get("lookback_candidates_days", [7])[0])
X, y, keys = make_sequences(feature_table, columns, lookback=lookback, horizon=1)
target_rows = keys.target_row.to_numpy(dtype=int)
source_lookup = df.set_index("clean_index", drop=False)
target_actual = source_lookup.loc[target_rows, "pm25"].to_numpy(dtype=float)
assert np.allclose(y, target_actual, rtol=0, atol=5e-5)  # sequence labels are intentionally float32
assert np.all((pd.to_datetime(keys.target_timestamp, utc=True) - pd.to_datetime(keys.origin_timestamp, utc=True)).dt.days.to_numpy() == 1)
lag_idx = columns.index("pm25_lag1")
origin_lookup = {(str(r.station_id), pd.Timestamp(r.timestamp)): float(r.pm25) for r in dev.itertuples()}
origin_actual = np.asarray([
    origin_lookup[(str(r.station_id), pd.Timestamp(r.origin_timestamp).tz_localize("UTC"))]
    for r in keys.itertuples()
])
assert np.allclose(X[:, -1, lag_idx], origin_actual, rtol=0, atol=5e-5)

folds = json.loads((OUT / "folds_temporal.json").read_text(encoding="utf-8"))["folds"]
saved = pd.read_csv(OUT / "predictions" / "cnn_lstm_optuna_temporal_oof.csv")
saved["sample_id"] = pd.to_numeric(saved.sample_id, errors="raise").astype(int)
fold_rows: list[dict[str, object]] = []
prediction_rows: list[dict[str, object]] = []
alignment_rows: list[dict[str, object]] = []
for fold in folds:
    k = int(fold["fold"])
    train = np.flatnonzero(np.isin(target_rows, np.asarray(fold["train_indices"], dtype=int)))
    valid = np.flatnonzero(np.isin(target_rows, np.asarray(fold["validation_indices"], dtype=int)))
    train_max = pd.to_datetime(keys.iloc[train].target_timestamp, utc=True).max()
    valid_min = pd.to_datetime(keys.iloc[valid].target_timestamp, utc=True).min()
    raw_train_max = source_lookup.loc[np.asarray(fold["train_indices"], dtype=int), "timestamp"].max()
    raw_valid_min = source_lookup.loc[np.asarray(fold["validation_indices"], dtype=int), "timestamp"].min()
    assert train_max < valid_min
    gap_days = int((raw_valid_min - raw_train_max).days - 1)
    assert gap_days == int(fold["embargo_days"]), (k, gap_days)
    assert set(target_rows[train]).issubset(set(map(int, fold["train_indices"])))
    assert set(target_rows[valid]).issubset(set(map(int, fold["validation_indices"])))
    assert set(target_rows[train]).isdisjoint(test_indices)
    assert set(target_rows[valid]).isdisjoint(test_indices)

    model_dir = OUT / "models"
    model_path = model_dir / f"cnn_lstm_optuna_temporal_fold_{k}.keras"
    prep_path = model_dir / f"cnn_lstm_optuna_temporal_fold_{k}_preprocessor.joblib"
    model = tf.keras.models.load_model(model_path)
    prep = joblib.load(prep_path)
    pred_train = model.predict(prep.transform(X[train]), verbose=0).reshape(-1)
    pred_valid = model.predict(prep.transform(X[valid]), verbose=0).reshape(-1)
    y_train, y_valid = y[train], y[valid]
    saved_fold = saved[saved.cv_fold.eq(k)].copy().set_index("sample_id")
    computed_valid = pd.DataFrame({"sample_id": target_rows[valid], "prediction": pred_valid}).set_index("sample_id")
    joined = saved_fold.join(computed_valid, how="outer", rsuffix="_reloaded")
    assert len(joined) == len(valid) and joined.y_pred.notna().all() and joined.prediction.notna().all()
    max_abs_diff = float(np.max(np.abs(joined.y_pred.to_numpy(dtype=float) - joined.prediction.to_numpy(dtype=float))))
    fold_rows.append({
        "cv_fold": k, "train_sequences": int(len(train)), "validation_sequences": int(len(valid)),
        "raw_fold_train_end_utc": raw_train_max.isoformat(), "raw_fold_validation_start_utc": raw_valid_min.isoformat(),
        "scored_train_target_end_utc": train_max.isoformat(), "scored_validation_target_start_utc": valid_min.isoformat(),
        "embargo_days_verified": gap_days,
        "train_rmse": float(mean_squared_error(y_train, pred_train) ** 0.5),
        "train_mae": float(mean_absolute_error(y_train, pred_train)),
        "validation_rmse_reloaded": float(mean_squared_error(y_valid, pred_valid) ** 0.5),
        "validation_mae_reloaded": float(mean_absolute_error(y_valid, pred_valid)),
        "validation_r2_reloaded": float(r2_score(y_valid, pred_valid)),
        "saved_prediction_max_abs_diff": max_abs_diff,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "preprocessor_sha256": hashlib.sha256(prep_path.read_bytes()).hexdigest(),
    })
    for seq_idx, prediction in zip(valid, pred_valid):
        key = keys.iloc[seq_idx]
        row = source_lookup.loc[int(key.target_row)]
        origin = pd.Timestamp(key.origin_timestamp).tz_localize("UTC")
        origin_key = (str(key.station_id), origin)
        persistence = origin_lookup[origin_key]
        prediction_rows.append({
            "sample_id": int(key.target_row), "station_id": str(key.station_id),
            "origin_timestamp": pd.Timestamp(key.origin_timestamp).isoformat(),
            "target_timestamp": pd.Timestamp(key.target_timestamp).isoformat(),
            "y_true": float(y[seq_idx]), "cnn_lstm_prediction_reloaded": float(prediction),
            "persistence_prediction_origin_pm25": float(persistence), "cv_fold": k,
        })

    train_cfg = pd.to_datetime(keys.iloc[train].target_timestamp, utc=True).to_numpy()
    valid_cfg = pd.to_datetime(keys.iloc[valid].target_timestamp, utc=True).to_numpy()
    alignment_rows.append({
        "cv_fold": k, "training_targets": len(train_cfg), "validation_targets": len(valid_cfg),
        "train_validation_target_overlap": int(len(np.intersect1d(train_cfg, valid_cfg))),
        "last_train_target": raw_train_max.isoformat(), "first_validation_target": raw_valid_min.isoformat(),
        "first_scored_validation_target": valid_min.isoformat(),
        "temporal_gap_days_excluding_endpoints": gap_days,
        "validation_sequences_include_only_prior_history": True,
    })

fold_metrics = pd.DataFrame(fold_rows)
predictions = pd.DataFrame(prediction_rows)
fold_metrics.to_csv(AUDIT / "reloaded_fold_metrics_and_fit_diagnostics.csv", index=False)
predictions.to_csv(AUDIT / "reloaded_temporal_oof_and_persistence.csv", index=False)
pd.DataFrame(alignment_rows).to_csv(AUDIT / "temporal_alignment_checks.csv", index=False)

# Verify all five future-lock hashes without reading holdout values for metrics/modeling.
lock = pd.read_csv(OUT / "future_artifacts_lock_initial.csv")
lock_rows = []
for row in lock.to_dict(orient="records"):
    path = Path(row["Path"])
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    lock_rows.append({"path": str(path), "expected_sha256": row["Hash"], "actual_sha256": actual_hash, "match": actual_hash.lower() == str(row["Hash"]).lower()})
hash_ok = all(r["match"] for r in lock_rows)
assert hash_ok
pd.DataFrame(lock_rows).to_csv(AUDIT / "future_artifact_hash_verification.csv", index=False)

# Record source lineage and verified facts without reporting future-test performance.
report = {
    "purpose": "Development-only reconciliation of sequence alignment, model reload, validation scores, persistence baselines, and split chronology.",
    "development_cutoff_inclusive": cutoff.isoformat(), "future_holdout_values_used_for_modeling_or_metrics": False,
    "lookback_days": lookback, "forecast_horizon_days": 1,
    "feature_columns": columns, "n_development_sequences": int(len(y)),
    "forecast_alignment_verified": True, "last_input_pm25_equals_origin_observation": True,
    "target_equals_next_day_source_row": True, "all_train_validation_embargoes_verified": True,
    "max_reloaded_prediction_difference": float(fold_metrics.saved_prediction_max_abs_diff.max()),
    "future_hashes_all_match": hash_ok, "future_hashes": lock_rows,
    "fold_metrics": fold_rows,
    "note": "RMSE/MAE confidence intervals are not estimated here. Fold comparisons and station-cluster bootstrap are computed by a separate development-only analysis.",
}
(AUDIT / "temporal_prediction_reconciliation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({
    "n_development_sequences": report["n_development_sequences"],
    "max_reloaded_prediction_difference": report["max_reloaded_prediction_difference"],
    "future_hashes_all_match": hash_ok,
    "folds": [{k: r[k] for k in ("cv_fold", "train_rmse", "validation_rmse_reloaded", "saved_prediction_max_abs_diff", "embargo_days_verified")} for r in fold_rows],
}, indent=2))
