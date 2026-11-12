"""Nested development-only CNN-LSTM feature-family sensitivity on common OOF rows.

Four predeclared predictor families are compared within inner chronological
validation. Each uses the fold's previously selected TPE parameters. Outer-fold
targets are identical for all variants; no future-test labels are loaded.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pm25_project.config import load_config
from pm25_project.features import build_origin_features
from pm25_project.models.cnn_lstm import build_cnn_lstm
from pm25_project.preprocessing import SequencePreprocessor
from pm25_project.sequences import make_sequences
from pm25_project.tuning import chronological_inner_indices

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "16"
EXP = OUT / "feature_family_experiment"
for sub in ("models", "histories"):
    (EXP / sub).mkdir(parents=True, exist_ok=True)
cfg, _ = load_config(ROOT / "configs" / "project.yaml")
reservation = json.loads((OUT / "future_test.json").read_text(encoding="utf-8"))
df = pd.read_csv(ROOT / "data" / "processed" / "cleaned_daily.csv", parse_dates=["timestamp"])
df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
dev = df.iloc[np.asarray(reservation["development_indices"], dtype=int)].copy()
cutoff = pd.to_datetime(reservation["development_end"], utc=True)
if dev.timestamp.max() > cutoff:
    raise ValueError("Development sequence crosses the saved cutoff")
features = build_origin_features(dev, cfg)
base_columns = list(cfg["features"].get("predictors", [])) + list(cfg["features"].get("calendar", []))
if cfg["features"].get("include_target_lags", True):
    base_columns.append("pm25_lag1")
if cfg["features"].get("include_past_rolling", True):
    base_columns.extend(f"pm25_roll{w}_mean" for w in cfg["features"].get("rolling_windows_days", [3, 7]))
base_columns = list(dict.fromkeys(base_columns))
calendar = set(cfg["features"].get("calendar", []))
variants = {
    "full_calendar_lag_roll3_roll7": base_columns,
    "no_calendar_lag_roll3_roll7": [c for c in base_columns if c not in calendar],
    "no_calendar_lag_roll3": [c for c in base_columns if c not in calendar and c != "pm25_roll7_mean"],
    "no_calendar_lag_only": [c for c in base_columns if c not in calendar and c not in {"pm25_roll3_mean", "pm25_roll7_mean"}],
}
lookback = int(cfg["project"].get("lookback_candidates_days", [7])[0])
arrays = {}
common_ids: set[int] | None = None
for name, cols in variants.items():
    X, y, keys = make_sequences(features, cols, lookback=lookback, horizon=1)
    ids = set(keys.target_row.astype(int))
    common_ids = ids if common_ids is None else common_ids & ids
    arrays[name] = (X, y, keys, cols)
if common_ids is None or not common_ids:
    raise RuntimeError("No common target IDs across feature families")
reference_ids = set(arrays[next(iter(variants))][2].target_row.astype(int))
if any(set(a[2].target_row.astype(int)) != reference_ids for a in arrays.values()):
    raise ValueError("Feature families do not share the same prediction samples")
folds = json.loads((OUT / "folds_temporal.json").read_text(encoding="utf-8"))["folds"]
inner_rows, outer_rows, prediction_rows, selected_rows = [], [], [], []

for fold in folds:
    k = int(fold["fold"])
    tuned = json.loads((OUT / "models" / f"cnn_lstm_optuna_temporal_fold_{k}_config.json").read_text(encoding="utf-8"))
    params = {n: tuned[n] for n in ("filters", "kernel_size", "lstm_units", "dense_units", "dropout", "learning_rate", "loss")}
    seed = int(tuned["seed"])
    batch = int(cfg["model"]["cnn_lstm"].get("batch_size", 64))
    fold_idxs = {}
    inner_scores, epochs = {}, {}
    for name, (X, y, keys, cols) in arrays.items():
        target_rows = keys.target_row.to_numpy(dtype=int)
        train = np.flatnonzero(np.isin(target_rows, np.asarray(fold["train_indices"], dtype=int)))
        valid = np.flatnonzero(np.isin(target_rows, np.asarray(fold["validation_indices"], dtype=int)))
        if not len(train) or not len(valid):
            raise ValueError(f"No common train/validation samples in fold {k}")
        fold_idxs[name] = (train, valid)
        dates = pd.to_datetime(keys.iloc[train].target_timestamp, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
        itr, iva = chronological_inner_indices(dates)
        prep = SequencePreprocessor().fit(X[train[itr]])
        xi, xv = prep.transform(X[train[itr]]), prep.transform(X[train[iva]])
        tf.keras.backend.clear_session()
        model = build_cnn_lstm(lookback, xi.shape[-1], {**params, "seed": seed, "pool_size": 2})
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
        ]
        hist = model.fit(xi, y[train[itr]], validation_data=(xv, y[train[iva]]), epochs=100,
                         batch_size=batch, callbacks=callbacks, verbose=0)
        pred = model.predict(xv, verbose=0).reshape(-1)
        inner_rmse = float(mean_squared_error(y[train[iva]], pred) ** 0.5)
        best_epoch = int(np.argmin(hist.history["val_loss"]) + 1)
        inner_scores[name], epochs[name] = inner_rmse, best_epoch
        inner_rows.append({"cv_fold": k, "variant": name, "features": json.dumps(cols),
                           "inner_train_sequences": len(itr), "inner_validation_sequences": len(iva),
                           "inner_validation_rmse": inner_rmse,
                           "inner_validation_mae": float(mean_absolute_error(y[train[iva]], pred)),
                           "best_epoch": best_epoch, "min_train_loss": float(np.min(hist.history["loss"])),
                           "min_validation_loss": float(np.min(hist.history["val_loss"]))})
        pd.DataFrame(hist.history).to_csv(EXP / "histories" / f"fold{k}_{name}_inner.csv", index=False)

    chosen = min(variants, key=lambda name: (inner_scores[name], name))
    selected_rows.append({"cv_fold": k, "selected_variant": chosen, "inner_validation_rmse": inner_scores[chosen],
                          "best_epoch": epochs[chosen], "selection_source": "inner chronological validation only"})
    for name, (X, y, keys, cols) in arrays.items():
        train, valid = fold_idxs[name]
        prep = SequencePreprocessor().fit(X[train])
        xt, xv = prep.transform(X[train]), prep.transform(X[valid])
        tf.keras.backend.clear_session()
        model = build_cnn_lstm(lookback, xt.shape[-1], {**params, "seed": seed, "pool_size": 2})
        started = time.perf_counter()
        history = model.fit(xt, y[train], epochs=epochs[name], batch_size=batch, verbose=0)
        fit_seconds = time.perf_counter() - started
        pred_train = model.predict(xt, verbose=0).reshape(-1)
        pred_valid = model.predict(xv, verbose=0).reshape(-1)
        rmse = float(mean_squared_error(y[valid], pred_valid) ** 0.5)
        mae = float(mean_absolute_error(y[valid], pred_valid))
        model.save(EXP / "models" / f"fold{k}_{name}.keras")
        joblib.dump(prep, EXP / "models" / f"fold{k}_{name}_preprocessor.joblib")
        model_cfg = {**params, "seed": seed, "lookback_days": lookback, "pool_size": 2,
                     "best_epoch": epochs[name], "inner_validation_rmse": inner_scores[name],
                     "feature_columns": cols, "outer_validation_used_for_selection": False}
        (EXP / "models" / f"fold{k}_{name}_config.json").write_text(json.dumps(model_cfg, indent=2), encoding="utf-8")
        outer_rows.append({"cv_fold": k, "variant": name, "selected_by_inner_validation": name == chosen,
                           "outer_train_sequences": len(train), "outer_validation_sequences": len(valid),
                           "best_epoch": epochs[name], "inner_validation_rmse": inner_scores[name],
                           "train_rmse_in_sample": float(mean_squared_error(y[train], pred_train) ** 0.5),
                           "outer_validation_rmse": rmse, "outer_validation_mae": mae,
                           "outer_validation_r2": float(r2_score(y[valid], pred_valid)), "fit_seconds": fit_seconds})
        base_idx = cols.index("pm25_lag1")
        for j, seq_idx in enumerate(valid):
            key = keys.iloc[seq_idx]
            prediction_rows.append({"sample_id": int(key.target_row), "station_id": str(key.station_id),
                                    "y_true": float(y[seq_idx]), "y_pred": float(pred_valid[j]),
                                    "persistence": float(X[seq_idx, -1, base_idx]), "cv_fold": k,
                                    "variant": name, "selected_by_inner_validation": name == chosen,
                                    "origin_timestamp": pd.Timestamp(key.origin_timestamp).isoformat()+"+00:00",
                                    "target_timestamp": pd.Timestamp(key.target_timestamp).isoformat()+"+00:00"})
        pd.DataFrame(history.history).to_csv(EXP / "histories" / f"fold{k}_{name}_outer.csv", index=False)
        tf.keras.backend.clear_session()

pd.DataFrame(inner_rows).to_csv(EXP / "inner_feature_family_scores.csv", index=False)
pd.DataFrame(outer_rows).to_csv(EXP / "outer_feature_family_scores.csv", index=False)
pd.DataFrame(prediction_rows).to_csv(EXP / "feature_family_oof_predictions.csv", index=False)
pd.DataFrame(selected_rows).to_csv(EXP / "selected_feature_family_by_fold.csv", index=False)
metadata = {"experiment": "nested four-way feature-family sensitivity; development temporal folds only",
            "variants": {k: v for k, v in variants.items()}, "common_oof_sample_count": len(common_ids),
            "inner_selection_metric": "RMSE", "existing_fold_tpe_parameters_fixed": True,
            "outer_validation_used_for_selection": False, "future_test_labels_or_metrics_used": False,
            "status": "exploratory development sensitivity; not used to alter frozen holdout"}
(EXP / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps({"common_oof_sample_count": len(common_ids), "selected_by_fold": selected_rows,
                  "outer_scores": outer_rows}, indent=2))
