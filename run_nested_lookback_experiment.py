"""Nested 3-vs-7-day CNN-LSTM lookback comparison on common OOF target rows."""
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
EXP = OUT / "lookback_experiment"
(EXP / "models").mkdir(parents=True, exist_ok=True)
(EXP / "histories").mkdir(parents=True, exist_ok=True)
cfg, _ = load_config(ROOT / "configs" / "project.yaml")
df = pd.read_csv(ROOT / "data" / "processed" / "cleaned_daily.csv", parse_dates=["timestamp"])
df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
reservation = json.loads((OUT / "future_test.json").read_text(encoding="utf-8"))
dev = df.iloc[np.asarray(reservation["development_indices"], dtype=int)].copy()
cutoff = pd.to_datetime(reservation["development_end"], utc=True)
if dev.timestamp.max() > cutoff:
    raise ValueError("Development rows cross the reserved holdout cutoff")
feature_table = build_origin_features(dev, cfg)
columns = list(cfg["features"].get("predictors", [])) + list(cfg["features"].get("calendar", []))
if cfg["features"].get("include_target_lags", True):
    columns.append("pm25_lag1")
if cfg["features"].get("include_past_rolling", True):
    columns.extend(f"pm25_roll{w}_mean" for w in cfg["features"].get("rolling_windows_days", [3, 7]))
columns = list(dict.fromkeys(columns))
lookbacks = [3, 7]
arrays = {}
for window in lookbacks:
    X, y, keys = make_sequences(feature_table, columns, lookback=window, horizon=1)
    arrays[window] = (X, y, keys, keys.target_row.to_numpy(dtype=int))
common_ids = set.intersection(*(set(a[3]) for a in arrays.values()))
folds = json.loads((OUT / "folds_temporal.json").read_text(encoding="utf-8"))["folds"]
inner_rows, outer_rows, prediction_rows, selected_rows = [], [], [], []

for fold in folds:
    k = int(fold["fold"])
    tuned = json.loads((OUT / "models" / f"cnn_lstm_optuna_temporal_fold_{k}_config.json").read_text(encoding="utf-8"))
    params = {name: tuned[name] for name in ("filters", "kernel_size", "lstm_units", "dense_units", "dropout", "learning_rate", "loss")}
    seed = int(tuned["seed"])
    batch = int(cfg["model"]["cnn_lstm"].get("batch_size", 64))
    scores, epochs_by_window, fold_arrays = {}, {}, {}
    for window in lookbacks:
        X, y, keys, target_rows = arrays[window]
        common_mask = np.isin(target_rows, list(common_ids))
        train = np.flatnonzero(np.isin(target_rows, np.asarray(fold["train_indices"], dtype=int)) & common_mask)
        valid = np.flatnonzero(np.isin(target_rows, np.asarray(fold["validation_indices"], dtype=int)) & common_mask)
        if not len(train) or not len(valid):
            raise ValueError(f"Lookback {window}, fold {k}: no common aligned train/validation samples")
        fold_arrays[window] = (X, y, keys, train, valid)
    # The common OOF target set makes inner train and validation dates identical across lookbacks.
    date_sets = [set(pd.to_datetime(fold_arrays[w][2].iloc[fold_arrays[w][3]].target_timestamp, utc=True).astype(str)) for w in lookbacks]
    if not all(x == date_sets[0] for x in date_sets[1:]):
        # Compare sample-level rows below; dates can repeat across stations, so sets alone are only a sanity check.
        ids_by_window = [set(map(int, arrays[w][2].iloc[fold_arrays[w][3]].target_row)) for w in lookbacks]
        if not all(x == ids_by_window[0] for x in ids_by_window[1:]):
            raise ValueError("Lookback variants do not share exactly the same outer-training target rows")
    for window in lookbacks:
        X, y, keys, train, valid = fold_arrays[window]
        inner_train, inner_valid = chronological_inner_indices(pd.to_datetime(keys.iloc[train].target_timestamp, utc=True).to_numpy())
        prep = SequencePreprocessor().fit(X[train[inner_train]])
        xi, xv = prep.transform(X[train[inner_train]]), prep.transform(X[train[inner_valid]])
        tf.keras.backend.clear_session()
        # Keep the tuned filter/LSTM/dropout/optimizer settings, while constraining
        # the convolution kernel to fit the candidate's shorter input window.
        model = build_cnn_lstm(window, xi.shape[-1], {**params, "kernel_size": min(params["kernel_size"], window), "seed": seed})
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
        ]
        hist = model.fit(xi, y[train[inner_train]], validation_data=(xv, y[train[inner_valid]]),
                         epochs=100, batch_size=batch, callbacks=callbacks, verbose=0)
        inner_prediction = model.predict(xv, verbose=0).reshape(-1)
        rmse = float(mean_squared_error(y[train[inner_valid]], inner_prediction)**0.5)
        best_epoch = int(np.argmin(hist.history["val_loss"]) + 1)
        scores[window], epochs_by_window[window] = rmse, best_epoch
        inner_rows.append({"cv_fold": k, "lookback_days": window, "inner_validation_sequences": len(inner_valid),
                           "inner_validation_rmse": rmse, "inner_validation_mae": float(mean_absolute_error(y[train[inner_valid]], inner_prediction)),
                           "best_epoch": best_epoch, "min_train_loss": float(np.min(hist.history["loss"])),
                           "min_validation_loss": float(np.min(hist.history["val_loss"])),
                           "last_validation_loss": float(hist.history["val_loss"][-1])})
        pd.DataFrame(hist.history).to_csv(EXP / "histories" / f"fold{k}_lookback{window}_inner.csv", index=False)
    chosen = min(lookbacks, key=lambda w: (scores[w], w))
    selected_rows.append({"cv_fold": k, "selected_lookback_days": chosen,
                          "inner_validation_rmse": scores[chosen], "best_epoch": epochs_by_window[chosen],
                          "selection_source": "inner chronological validation only"})
    for window in lookbacks:
        X, y, keys, train, valid = fold_arrays[window]
        prep = SequencePreprocessor().fit(X[train])
        xt, xv = prep.transform(X[train]), prep.transform(X[valid])
        tf.keras.backend.clear_session()
        model = build_cnn_lstm(window, xt.shape[-1], {**params, "kernel_size": min(params["kernel_size"], window), "seed": seed})
        started = time.perf_counter()
        history = model.fit(xt, y[train], epochs=epochs_by_window[window], batch_size=batch, verbose=0)
        fit_time = time.perf_counter() - started
        pred_train = model.predict(xt, verbose=0).reshape(-1)
        pred_valid = model.predict(xv, verbose=0).reshape(-1)
        train_rmse = float(mean_squared_error(y[train], pred_train)**0.5)
        outer_rmse = float(mean_squared_error(y[valid], pred_valid)**0.5)
        outer_mae = float(mean_absolute_error(y[valid], pred_valid))
        outer_r2 = float(r2_score(y[valid], pred_valid))
        model_path = EXP / "models" / f"fold{k}_lookback{window}.keras"
        model.save(model_path)
        joblib.dump(prep, EXP / "models" / f"fold{k}_lookback{window}_preprocessor.joblib")
        model_cfg = {**params, "seed": seed, "lookback_days": window, "best_epoch": epochs_by_window[window],
                     "effective_kernel_size": min(params["kernel_size"], window),
                     "inner_validation_rmse": scores[window], "selected_by_inner_validation": window == chosen,
                     "feature_columns": columns, "outer_validation_used_for_selection": False}
        (EXP / "models" / f"fold{k}_lookback{window}_config.json").write_text(json.dumps(model_cfg, indent=2), encoding="utf-8")
        outer_rows.append({"cv_fold": k, "lookback_days": window, "selected_by_inner_validation": window == chosen,
                           "outer_train_sequences": len(train), "outer_validation_sequences": len(valid),
                           "best_epoch": epochs_by_window[window], "inner_validation_rmse": scores[window],
                           "train_rmse_in_sample": train_rmse, "outer_validation_rmse": outer_rmse,
                           "outer_validation_mae": outer_mae, "outer_validation_r2": outer_r2, "fit_seconds": fit_time})
        for j, seq in enumerate(valid):
            key = keys.iloc[seq]
            prediction_rows.append({"sample_id": int(key.target_row), "station_id": str(key.station_id),
                                    "y_true": float(y[seq]), "y_pred": float(pred_valid[j]),
                                    "persistence": float(X[seq, -1, columns.index("pm25_lag1")]),
                                    "origin_timestamp": pd.Timestamp(key.origin_timestamp).isoformat()+"+00:00",
                                    "target_timestamp": pd.Timestamp(key.target_timestamp).isoformat()+"+00:00",
                                    "cv_fold": k, "lookback_days": window, "selected_by_inner_validation": window == chosen})
        pd.DataFrame(history.history).to_csv(EXP / "histories" / f"fold{k}_lookback{window}_outer.csv", index=False)
        tf.keras.backend.clear_session()

pd.DataFrame(inner_rows).to_csv(EXP / "inner_lookback_scores.csv", index=False)
pd.DataFrame(outer_rows).to_csv(EXP / "outer_lookback_scores.csv", index=False)
pd.DataFrame(prediction_rows).to_csv(EXP / "lookback_oof_predictions.csv", index=False)
pd.DataFrame(selected_rows).to_csv(EXP / "selected_lookback_by_fold.csv", index=False)
metadata = {
    "experiment": "nested 3-vs-7-day lookback comparison, development temporal CV only",
    "lookbacks_days": lookbacks, "common_development_target_rows": int(len(common_ids)),
    "fixed_hyperparameters": "each fold's previously selected Optuna config; no new outer-validation tuning",
    "selection": "lowest inner chronological-validation RMSE; then refit and evaluate on the same outer fold",
    "future_test_used_for_scoring_or_selection": False,
    "available_lookback_14_sequences_by_fold": {str(k): {"train": 0, "validation": 0} for k in []},
    "interpretation": "14-day history has only 505 development sequences overall (fold validation counts 45, 2, 8, 145, 255), so it was not included as a candidate in this nested comparison.",
}
(EXP / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps({"common_target_count": len(common_ids), "selected": selected_rows,
                  "outer_scores": outer_rows}, indent=2))
