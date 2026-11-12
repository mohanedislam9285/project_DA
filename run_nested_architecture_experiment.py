"""Development-only nested CNN-LSTM architecture ablation.

Within each saved outer temporal fold, inner chronological validation compares
four predeclared CNN-LSTM variants: pooling on/off crossed with direct-target or
persistence-residual targets. Each variant inherits that fold's already saved
inner-TPE hyperparameters. The chosen variant is selected only on inner data;
all four variants are then refit on outer training data and scored on the same
outer validation sequences. The future holdout is never loaded for scoring.
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
EXP = OUT / "architecture_experiment"
for sub in ("models", "histories"):
    (EXP / sub).mkdir(parents=True, exist_ok=True)
cfg, _ = load_config(ROOT / "configs" / "project.yaml")
df = pd.read_csv(ROOT / "data" / "processed" / "cleaned_daily.csv", parse_dates=["timestamp"])
df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
reservation = json.loads((OUT / "future_test.json").read_text(encoding="utf-8"))
dev = df.iloc[np.asarray(reservation["development_indices"], dtype=int)].copy()
cutoff = pd.to_datetime(reservation["development_end"], utc=True)
if dev.timestamp.max() > cutoff:
    raise ValueError("Development selection data cross the saved cutoff")

features = build_origin_features(dev, cfg)
columns = list(cfg["features"].get("predictors", [])) + list(cfg["features"].get("calendar", []))
if cfg["features"].get("include_target_lags", True):
    columns.append("pm25_lag1")
if cfg["features"].get("include_past_rolling", True):
    columns.extend(f"pm25_roll{w}_mean" for w in cfg["features"].get("rolling_windows_days", [3, 7]))
columns = list(dict.fromkeys(columns))
lookback = int(cfg["project"].get("lookback_candidates_days", [7])[0])
X, y, keys = make_sequences(features, columns, lookback=lookback, horizon=1)
target_rows = keys.target_row.to_numpy(dtype=int)
dates = pd.to_datetime(keys.target_timestamp, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
base_idx = columns.index("pm25_lag1")
base = X[:, -1, base_idx].astype(np.float64)

fold_def = json.loads((OUT / "folds_temporal.json").read_text(encoding="utf-8"))
variants = [
    {"variant": "direct_pool2", "pool_size": 2, "target_mode": "direct"},
    {"variant": "direct_no_pool", "pool_size": None, "target_mode": "direct"},
    {"variant": "residual_pool2", "pool_size": 2, "target_mode": "residual"},
    {"variant": "residual_no_pool", "pool_size": None, "target_mode": "residual"},
]
inner_rows: list[dict[str, object]] = []
outer_rows: list[dict[str, object]] = []
oof_rows: list[dict[str, object]] = []
chosen_rows: list[dict[str, object]] = []

for fold in fold_def["folds"]:
    fold_id = int(fold["fold"])
    train = np.flatnonzero(np.isin(target_rows, np.asarray(fold["train_indices"], dtype=int)))
    valid = np.flatnonzero(np.isin(target_rows, np.asarray(fold["validation_indices"], dtype=int)))
    inner_train, inner_valid = chronological_inner_indices(dates[train])
    saved_cfg_path = OUT / "models" / f"cnn_lstm_optuna_temporal_fold_{fold_id}_config.json"
    tuned = json.loads(saved_cfg_path.read_text(encoding="utf-8"))
    hp = {name: tuned[name] for name in ("filters", "kernel_size", "lstm_units", "dense_units", "dropout", "learning_rate", "loss")}
    seed = int(tuned["seed"])
    common = {**hp, "seed": seed, "batch_size": int(cfg["model"]["cnn_lstm"].get("batch_size", 64))}

    inner_prep = SequencePreprocessor().fit(X[train[inner_train]])
    xi_train = inner_prep.transform(X[train[inner_train]])
    xi_valid = inner_prep.transform(X[train[inner_valid]])
    yi_train, yi_valid = y[train[inner_train]], y[train[inner_valid]]
    base_i_train, base_i_valid = base[train[inner_train]], base[train[inner_valid]]
    inner_scores: dict[str, float] = {}
    variant_epochs: dict[str, int] = {}

    for spec in variants:
        mode = spec["target_mode"]
        fit_y = yi_train - base_i_train if mode == "residual" else yi_train
        val_y = yi_valid - base_i_valid if mode == "residual" else yi_valid
        tf.keras.backend.clear_session()
        model = build_cnn_lstm(lookback, xi_train.shape[-1], {**common, "pool_size": spec["pool_size"]})
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
        ]
        started = time.perf_counter()
        hist = model.fit(xi_train, fit_y, validation_data=(xi_valid, val_y), epochs=100,
                         batch_size=common["batch_size"], callbacks=callbacks, verbose=0)
        elapsed = time.perf_counter() - started
        best_epoch = int(np.argmin(hist.history["val_loss"]) + 1)
        inner_pred = model.predict(xi_valid, verbose=0).reshape(-1)
        if mode == "residual":
            inner_pred = base_i_valid + inner_pred
        inner_rmse = float(mean_squared_error(yi_valid, inner_pred) ** 0.5)
        inner_mae = float(mean_absolute_error(yi_valid, inner_pred))
        inner_scores[spec["variant"]] = inner_rmse
        variant_epochs[spec["variant"]] = best_epoch
        inner_rows.append({
            "cv_fold": fold_id, **spec, **common, "inner_validation_sequences": len(inner_valid),
            "best_epoch": best_epoch, "inner_validation_rmse": inner_rmse,
            "inner_validation_mae": inner_mae, "min_train_loss": float(np.min(hist.history["loss"])),
            "min_inner_val_loss": float(np.min(hist.history["val_loss"])),
            "last_train_loss": float(hist.history["loss"][-1]), "last_inner_val_loss": float(hist.history["val_loss"][-1]),
            "fit_seconds": elapsed,
        })
        pd.DataFrame(hist.history).to_csv(EXP / "histories" / f"fold{fold_id}_{spec['variant']}_inner.csv", index=False)

    selected_name = min(inner_scores, key=lambda name: (inner_scores[name], name))
    chosen = next(v for v in variants if v["variant"] == selected_name)
    chosen_rows.append({"cv_fold": fold_id, "selected_variant": selected_name,
                        "inner_validation_rmse": inner_scores[selected_name], "best_epoch": variant_epochs[selected_name],
                        "selection_source": "inner chronological validation only"})

    outer_prep = SequencePreprocessor().fit(X[train])
    x_train = outer_prep.transform(X[train])
    x_valid = outer_prep.transform(X[valid])
    y_train, y_valid = y[train], y[valid]
    base_train, base_valid = base[train], base[valid]
    for spec in variants:
        mode = spec["target_mode"]
        fit_y = y_train - base_train if mode == "residual" else y_train
        tf.keras.backend.clear_session()
        params = {**common, "pool_size": spec["pool_size"]}
        model = build_cnn_lstm(lookback, x_train.shape[-1], params)
        epochs = variant_epochs[spec["variant"]]
        started = time.perf_counter()
        hist = model.fit(x_train, fit_y, epochs=epochs, batch_size=common["batch_size"], verbose=0)
        fit_seconds = time.perf_counter() - started
        pred_train = model.predict(x_train, verbose=0).reshape(-1)
        pred_valid = model.predict(x_valid, verbose=0).reshape(-1)
        if mode == "residual":
            pred_train += base_train
            pred_valid += base_valid
        train_rmse = float(mean_squared_error(y_train, pred_train) ** 0.5)
        rmse = float(mean_squared_error(y_valid, pred_valid) ** 0.5)
        mae = float(mean_absolute_error(y_valid, pred_valid))
        r2 = float(r2_score(y_valid, pred_valid))
        model_path = EXP / "models" / f"fold{fold_id}_{spec['variant']}.keras"
        model.save(model_path)
        joblib.dump(outer_prep, EXP / "models" / f"fold{fold_id}_{spec['variant']}_preprocessor.joblib")
        model_config = {
            **params, "target_mode": mode, "lookback_days": lookback, "feature_columns": columns,
            "best_epoch_from_inner_validation": epochs, "inner_validation_rmse": inner_scores[spec["variant"]],
            "development_cutoff_inclusive": cutoff.isoformat(), "outer_fold": fold_id,
            "outer_validation_used_for_selection": False,
            "persistence_residual": "prediction = pm25_at_origin + model_delta" if mode == "residual" else None,
        }
        (EXP / "models" / f"fold{fold_id}_{spec['variant']}_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
        outer_rows.append({
            "cv_fold": fold_id, **spec, **common, "selected_by_inner_validation": spec["variant"] == selected_name,
            "best_epoch": epochs, "outer_train_sequences": len(train), "outer_validation_sequences": len(valid),
            "train_rmse_in_sample": train_rmse, "outer_validation_rmse": rmse,
            "outer_validation_mae": mae, "outer_validation_r2": r2, "fit_seconds": fit_seconds,
            "inner_validation_rmse": inner_scores[spec["variant"]],
        })
        for j, seq_idx in enumerate(valid):
            key = keys.iloc[seq_idx]
            oof_rows.append({
                "sample_id": int(key.target_row), "station_id": str(key.station_id),
                "origin_timestamp": pd.Timestamp(key.origin_timestamp).isoformat() + "+00:00",
                "target_timestamp": pd.Timestamp(key.target_timestamp).isoformat() + "+00:00",
                "y_true": float(y[seq_idx]), "persistence": float(base[seq_idx]),
                "y_pred": float(pred_valid[j]), "cv_fold": fold_id,
                "variant": spec["variant"], "selected_by_inner_validation": spec["variant"] == selected_name,
            })
        pd.DataFrame(hist.history).to_csv(EXP / "histories" / f"fold{fold_id}_{spec['variant']}_outer_fit.csv", index=False)
        tf.keras.backend.clear_session()

pd.DataFrame(inner_rows).to_csv(EXP / "inner_architecture_scores.csv", index=False)
pd.DataFrame(outer_rows).to_csv(EXP / "outer_architecture_scores.csv", index=False)
pd.DataFrame(oof_rows).to_csv(EXP / "architecture_oof_predictions.csv", index=False)
pd.DataFrame(chosen_rows).to_csv(EXP / "selected_variant_by_fold.csv", index=False)
metadata = {
    "experiment": "nested architecture ablation, development temporal CV only",
    "architectures": variants, "outer_folds": len(fold_def["folds"]),
    "selection": "lowest RMSE on chronological inner validation; fold-specific Optuna hyperparameters held fixed",
    "lookback_days": lookback, "forecast_horizon_days": 1,
    "seeds": sorted({int(json.loads((OUT / "models" / f"cnn_lstm_optuna_temporal_fold_{k}_config.json").read_text())["seed"]) for k in range(1, 6)}),
    "future_test_read_for_scoring_or_selection": False,
    "preprocessing": "median imputer and scaler fit on inner training for selection, then outer training for refit",
    "result_files": ["inner_architecture_scores.csv", "outer_architecture_scores.csv", "architecture_oof_predictions.csv", "selected_variant_by_fold.csv"],
}
(EXP / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps({"completed_folds": len(fold_def["folds"]), "inner_scores": len(inner_rows), "outer_scores": len(outer_rows), "selected": chosen_rows}, indent=2))
