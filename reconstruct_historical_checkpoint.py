"""Attempt a development-only historical checkpoint reconstruction.

Configuration and epoch are selected exclusively from development rows using
the same candidate, inner chronological split, preprocessing, and refit steps
recorded by final_evaluation.py. The saved future predictions are used only
after training, for a blind prediction-array reproduction check. Their labels
are never loaded or scored here. This script does not call the one-time
evaluator or modify any frozen future artifact.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pm25_project.config import load_config
from pm25_project.cnn_lstm_benchmark import cnn_lstm_candidates
from pm25_project.features import build_origin_features
from pm25_project.metrics import regression_metrics
from pm25_project.models.cnn_lstm import build_cnn_lstm
from pm25_project.preprocessing import SequencePreprocessor
from pm25_project.sequences import make_sequences
from pm25_project.tuning import chronological_inner_indices


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "16"
DEST = OUT / "historical_reconstruction_attempt"
DEST.mkdir(parents=True, exist_ok=True)
cfg, _ = load_config(ROOT / "configs" / "project.yaml")
all_rows = pd.read_csv(ROOT / "data" / "processed" / "cleaned_daily.csv", parse_dates=["timestamp"])
all_rows["timestamp"] = pd.to_datetime(all_rows.timestamp, utc=True)
reservation = json.loads((OUT / "future_test.json").read_text(encoding="utf-8"))
dev_rows = set(map(int, reservation["development_indices"]))
future_rows = set(map(int, reservation["future_test_indices"]))
cutoff = pd.to_datetime(reservation["development_end"], utc=True)
dev_frame = all_rows.iloc[sorted(dev_rows)]
if dev_frame.timestamp.max() > cutoff:
    raise ValueError("Historical reconstruction training frame crosses the development cutoff")

features = build_origin_features(all_rows, cfg)
feature_columns = list(cfg["features"].get("predictors", [])) + list(cfg["features"].get("calendar", []))
if cfg["features"].get("include_target_lags", True):
    feature_columns.append("pm25_lag1")
if cfg["features"].get("include_past_rolling", True):
    feature_columns.extend(f"pm25_roll{w}_mean" for w in cfg["features"].get("rolling_windows_days", [3, 7]))
feature_columns = list(dict.fromkeys(feature_columns))
candidates = cnn_lstm_candidates(cfg)
cache = []
for candidate in candidates:
    window = int(candidate["lookback_days"])
    x, y, keys = make_sequences(features, feature_columns, lookback=window, horizon=1)
    dev = np.flatnonzero(keys.target_row.isin(dev_rows).to_numpy())
    future = np.flatnonzero(keys.target_row.isin(future_rows).to_numpy())
    cache.append((candidate, x, y, keys, dev, future))
common_development_ids = set.intersection(*(
    set(keys.iloc[dev].target_row.astype(int)) for _, _, _, keys, dev, _ in cache
))
if not common_development_ids:
    raise ValueError("Candidate lookbacks have no common development target rows")

candidate_scores = []
for candidate, x, y, keys, dev, _ in cache:
    dev = dev[keys.iloc[dev].target_row.astype(int).isin(common_development_ids).to_numpy()]
    inner_train, inner_valid = chronological_inner_indices(keys.iloc[dev].target_timestamp.to_numpy())
    params = {**candidate, "seed": int(cfg["project"]["seed"])}
    prep = SequencePreprocessor().fit(x[dev[inner_train]])
    xi, xv = prep.transform(x[dev[inner_train]]), prep.transform(x[dev[inner_valid]])
    tf.keras.backend.clear_session()
    model = build_cnn_lstm(int(candidate["lookback_days"]), xi.shape[-1], params)
    history = model.fit(
        xi, y[dev[inner_train]], validation_data=(xv, y[dev[inner_valid]]),
        epochs=int(params.get("epochs", 100)), batch_size=int(params.get("batch_size", 64)),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(params.get("patience", 10)), restore_best_weights=True)],
        verbose=0,
    )
    prediction = model.predict(xv, verbose=0).reshape(-1)
    score = regression_metrics(y[dev[inner_valid]], prediction)
    candidate_scores.append({
        "candidate": candidate, "inner_rmse": float(score["rmse"]),
        "inner_mae": float(score["mae"]),
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "inner_n": int(score["n_samples"]),
    })
    pd.DataFrame(history.history).to_csv(DEST / f"candidate_{len(candidate_scores)}_inner_history.csv", index=False)

selected = min(candidate_scores, key=lambda row: row["inner_rmse"])
chosen_candidate = selected["candidate"]
candidate, x, y, keys, dev, future = next(row for row in cache if row[0] == chosen_candidate)
dev = dev[keys.iloc[dev].target_row.astype(int).isin(common_development_ids).to_numpy()]
prep = SequencePreprocessor().fit(x[dev])
xt = prep.transform(x[dev])
tf.keras.backend.clear_session()
model = build_cnn_lstm(int(candidate["lookback_days"]), xt.shape[-1], {**candidate, "seed": int(cfg["project"]["seed"])})
started = time.perf_counter()
history = model.fit(xt, y[dev], epochs=selected["best_epoch"], batch_size=int(candidate.get("batch_size", 64)), verbose=0)
fit_seconds = time.perf_counter() - started
model_path = DEST / "cnn_lstm_historical_reconstruction.keras"
preprocessor_path = DEST / "cnn_lstm_historical_reconstruction_preprocessor.joblib"
config_path = DEST / "cnn_lstm_historical_reconstruction_config.json"
model.save(model_path)
joblib.dump(prep, preprocessor_path)
config = {
    **candidate, "seed": int(cfg["project"]["seed"]), "best_epoch": int(selected["best_epoch"]),
    "lookback_days": int(candidate["lookback_days"]), "feature_columns": feature_columns,
    "inner_validation_rmse": selected["inner_rmse"], "inner_validation_mae": selected["inner_mae"],
    "selection_method": "minimum inner chronological-validation RMSE on development sequences only",
    "training_target_rows": int(len(dev)), "development_cutoff_inclusive": cutoff.isoformat(),
    "future_test_labels_loaded_or_scored": False,
}
config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
pd.DataFrame(history.history).to_csv(DEST / "historical_reconstruction_training_history.csv", index=False)

# Reload and compare predictions only. Deliberately do not read saved y_true.
saved = pd.read_csv(OUT / "predictions" / "future_test.csv", usecols=[
    "sample_id", "station_id", "origin_timestamp", "target_timestamp", "y_pred", "model_id"
])
saved = saved[saved.model_id.eq("CNN_LSTM")].drop(columns="model_id")
saved["sample_id"] = pd.to_numeric(saved.sample_id, errors="raise").astype(int)
loaded_model = tf.keras.models.load_model(model_path)
loaded_prep = joblib.load(preprocessor_path)
future_ids = keys.iloc[future].target_row.astype(int).to_numpy()
future_pred = loaded_model.predict(loaded_prep.transform(x[future]), verbose=0).reshape(-1)
computed = pd.DataFrame({"sample_id": future_ids, "reconstructed_prediction": future_pred})
joined = saved.merge(computed, on="sample_id", how="outer", validate="one_to_one", indicator=True)
joined["prediction_abs_difference"] = (joined.y_pred - joined.reconstructed_prediction).abs()
joined.to_csv(DEST / "future_prediction_reproduction_check.csv", index=False, encoding="utf-8")
matched = joined[joined._merge.eq("both")]
reproduction = {
    "status": "exact_prediction_reproduction" if len(joined) == len(matched) and len(matched) and matched.prediction_abs_difference.max() <= 1e-5 else "checkpoint_not_reconstructed_from_available_metadata",
    "method": "One development-only nested final-fit reconstruction, followed by prediction-array comparison only; no holdout labels read.",
    "selected_candidate": chosen_candidate, "selected_epochs": int(selected["best_epoch"]),
    "matches_recorded_epoch_count": int(selected["best_epoch"]) == 22,
    "reconstruction_fit_seconds": fit_seconds,
    "saved_prediction_count": int(len(saved)), "reconstructed_prediction_count": int(len(computed)),
    "matched_prediction_count": int(len(matched)), "saved_only_count": int(joined._merge.eq("left_only").sum()),
    "reconstruction_only_count": int(joined._merge.eq("right_only").sum()),
    "max_prediction_abs_difference": float(matched.prediction_abs_difference.max()) if len(matched) else None,
    "mean_prediction_abs_difference": float(matched.prediction_abs_difference.mean()) if len(matched) else None,
    "future_test_metrics_calculated": False,
    "future_artifacts_modified": False,
    "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    "preprocessor_sha256": hashlib.sha256(preprocessor_path.read_bytes()).hexdigest(),
    "environment": {"python": platform.python_version(), "tensorflow": tf.__version__},
    "candidate_inner_scores": candidate_scores,
    "note": "A prediction match can corroborate the saved evaluation model. A mismatch does not identify which missing historical configuration or software state differed; no test-based tuning follows.",
}
(DEST / "reconstruction_verification.json").write_text(json.dumps(reproduction, indent=2), encoding="utf-8")
print(json.dumps(reproduction, indent=2))
