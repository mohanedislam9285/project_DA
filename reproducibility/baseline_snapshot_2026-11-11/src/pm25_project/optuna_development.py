"""Nested, development-only CNN-LSTM TPE search for the five temporal folds."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_origin_features
from .metrics import regression_metrics
from .models.cnn_lstm import build_cnn_lstm
from .preprocessing import SequencePreprocessor
from .sequences import make_sequences
from .tuning import chronological_inner_indices


def run_optuna_temporal(df: pd.DataFrame, cfg: dict, trial_budget: int = 50) -> dict:
    """Run seeded 50-trial TPE studies inside each outer development fold.

    Outer validation labels are never passed to the Optuna objective. The only
    data used to select a trial are the chronological inner split of the outer
    training partition. Results are saved beside, and do not replace, the
    earlier finite-grid study.
    """
    import optuna
    import tensorflow as tf
    import joblib

    root = Path(cfg["_config_dir"])
    out = root / cfg["paths"]["results_root"] / cfg["project"]["group_id"]
    fold_path = out / "folds_temporal.json"
    if not fold_path.is_file():
        raise FileNotFoundError(f"Missing development folds: {fold_path}")
    fold_def = json.loads(fold_path.read_text(encoding="utf-8"))
    if fold_def.get("scheme") != "temporal_panel_expanding":
        raise ValueError("Optuna search requires the saved expanding temporal folds")

    future_path = out / "future_test.json"
    reservation = json.loads(future_path.read_text(encoding="utf-8"))
    development_indices = np.asarray(reservation["development_indices"], dtype=int)
    dev = df.iloc[development_indices].copy()
    cutoff = pd.to_datetime(reservation["development_end"], utc=True)
    if (pd.to_datetime(dev.timestamp, utc=True) > cutoff).any():
        raise ValueError("Saved development indices cross the recorded cutoff")

    features = build_origin_features(dev, cfg)
    columns = list(cfg["features"].get("predictors", [])) + list(cfg["features"].get("calendar", []))
    if cfg["features"].get("include_target_lags", True):
        columns.append("pm25_lag1")
    if cfg["features"].get("include_past_rolling", True):
        columns += [f"pm25_roll{w}_mean" for w in cfg["features"].get("rolling_windows_days", [3, 7])]
    columns = list(dict.fromkeys(columns))
    lookback = int(cfg["project"].get("lookback_candidates_days", [7])[0])
    search_space = cfg.get("_search_spaces", {}).get("cnn_lstm_tpe", {})
    if not search_space:
        raise ValueError("Declare cnn_lstm_tpe in configs/search_spaces.yaml")
    X, y, keys = make_sequences(features, columns, lookback=lookback, horizon=1)
    target_rows = keys.target_row.to_numpy(dtype=int)
    dates = pd.to_datetime(keys.target_timestamp, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    trials_dir = out / "optuna"
    model_dir = out / "models"
    history_dir = out / "histories"
    prediction_dir = out / "predictions"
    for path in (trials_dir, model_dir, history_dir, prediction_dir):
        path.mkdir(parents=True, exist_ok=True)
    db_path = trials_dir / "cnn_lstm_temporal_tpe.sqlite3"
    storage = f"sqlite:///{db_path.as_posix()}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    trial_rows, fold_rows, prediction_rows, hp_rows = [], [], [], []
    for fold in fold_def["folds"]:
        fold_number = int(fold["fold"])
        train = np.flatnonzero(np.isin(target_rows, np.asarray(fold["train_indices"], dtype=int)))
        valid = np.flatnonzero(np.isin(target_rows, np.asarray(fold["validation_indices"], dtype=int)))
        if not len(train) or not len(valid):
            raise ValueError(f"No valid sequences for temporal fold {fold_number}")
        inner_train, inner_valid = chronological_inner_indices(dates[train])
        prep_inner = SequencePreprocessor().fit(X[train[inner_train]])
        x_inner_train = prep_inner.transform(X[train[inner_train]])
        x_inner_valid = prep_inner.transform(X[train[inner_valid]])
        y_inner_train = y[train[inner_train]]
        y_inner_valid = y[train[inner_valid]]
        study_name = f"cnn_lstm_group16_temporal_fold_{fold_number}"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=int(cfg.get("_search_spaces", {}).get("budget", {}).get("seed", 42)) + fold_number, n_startup_trials=int(search_space["n_startup_trials"])),
        )

        def objective(trial):
            params = {
                "filters": trial.suggest_categorical("filters", search_space["filters"]),
                "kernel_size": trial.suggest_categorical("kernel_size", search_space["kernel_size"]),
                "lstm_units": trial.suggest_categorical("lstm_units", search_space["lstm_units"]),
                "dropout": trial.suggest_float("dropout", float(search_space["dropout_min"]), float(search_space["dropout_max"])),
                "learning_rate": trial.suggest_float("learning_rate", float(search_space["learning_rate_min"]), float(search_space["learning_rate_max"]), log=bool(search_space["log_learning_rate"])),
                "batch_size": int(search_space["batch_size"]),
                "dense_units": int(cfg["model"]["cnn_lstm"].get("dense_units", 32)),
                "loss": "mse",
                "seed": 42 + fold_number,
            }
            if params["kernel_size"] > lookback:
                raise ValueError("Kernel size exceeds lookback")
            tf.keras.backend.clear_session()
            model = build_cnn_lstm(lookback, x_inner_train.shape[-1], params)
            callbacks = [
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=float(search_space["reduce_lr_factor"]), patience=int(search_space["reduce_lr_patience"]), min_lr=float(search_space["min_learning_rate"]), verbose=0
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True, verbose=0
                ),
            ]
            start = time.perf_counter()
            hist = model.fit(
                x_inner_train, y_inner_train,
                validation_data=(x_inner_valid, y_inner_valid),
                epochs=int(search_space["max_epochs"]), batch_size=int(params["batch_size"]), callbacks=callbacks, verbose=0,
            )
            pred = model.predict(x_inner_valid, verbose=0).reshape(-1)
            score = regression_metrics(y_inner_valid, pred)["rmse"]
            trial.set_user_attr("best_epoch", int(np.argmin(hist.history["val_loss"]) + 1))
            trial.set_user_attr("fit_time_sec", float(time.perf_counter() - start))
            return float(score)

        remaining = max(0, int(trial_budget) - len(study.trials))
        if remaining:
            study.optimize(objective, n_trials=remaining, n_jobs=1, show_progress_bar=False)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and np.isfinite(t.value)]
        if len(completed) < trial_budget:
            raise RuntimeError(f"Fold {fold_number}: only {len(completed)} completed TPE trials; required {trial_budget}")
        best = min(completed, key=lambda t: (float(t.value), t.number))
        best_epoch = int(best.user_attrs["best_epoch"])
        params = {
        **best.params,
        "batch_size": int(search_space["batch_size"]),
            "dense_units": int(cfg["model"]["cnn_lstm"].get("dense_units", 32)),
            "loss": "mse",
            "seed": 42 + fold_number,
        }

        prep = SequencePreprocessor().fit(X[train])
        x_train, x_valid = prep.transform(X[train]), prep.transform(X[valid])
        tf.keras.backend.clear_session()
        model = build_cnn_lstm(lookback, x_train.shape[-1], params)
        start = time.perf_counter()
        outer_history = model.fit(
            x_train, y[train], epochs=best_epoch, batch_size=int(params.get("batch_size", 64)), verbose=0
        )
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        prediction = model.predict(x_valid, verbose=0).reshape(-1)
        inference_seconds = time.perf_counter() - start
        metrics = regression_metrics(y[valid], prediction)

        model_path = model_dir / f"cnn_lstm_optuna_temporal_fold_{fold_number}.keras"
        model.save(model_path)
        joblib.dump(prep, model_dir / f"cnn_lstm_optuna_temporal_fold_{fold_number}_preprocessor.joblib")
        config = {
            **params, "lookback_days": lookback, "best_epoch": best_epoch,
            "inner_validation_rmse": float(best.value), "trial_number": int(best.number),
            "n_trials": len(completed), "feature_columns": columns,
            "development_cutoff_exclusive": "2024-12-08T00:00:00Z",
            "selection_data": "outer training partition, chronological inner validation only",
        }
        (model_dir / f"cnn_lstm_optuna_temporal_fold_{fold_number}_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        pd.DataFrame(outer_history.history).to_csv(
            history_dir / f"cnn_lstm_optuna_temporal_fold_{fold_number}.csv", index=False
        )
        fold_rows.append({
            "model_id": "CNN_LSTM_OPTUNA", "model_group": "Group_16", "model_family": "CNN-LSTM",
            "dataset_split": "validation", "cv_scheme": "temporal_panel_expanding",
            "cv_purpose": "development_selection", "cv_fold": fold_number,
            "metric_rmse": metrics["rmse"], "metric_mae": metrics["mae"], "metric_r2": metrics["r2"],
            "n_samples": metrics["n_samples"], "training_sequences": len(train),
            "validation_start": fold.get("validation_start"), "validation_end": fold.get("validation_end"),
            "best_trial": int(best.number), "n_trials": len(completed), "best_epoch": best_epoch,
            "inner_rmse": float(best.value), "fit_time_sec": fit_seconds,
            "search_time_sec": float(sum(t.user_attrs.get("fit_time_sec", 0.0) for t in completed)),
            "inference_time_sec": inference_seconds, "n_parameters": int(model.count_params()),
            "model_size_mb": model_path.stat().st_size / 1024**2,
            "seed": 42 + fold_number, "configuration": json.dumps(config, sort_keys=True),
        })
        for j, idx in enumerate(valid):
            key = keys.iloc[idx]
            prediction_rows.append({
                "sample_id": int(key.target_row), "station_id": key.station_id,
                "origin_timestamp": key.origin_timestamp, "target_timestamp": key.target_timestamp,
                "y_true": float(y[idx]), "y_pred": float(prediction[j]),
                "residual": float(y[idx] - prediction[j]), "dataset_split": "validation",
                "cv_scheme": "temporal_panel_expanding", "cv_fold": fold_number,
                "model_id": "CNN_LSTM_OPTUNA", "config_id": f"fold{fold_number}_trial{best.number}",
                "seed": 42 + fold_number,
            })
        for trial in study.trials:
            trial_rows.append({
                "cv_fold": fold_number, "trial_number": int(trial.number),
                "state": trial.state.name, "inner_validation_rmse": trial.value,
                "best_epoch": trial.user_attrs.get("best_epoch"),
                "fit_time_sec": trial.user_attrs.get("fit_time_sec"),
                **trial.params,
            })
        for name, value in params.items():
            hp_rows.append({"cv_fold": fold_number, "best_trial": int(best.number), "hyperparameter": name, "value": value})
        tf.keras.backend.clear_session()

    trials = pd.DataFrame(trial_rows)
    folds = pd.DataFrame(fold_rows)
    preds = pd.DataFrame(prediction_rows)
    trials.to_csv(out / "optuna_trials.csv", index=False, encoding="utf-8")
    top3 = trials[trials.state.eq("COMPLETE")].sort_values(["cv_fold", "inner_validation_rmse", "trial_number"]).groupby("cv_fold", sort=True).head(3).copy()
    top3.insert(1, "config_rank", top3.groupby("cv_fold").inner_validation_rmse.rank(method="first").astype(int))
    parameter_columns = [c for c in ("filters", "kernel_size", "lstm_units", "dropout", "learning_rate", "batch_size") if c in top3]
    top3["configuration"] = top3[parameter_columns].to_dict(orient="records")
    top3["configuration"] = top3.configuration.map(lambda x: json.dumps(x, sort_keys=True))
    top3[["cv_fold", "config_rank", "trial_number", "inner_validation_rmse", "best_epoch", "configuration"]].to_csv(out / "optuna_top3_configs.csv", index=False, encoding="utf-8")
    folds.to_csv(out / "temporal_cv_optuna_results.csv", index=False, encoding="utf-8")
    preds.to_csv(prediction_dir / "cnn_lstm_optuna_temporal_oof.csv", index=False, encoding="utf-8")
    pd.DataFrame(hp_rows).to_csv(out / "optuna_best_hyperparameters.csv", index=False, encoding="utf-8")
    summary = folds[["metric_rmse", "metric_mae", "metric_r2"]].agg(["mean", "std"]).to_dict()
    (out / "optuna_study_metadata.json").write_text(json.dumps({
        "study": "Group 16 CNN-LSTM nested TPE development CV",
        "sampler": "Optuna TPESampler", "optuna_version": optuna.__version__,
        "seed": 42, "trial_budget_per_outer_fold": int(trial_budget),
        "n_outer_folds": int(len(folds)), "completed_trials": int(len(trials)),
        "development_cutoff_exclusive": "2024-12-08T00:00:00Z",
        "objective": "inner chronological validation RMSE in original ug/m3 units",
        "scheduler": "ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)",
        "early_stopping": "validation MSE, patience=10, restore_best_weights=True",
        "search_space": search_space,
        "summary": summary,
    }, indent=2), encoding="utf-8")
    return {"folds": int(len(folds)), "completed_trials": int(len(trials)), "summary": summary}
