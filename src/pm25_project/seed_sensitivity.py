"""Repeat the selected CNN-LSTM configuration across fixed training seeds."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .cnn_lstm_benchmark import _prepare, cnn_lstm_candidates
from .features import build_origin_features
from .metrics import regression_metrics
from .models.cnn_lstm import build_cnn_lstm
from .preprocessing import SequencePreprocessor
from .tuning import chronological_inner_indices


def _seed_values(cfg: dict) -> list[int]:
    values = cfg.get("_search_spaces", {}).get("cnn_lstm_seed_sensitivity")
    if not isinstance(values, list) or len(values) < 3:
        raise ValueError("Declare at least three cnn_lstm_seed_sensitivity values in search_spaces.yaml")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("CNN-LSTM sensitivity seeds must be integers")
    if len(set(values)) != len(values):
        raise ValueError("CNN-LSTM sensitivity seeds must be unique")
    return values


def summarize_seed_results(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"seed", "cv_fold", "metric", "value", "n_samples"}
    if not required.issubset(rows.columns):
        raise ValueError(f"Seed results missing columns: {sorted(required - set(rows.columns))}")
    sample_counts = rows[rows.metric.eq("rmse")].pivot(
        index="cv_fold", columns="seed", values="n_samples"
    )
    if sample_counts.empty or sample_counts.isna().any().any():
        raise ValueError("Every seed must produce RMSE on every temporal fold")
    if not sample_counts.nunique(axis=1).eq(1).all():
        raise ValueError("Seed comparisons require identical validation sample counts within each fold")
    return (
        rows.groupby(["seed", "metric"], as_index=False)
        .agg(
            mean_fold_value=("value", "mean"),
            std_across_folds=("value", "std"),
            n_folds=("cv_fold", "nunique"),
        )
        .sort_values(["metric", "seed"], ignore_index=True)
    )


def run_cnn_lstm_seed_sensitivity(df: pd.DataFrame, cfg: dict) -> dict:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "CNN-LSTM seed sensitivity requires TensorFlow; install the [deep] extra."
        ) from exc

    seeds = _seed_values(cfg)
    root = Path(cfg["_config_dir"])
    out = root / cfg["paths"]["results_root"] / cfg["project"]["group_id"]
    fold_path = out / "folds_temporal.json"
    if not fold_path.is_file():
        raise FileNotFoundError("Run the folds stage before CNN-LSTM seed sensitivity")
    folds = json.loads(fold_path.read_text(encoding="utf-8"))["folds"]
    feature_table = build_origin_features(df, cfg)
    feature_columns = list(cfg["features"].get("predictors", []))
    feature_columns.extend(cfg["features"].get("calendar", []))
    if cfg["features"].get("include_target_lags", True):
        feature_columns.append("pm25_lag1")
    if cfg["features"].get("include_past_rolling", True):
        feature_columns.extend(
            f"pm25_roll{window}_mean"
            for window in cfg["features"].get("rolling_windows_days", [3, 7])
        )
    feature_columns = list(dict.fromkeys(feature_columns))
    candidates = cnn_lstm_candidates(cfg)
    selection_seed = int(cfg["project"]["seed"])
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    config_rows: list[dict] = []

    for fold in folds:
        scores: list[dict] = []
        cached: dict[int, tuple] = {}
        target_sets: list[set[int]] = []
        for candidate in candidates:
            data = _prepare(
                feature_table,
                feature_columns,
                candidate["lookback_days"],
                set(fold["train_indices"]),
                set(fold["validation_indices"]),
            )
            cached[candidate["lookback_days"]] = data
            target_sets.append(set(data[2].target_row.astype(int)))
        common_targets = set.intersection(*target_sets) if target_sets else set()
        if not common_targets:
            raise ValueError(
                "CNN-LSTM candidates share no fold samples; adjust the predeclared search "
                "or data-coverage gate before running seed sensitivity."
            )

        for candidate in candidates:
            x, y, keys, train_indices, _ = cached[candidate["lookback_days"]]
            train_indices = train_indices[
                keys.iloc[train_indices].target_row.astype(int).isin(common_targets).to_numpy()
            ]
            inner_train, inner_valid = chronological_inner_indices(
                keys.iloc[train_indices].target_timestamp.to_numpy()
            )
            prep = SequencePreprocessor().fit(x[train_indices[inner_train]])
            x_train = prep.transform(x[train_indices[inner_train]])
            x_valid = prep.transform(x[train_indices[inner_valid]])
            tf.keras.backend.clear_session()
            tune_config = {
                key: value for key, value in candidate.items() if key != "lookback_days"
            }
            tune_config["seed"] = selection_seed
            model = build_cnn_lstm(
                candidate["lookback_days"], x_train.shape[-1], tune_config
            )
            history = model.fit(
                x_train,
                y[train_indices[inner_train]],
                validation_data=(x_valid, y[train_indices[inner_valid]]),
                epochs=int(candidate.get("epochs", 100)),
                batch_size=int(candidate.get("batch_size", 64)),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=int(candidate.get("patience", 10)),
                        restore_best_weights=True,
                    )
                ],
                verbose=0,
            )
            validation_prediction = model.predict(x_valid, verbose=0).reshape(-1)
            score = regression_metrics(
                y[train_indices[inner_valid]], validation_prediction
            )["rmse"]
            scores.append(
                {
                    "candidate": candidate,
                    "rmse": score,
                    "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
                }
            )

        scores.sort(key=lambda result: result["rmse"])
        selected = scores[0]
        candidate = selected["candidate"]
        x, y, keys, train_indices, valid_indices = _prepare(
            feature_table,
            feature_columns,
            candidate["lookback_days"],
            set(fold["train_indices"]),
            set(fold["validation_indices"]),
        )
        train_indices = train_indices[
            keys.iloc[train_indices].target_row.astype(int).isin(common_targets).to_numpy()
        ]
        valid_indices = valid_indices[
            keys.iloc[valid_indices].target_row.astype(int).isin(common_targets).to_numpy()
        ]
        preprocessor = SequencePreprocessor().fit(x[train_indices])
        x_train = preprocessor.transform(x[train_indices])
        x_valid = preprocessor.transform(x[valid_indices])
        config_rows.append(
            {
                "cv_fold": int(fold["fold"]),
                "selection_seed": selection_seed,
                "selected_config": json.dumps(candidate, sort_keys=True),
                "inner_validation_rmse": selected["rmse"],
                "fixed_epochs": selected["best_epoch"],
                "n_validation_samples": len(valid_indices),
            }
        )

        for seed in seeds:
            fit_config = {
                key: value for key, value in candidate.items() if key != "lookback_days"
            }
            fit_config["seed"] = seed
            tf.keras.backend.clear_session()
            model = build_cnn_lstm(
                candidate["lookback_days"], x_train.shape[-1], fit_config
            )
            start = time.perf_counter()
            model.fit(
                x_train,
                y[train_indices],
                epochs=selected["best_epoch"],
                batch_size=int(candidate.get("batch_size", 64)),
                shuffle=True,
                verbose=0,
            )
            fit_seconds = time.perf_counter() - start
            prediction = model.predict(x_valid, verbose=0).reshape(-1)
            metrics = regression_metrics(y[valid_indices], prediction)
            for metric, value in metrics.items():
                if metric == "n_samples":
                    continue
                metric_rows.append(
                    {
                        "model_id": "CNN_LSTM",
                        "model_group": "Group_16",
                        "cv_scheme": "temporal_panel_expanding",
                        "cv_purpose": "seed_sensitivity",
                        "cv_fold": int(fold["fold"]),
                        "seed": seed,
                        "selection_seed": selection_seed,
                        "metric": metric,
                        "value": value,
                        "n_samples": metrics["n_samples"],
                        "train_size": len(train_indices),
                        "fixed_epochs": selected["best_epoch"],
                        "fit_time_sec": fit_seconds,
                        "hardware": platform.platform(),
                        "notes": "Fold-selected configuration and epoch count fixed across seeds; future test untouched.",
                    }
                )
            for index, sequence_index in enumerate(valid_indices):
                key = keys.iloc[sequence_index]
                prediction_rows.append(
                    {
                        "sample_id": int(key.target_row),
                        "station_id": key.station_id,
                        "target_timestamp": key.target_timestamp,
                        "y_true": y[sequence_index],
                        "y_pred": prediction[index],
                        "residual": y[sequence_index] - prediction[index],
                        "cv_fold": int(fold["fold"]),
                        "seed": seed,
                    }
                )

    results = pd.DataFrame(metric_rows)
    summary = summarize_seed_results(results)
    tables = out / "tables"
    predictions_dir = out / "predictions"
    tables.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(tables / "cnn_lstm_seed_sensitivity_folds.csv", index=False, encoding="utf-8")
    summary.to_csv(tables / "cnn_lstm_seed_sensitivity_summary.csv", index=False, encoding="utf-8")
    pd.DataFrame(config_rows).to_csv(
        tables / "cnn_lstm_seed_sensitivity_configs.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(prediction_rows).to_csv(
        predictions_dir / "cnn_lstm_seed_sensitivity_oof.csv",
        index=False,
        encoding="utf-8",
    )
    return {
        "model_id": "CNN_LSTM",
        "seeds": seeds,
        "folds": len(folds),
        "metric_rows": len(metric_rows),
        "predictions": len(prediction_rows),
        "summary": summary.to_dict(orient="records"),
    }
