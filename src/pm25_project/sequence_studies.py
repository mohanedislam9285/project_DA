"""Matched development-only sequence-model comparison studies."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .explainability import fold_ranking_stability, sequence_channel_permutation
from .comparisons import comparison_table
from .features import build_origin_features
from .metrics import paired_bootstrap_ci, regression_metrics
from .models.cnn_lstm import build_lstm
from .preprocessing import SequencePreprocessor
from .schemas import ABLATION, FEATURE_IMPORTANCE, RESULTS
from .sequences import make_sequences
from .summaries import add_fold_summaries
from .tuning import chronological_inner_indices
from .sensitivity import adjacent_values


def _sequence_feature_columns(cfg, feature_table):
    columns = list(cfg["features"].get("predictors", []))
    columns.extend(name for name in cfg["features"].get("calendar", []) if name in feature_table)
    if cfg["features"].get("include_target_lags", True):
        columns.append("pm25_lag1")
    if cfg["features"].get("include_past_rolling", True):
        columns.extend(f"pm25_roll{window}_mean" for window in cfg["features"].get("rolling_windows_days", [3, 7]))
    return list(dict.fromkeys(columns))


def _candidate_configs(cfg):
    cnn_config = cfg["model"]["cnn_lstm"]
    base = {key: cnn_config[key] for key in ("dense_units", "loss", "batch_size", "epochs", "patience") if key in cnn_config}
    declared = cfg.get("_search_spaces", {}).get("implemented_lstm_candidates")
    if not declared:
        raise ValueError("Declare standalone LSTM candidates in configs/search_spaces.yaml")
    return [{**base, **candidate} for candidate in declared]


def run_lstm_temporal_ablation(df, cfg):
    """Train a standalone LSTM on CNN-LSTM's exact seven-day fold sample cohort."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("The LSTM comparator requires TensorFlow; install the [deep] extra") from exc

    root = Path(cfg["_config_dir"])
    out = root / cfg["paths"]["results_root"] / cfg["project"]["group_id"]
    fold_path = out / "folds_temporal.json"
    cnn_path = out / "predictions" / "cnn_lstm_temporal_oof.csv"
    if not fold_path.exists() or not cnn_path.exists():
        raise FileNotFoundError("Run temporal folds and CNN-LSTM development benchmark before the LSTM ablation")
    folds = json.loads(fold_path.read_text(encoding="utf-8")).get("folds", [])
    fold_def = json.loads(fold_path.read_text(encoding="utf-8"))
    if fold_def.get("scheme") != "temporal_panel_expanding":
        raise ValueError("Sequence-family ablation requires expanding temporal development folds")

    feature_table = build_origin_features(df, cfg)
    feature_columns = _sequence_feature_columns(cfg, feature_table)
    x, y, keys = make_sequences(feature_table, feature_columns, lookback=7, horizon=1)
    target_rows = keys.target_row.to_numpy(dtype=int)
    dates = pd.to_datetime(keys.target_timestamp, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    candidates = _candidate_configs(cfg)
    result_rows, prediction_rows, importance_rows, top3_rows = [], [], [], []

    for fold in folds:
        train = np.flatnonzero(np.isin(target_rows, fold["train_indices"]))
        valid = np.flatnonzero(np.isin(target_rows, fold["validation_indices"]))
        if not len(train) or not len(valid):
            raise ValueError(f"No complete LSTM windows in temporal fold {fold.get('fold')}")
        candidate_scores = []
        for candidate in candidates:
            inner_train, inner_valid = chronological_inner_indices(dates[train])
            prep = SequencePreprocessor().fit(x[train[inner_train]])
            x_inner_train = prep.transform(x[train[inner_train]])
            x_inner_valid = prep.transform(x[train[inner_valid]])
            tf.keras.backend.clear_session()
            candidate_config = {key: value for key, value in candidate.items() if key != "lookback_days"}
            candidate_config["seed"] = cfg["project"]["seed"]
            model = build_lstm(7, x_inner_train.shape[-1], candidate_config)
            started = time.perf_counter()
            history = model.fit(
                x_inner_train,
                y[train[inner_train]],
                validation_data=(x_inner_valid, y[train[inner_valid]]),
                epochs=int(candidate.get("epochs", 100)),
                batch_size=int(candidate.get("batch_size", 64)),
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=int(candidate.get("patience", 10)),
                    restore_best_weights=True,
                )],
                verbose=0,
            )
            predicted = model.predict(x_inner_valid, verbose=0).reshape(-1)
            candidate_scores.append({
                "config": candidate,
                "rmse": regression_metrics(y[train[inner_valid]], predicted)["rmse"],
                "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
                "search_time_sec": time.perf_counter() - started,
            })
        candidate_scores.sort(key=lambda item: item["rmse"])
        chosen = candidate_scores[0]
        config = {key: value for key, value in chosen["config"].items() if key != "lookback_days"}
        config["seed"] = cfg["project"]["seed"] + int(fold["fold"])
        prep = SequencePreprocessor().fit(x[train])
        x_train, x_valid = prep.transform(x[train]), prep.transform(x[valid])
        tf.keras.backend.clear_session()
        model = build_lstm(7, x_train.shape[-1], config)
        started = time.perf_counter()
        model.fit(x_train, y[train], epochs=chosen["best_epoch"], batch_size=int(chosen["config"].get("batch_size", 64)), verbose=0)
        fit_time = time.perf_counter() - started
        started = time.perf_counter()
        prediction = model.predict(x_valid, verbose=0).reshape(-1)
        inference_time = time.perf_counter() - started
        metrics = regression_metrics(y[valid], prediction)

        for metric, value in metrics.items():
            if metric == "n_samples":
                continue
            result_rows.append({
                "model_id": "LSTM",
                "model_group": "deep_sequence",
                "model_family": "standalone LSTM",
                "dataset_split": "validation",
                "cv_scheme": "temporal_panel_expanding",
                "cv_purpose": "development_selection",
                "cv_fold": fold["fold"],
                "niveau": "fold",
                "metric": metric,
                "is_primary": metric == "rmse",
                "value": value,
                "n_samples": metrics["n_samples"],
                "train_size": len(train),
                "horizon_step": 1,
                "forecast_origin": "end_of_day_t",
                "random_state": config["seed"],
                "fit_time_sec": fit_time,
                "search_time_sec": sum(item["search_time_sec"] for item in candidate_scores),
                "inference_time_sec": inference_time,
                "n_parameters": model.count_params(),
                "hardware": platform.platform(),
                "n_features": len(feature_columns),
                "fold_file": "folds_temporal.json",
                "notes": "Standalone LSTM; same lookback, features, validation sample IDs, and training-only early stopping as CNN-LSTM.",
            })
        for index, prediction_value in enumerate(prediction):
            key = keys.iloc[valid[index]]
            prediction_rows.append({
                "sample_id": int(key.target_row),
                "station_id": key.station_id,
                "origin_timestamp": key.origin_timestamp,
                "target_timestamp": key.target_timestamp,
                "y_true": float(y[valid[index]]),
                "y_pred": float(prediction_value),
                "residual": float(y[valid[index]] - prediction_value),
                "dataset_split": "validation",
                "cv_scheme": "temporal_panel_expanding",
                "cv_fold": fold["fold"],
                "model_id": "LSTM",
                "config_id": f"fold{fold['fold']}_rank1",
                "seed": config["seed"],
            })
        importance = sequence_channel_permutation(
            model, x_valid, y[valid], prep.feature_names_out(feature_columns),
            seed=config["seed"], n_repeats=10,
        )
        importance_rows.extend({
            "model_id": "LSTM",
            "model_group": "deep_sequence",
            "cv_repetition": 1,
            "cv_fold": fold["fold"],
            "method": "sequence_channel_permutation_rmse",
            "is_mandatory": True,
            "feature": row.feature,
            "importance": row.importance,
            "rank": row.rank,
        } for row in importance.itertuples())
        if any(item["config"]["lookback_days"] != 7 for item in candidate_scores[:3]):
            raise ValueError("Top standalone LSTM configurations must share the seven-day lookback")
        for rank, item in enumerate(candidate_scores[:3], 1):
            if rank == 1:
                candidate_prediction = prediction
            else:
                tf.keras.backend.clear_session()
                alternate_config = {key: value for key, value in item["config"].items() if key != "lookback_days"}
                alternate_config["seed"] = config["seed"]
                alternate = build_lstm(7, x_train.shape[-1], alternate_config)
                alternate.fit(x_train, y[train], epochs=item["best_epoch"], batch_size=int(item["config"].get("batch_size", 64)), verbose=0)
                candidate_prediction = alternate.predict(x_valid, verbose=0).reshape(-1)
            candidate_metrics = regression_metrics(y[valid], candidate_prediction)
            top3_rows.append({
                "model_id": "LSTM",
                "cv_fold": fold["fold"],
                "config_rank": rank,
                "inner_validation_rmse": item["rmse"],
                "best_epoch": item["best_epoch"],
                "outer_validation_rmse": candidate_metrics["rmse"],
                "outer_validation_mae": candidate_metrics["mae"],
                "outer_validation_r2": candidate_metrics["r2"],
                "n_samples": candidate_metrics["n_samples"],
                "configuration": json.dumps(item["config"], sort_keys=True),
            })
        (out / "models").mkdir(parents=True, exist_ok=True)
        (out / "models" / f"lstm_temporal_fold_{fold['fold']}_config.json").write_text(
            json.dumps({
                **chosen["config"],
                "seed": config["seed"],
                "best_epoch": chosen["best_epoch"],
                "inner_validation_rmse": chosen["rmse"],
                "top3_inner_configs": [
                    {"configuration": item["config"], "inner_validation_rmse": item["rmse"], "best_epoch": item["best_epoch"]}
                    for item in candidate_scores[:3]
                ],
            }, indent=2),
            encoding="utf-8",
        )

    result_table = add_fold_summaries(pd.DataFrame(result_rows, columns=RESULTS), seed=cfg["project"]["seed"])
    results_path = out / "results.csv"
    if results_path.exists():
        previous = pd.read_csv(results_path)
        previous = previous[~((previous.model_id == "LSTM") & (previous.cv_scheme == "temporal_panel_expanding"))]
        result_table = pd.concat([previous, result_table], ignore_index=True)
    result_table.to_csv(results_path, index=False, encoding="utf-8")
    comparison_table(result_table, seed=cfg["project"]["seed"]).to_csv(out / "comparisons.csv", index=False, encoding="utf-8")

    prediction_path = out / "predictions" / "lstm_temporal_oof.csv"
    prediction_table = pd.DataFrame(prediction_rows)
    prediction_table.to_csv(prediction_path, index=False, encoding="utf-8")
    top3_path = out / "lstm_top3_configs.csv"
    pd.DataFrame(top3_rows).to_csv(top3_path, index=False, encoding="utf-8")
    _write_model_family_ablation(out, prediction_table, pd.read_csv(cnn_path), cfg)

    importance_path = out / "feature_importance.csv"
    importance_table = pd.DataFrame(importance_rows, columns=FEATURE_IMPORTANCE)
    if importance_path.exists():
        previous = pd.read_csv(importance_path)
        previous = previous[~((previous.model_id == "LSTM") & (previous.method == "sequence_channel_permutation_rmse"))]
        importance_table = pd.concat([previous, importance_table], ignore_index=True)
    importance_table.to_csv(importance_path, index=False, encoding="utf-8")
    fold_ranking_stability(importance_table).to_csv(out / "ranking_stability.csv", index=False, encoding="utf-8")
    return {"model_id": "LSTM", "folds": len(folds), "predictions": len(prediction_rows), "feature_count": len(feature_columns)}


def run_cnn_lstm_sensitivity(df, cfg):
    """Perturb one predeclared CNN-LSTM hyperparameter step around each fold's selected config."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("CNN-LSTM sensitivity requires TensorFlow; install the [deep] extra") from exc
    from .cnn_lstm_benchmark import _prepare
    from .models.cnn_lstm import build_cnn_lstm

    root = Path(cfg["_config_dir"])
    out = root / cfg["paths"]["results_root"] / cfg["project"]["group_id"]
    fold_path = out / "folds_temporal.json"
    prediction_path = out / "predictions" / "cnn_lstm_temporal_oof.csv"
    if not fold_path.exists() or not prediction_path.exists():
        raise FileNotFoundError("Run CNN-LSTM development benchmark before its sensitivity analysis")
    folds = json.loads(fold_path.read_text(encoding="utf-8"))["folds"]
    feature_table = build_origin_features(df, cfg)
    feature_columns = _sequence_feature_columns(cfg, feature_table)
    grids = {
        "filters": [32, 64, 128],
        "kernel_size": [2, 3, 5],
        "lstm_units": [32, 64, 128],
        "dropout": [0.0, 0.1, 0.2, 0.3],
        "learning_rate": [0.0001, 0.0005, 0.001],
    }
    original_predictions = pd.read_csv(prediction_path)
    rows = []
    for fold in folds:
        config_path = out / "models" / f"cnn_lstm_temporal_fold_{fold['fold']}_config.json"
        selected = json.loads(config_path.read_text(encoding="utf-8"))
        lookback = int(selected["lookback_days"])
        x, y, keys, train, valid = _prepare(
            feature_table, feature_columns, lookback, set(fold["train_indices"]), set(fold["validation_indices"])
        )
        reference = original_predictions[
            (original_predictions.cv_fold == fold["fold"])
            & (original_predictions.model_id == "CNN_LSTM")
        ].copy()
        reference["sample_id"] = pd.to_numeric(reference.sample_id, errors="raise").astype(int)
        expected = keys.iloc[valid].target_row.astype(int).to_numpy()
        if set(reference.sample_id) != set(expected):
            raise ValueError(f"CNN-LSTM sensitivity fold {fold['fold']} does not match the saved validation cohort")
        reference_by_id = reference.set_index("sample_id").y_pred
        reference_prediction = np.asarray([reference_by_id.loc[int(sample_id)] for sample_id in expected], dtype=float)
        reference_rmse = regression_metrics(y[valid], reference_prediction)["rmse"]
        prep = SequencePreprocessor().fit(x[train])
        x_train, x_valid = prep.transform(x[train]), prep.transform(x[valid])
        best_epoch = int(selected["best_epoch"])
        config_seed = int(selected["seed"])
        for parameter, values in grids.items():
            for value in adjacent_values(values, selected[parameter]):
                candidate_config = {
                    key: selected[key]
                    for key in ("filters", "kernel_size", "lstm_units", "dropout", "learning_rate", "dense_units", "loss")
                    if key in selected
                }
                candidate_config[parameter] = value
                candidate_config["seed"] = config_seed
                tf.keras.backend.clear_session()
                model = build_cnn_lstm(lookback, x_train.shape[-1], candidate_config)
                model.fit(x_train, y[train], epochs=best_epoch, batch_size=int(selected.get("batch_size", 64)), verbose=0)
                prediction = model.predict(x_valid, verbose=0).reshape(-1)
                metrics = regression_metrics(y[valid], prediction)
                rows.append({
                    "model_id": "CNN_LSTM",
                    "cv_fold": fold["fold"],
                    "parameter": parameter,
                    "variant": "one_step_lower" if value < selected[parameter] else "one_step_upper",
                    "reference_value": selected[parameter],
                    "sensitivity_value": value,
                    "best_epoch_fixed": best_epoch,
                    "reference_rmse": reference_rmse,
                    "rmse": metrics["rmse"],
                    "delta_rmse_vs_selected": metrics["rmse"] - reference_rmse,
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "n_samples": metrics["n_samples"],
                    "configuration": json.dumps(candidate_config, sort_keys=True),
                })
    table = pd.DataFrame(rows)
    table.to_csv(out / "cnn_lstm_sensitivity_folds.csv", index=False, encoding="utf-8")
    summary = []
    for (parameter, variant), group in table.groupby(["parameter", "variant"], sort=True):
        deltas = group.delta_rmse_vs_selected.to_numpy(dtype=float)
        low, high = paired_bootstrap_ci(deltas, seed=cfg["project"]["seed"]) if len(deltas) > 1 else (np.nan, np.nan)
        baseline = float(group.reference_rmse.mean())
        mean_delta = float(deltas.mean())
        summary.append({
            "parameter": parameter,
            "variant": variant,
            "mean_rmse": float(group.rmse.mean()),
            "mean_delta_rmse": mean_delta,
            "delta_ci_lower": low,
            "delta_ci_upper": high,
            "relative_delta": mean_delta / baseline if baseline else np.nan,
            "stability_verdict": "stable_within_5pct" if abs(mean_delta) <= 0.05 * baseline else "sensitive_over_5pct",
            "n_paired_folds": len(deltas),
        })
    pd.DataFrame(summary).to_csv(out / "cnn_lstm_sensitivity_summary.csv", index=False, encoding="utf-8")
    return {"fold_rows": len(table), "parameters": sorted(table.parameter.unique().tolist())}


def run_lstm_sensitivity(df, cfg):
    """Perturb standalone LSTM hyperparameters by one predeclared step on development folds."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("LSTM sensitivity requires TensorFlow; install the [deep] extra") from exc
    root = Path(cfg["_config_dir"])
    out = root / cfg["paths"]["results_root"] / cfg["project"]["group_id"]
    fold_path = out / "folds_temporal.json"
    prediction_path = out / "predictions" / "lstm_temporal_oof.csv"
    if not fold_path.exists() or not prediction_path.exists():
        raise FileNotFoundError("Run the LSTM development ablation before its sensitivity analysis")
    folds = json.loads(fold_path.read_text(encoding="utf-8"))["folds"]
    feature_table = build_origin_features(df, cfg)
    feature_columns = _sequence_feature_columns(cfg, feature_table)
    x, y, keys = make_sequences(feature_table, feature_columns, lookback=7, horizon=1)
    target_rows = keys.target_row.to_numpy(dtype=int)
    grids = {
        "lstm_units": [32, 64, 128],
        "dropout": [0.0, 0.1, 0.2, 0.3, 0.5],
        "learning_rate": [0.0001, 0.0005, 0.001],
    }
    original_predictions = pd.read_csv(prediction_path)
    rows = []
    for fold in folds:
        selected = json.loads((out / "models" / f"lstm_temporal_fold_{fold['fold']}_config.json").read_text(encoding="utf-8"))
        train = np.flatnonzero(np.isin(target_rows, fold["train_indices"]))
        valid = np.flatnonzero(np.isin(target_rows, fold["validation_indices"]))
        if not len(train) or not len(valid):
            raise ValueError(f"No complete LSTM windows for sensitivity in fold {fold.get('fold')}")
        reference = original_predictions[original_predictions.cv_fold == fold["fold"]].copy()
        reference["sample_id"] = pd.to_numeric(reference.sample_id, errors="raise").astype(int)
        expected = keys.iloc[valid].target_row.astype(int).to_numpy()
        if set(reference.sample_id) != set(expected):
            raise ValueError(f"LSTM sensitivity fold {fold['fold']} does not match the saved validation cohort")
        reference_by_id = reference.set_index("sample_id").y_pred
        reference_prediction = np.asarray([reference_by_id.loc[int(sample_id)] for sample_id in expected], dtype=float)
        reference_rmse = regression_metrics(y[valid], reference_prediction)["rmse"]
        prep = SequencePreprocessor().fit(x[train])
        x_train, x_valid = prep.transform(x[train]), prep.transform(x[valid])
        for parameter, values in grids.items():
            for value in adjacent_values(values, selected[parameter]):
                candidate_config = {
                    key: selected[key]
                    for key in ("lstm_units", "dense_units", "dropout", "learning_rate", "loss")
                    if key in selected
                }
                candidate_config[parameter] = value
                candidate_config["seed"] = int(selected["seed"])
                tf.keras.backend.clear_session()
                model = build_lstm(7, x_train.shape[-1], candidate_config)
                model.fit(x_train, y[train], epochs=int(selected["best_epoch"]), batch_size=int(selected.get("batch_size", 64)), verbose=0)
                prediction = model.predict(x_valid, verbose=0).reshape(-1)
                metrics = regression_metrics(y[valid], prediction)
                rows.append({
                    "model_id": "LSTM",
                    "cv_fold": fold["fold"],
                    "parameter": parameter,
                    "variant": "one_step_lower" if value < selected[parameter] else "one_step_upper",
                    "reference_value": selected[parameter],
                    "sensitivity_value": value,
                    "best_epoch_fixed": int(selected["best_epoch"]),
                    "reference_rmse": reference_rmse,
                    "rmse": metrics["rmse"],
                    "delta_rmse_vs_selected": metrics["rmse"] - reference_rmse,
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "n_samples": metrics["n_samples"],
                    "configuration": json.dumps(candidate_config, sort_keys=True),
                })
    table = pd.DataFrame(rows)
    table.to_csv(out / "lstm_sensitivity_folds.csv", index=False, encoding="utf-8")
    summary = []
    for (parameter, variant), group in table.groupby(["parameter", "variant"], sort=True):
        deltas = group.delta_rmse_vs_selected.to_numpy(dtype=float)
        low, high = paired_bootstrap_ci(deltas, seed=cfg["project"]["seed"]) if len(deltas) > 1 else (np.nan, np.nan)
        baseline = float(group.reference_rmse.mean())
        mean_delta = float(deltas.mean())
        summary.append({
            "parameter": parameter, "variant": variant, "mean_rmse": float(group.rmse.mean()),
            "mean_delta_rmse": mean_delta, "delta_ci_lower": low, "delta_ci_upper": high,
            "relative_delta": mean_delta / baseline if baseline else np.nan,
            "stability_verdict": "stable_within_5pct" if abs(mean_delta) <= 0.05 * baseline else "sensitive_over_5pct",
            "n_paired_folds": len(deltas),
        })
    pd.DataFrame(summary).to_csv(out / "lstm_sensitivity_summary.csv", index=False, encoding="utf-8")
    return {"fold_rows": len(table), "parameters": sorted(table.parameter.unique().tolist())}


def _write_model_family_ablation(out, lstm_predictions, cnn_predictions, cfg):
    keys = ["sample_id", "cv_fold"]
    lstm = lstm_predictions[["sample_id", "cv_fold", "y_true", "y_pred"]].copy()
    cnn = cnn_predictions[cnn_predictions.model_id == "CNN_LSTM"][["sample_id", "cv_fold", "y_true", "y_pred"]].copy()
    merged = lstm.merge(cnn, on=keys, suffixes=("_lstm", "_cnn"), validate="one_to_one")
    if len(merged) != len(lstm) or len(merged) != len(cnn):
        raise ValueError("LSTM and CNN-LSTM predictions do not cover the same validation sample IDs")
    if not np.allclose(merged.y_true_lstm, merged.y_true_cnn):
        raise ValueError("LSTM and CNN-LSTM prediction artifacts disagree on observed targets")
    rows = []
    fold_deltas = []
    for fold, group in merged.groupby("cv_fold", sort=True):
        for model_variant, prediction_column in (("standalone_LSTM", "y_pred_lstm"), ("CNN_LSTM", "y_pred_cnn")):
            metrics = regression_metrics(group.y_true_lstm, group[prediction_column])
            for metric in ("rmse", "mae", "r2"):
                rows.append({
                    "model_id": "CNN_LSTM",
                    "model_group": "primary_hybrid",
                    "ablation_axis": "model_family",
                    "variant": model_variant,
                    "cv_scheme": "temporal_panel_expanding",
                    "cv_fold": fold,
                    "metric": metric,
                    "value": metrics[metric],
                    "delta_vs_full": np.nan,
                    "random_state": cfg["project"]["seed"],
                    "notes": f"matched {len(group)} sequence validation rows; CNN-LSTM is reference",
                })
        fold_deltas.append({
            "cv_fold": fold,
            "rmse_delta_lstm_minus_cnn": regression_metrics(group.y_true_lstm, group.y_pred_lstm)["rmse"]
            - regression_metrics(group.y_true_lstm, group.y_pred_cnn)["rmse"],
        })
    table = pd.DataFrame(rows)
    for index, row in table.iterrows():
        reference = table[
            (table.cv_fold == row.cv_fold)
            & (table.metric == row.metric)
            & (table.variant == "CNN_LSTM")
        ].value.iloc[0]
        table.loc[index, "delta_vs_full"] = row.value - reference

    ablation_path = out / "ablation.csv"
    if ablation_path.exists():
        previous = pd.read_csv(ablation_path)
        previous = previous[~((previous.ablation_axis == "model_family") & (previous.model_id == "CNN_LSTM"))]
        table = pd.concat([previous, table.reindex(columns=ABLATION)], ignore_index=True)
    else:
        table = table.reindex(columns=ABLATION)
    table.to_csv(ablation_path, index=False, encoding="utf-8")
    deltas = np.array([row["rmse_delta_lstm_minus_cnn"] for row in fold_deltas], dtype=float)
    low, high = paired_bootstrap_ci(deltas, seed=cfg["project"]["seed"])
    pd.DataFrame([{
        "comparison": "standalone_LSTM_minus_CNN_LSTM",
        "mean_rmse_delta": float(deltas.mean()),
        "ci_lower": low,
        "ci_upper": high,
        "n_paired_folds": len(deltas),
        "n_samples_per_fold": [int(len(group)) for _, group in merged.groupby("cv_fold")],
    }]).to_csv(out / "model_family_ablation_summary.csv", index=False, encoding="utf-8")
