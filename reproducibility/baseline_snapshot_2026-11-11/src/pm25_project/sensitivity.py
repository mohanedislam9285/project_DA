"""Development-only hyperparameter and preprocessing sensitivity studies."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import supervised_frame
from .metrics import paired_bootstrap_ci, regression_metrics
from .models.model_registry import model_registry
from .schemas import ABLATION
from .tuning import candidate_configs, chronological_inner_indices, select_configuration


def adjacent_values(values, reference):
    """Return only the immediate predeclared neighbors of a reference value."""
    ordered = list(values)
    if reference not in ordered:
        return []
    position = ordered.index(reference)
    return [ordered[i] for i in (position - 1, position + 1) if 0 <= i < len(ordered)]


def _development_inputs(df, cfg):
    root = Path(cfg["_config_dir"])
    out = root / cfg["paths"]["results_root"] / cfg["project"]["group_id"]
    fold_path = out / "folds_temporal.json"
    if not fold_path.exists():
        raise FileNotFoundError("Run the development folds stage before sensitivity analysis")
    folds = json.loads(fold_path.read_text(encoding="utf-8")).get("folds", [])
    if not folds:
        raise ValueError("No temporal development folds are available")
    samples, features = supervised_frame(df, cfg)
    target_rows = samples.target_row.to_numpy(dtype=int)
    dates = pd.to_datetime(samples.target_timestamp, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    return out, folds, samples, features, target_rows, dates


def _fold_sample_indices(target_rows, fold):
    train = np.flatnonzero(np.isin(target_rows, fold["train_indices"]))
    valid = np.flatnonzero(np.isin(target_rows, fold["validation_indices"]))
    if not len(train) or not len(valid):
        raise ValueError(f"Temporal fold {fold.get('fold')} has no supervised train/validation samples")
    return train, valid


def run_hyperparameter_sensitivity(df, cfg, model_ids=None):
    """Score each model's top three inner candidates and one-step neighbors on outer folds."""
    out, folds, samples, features, target_rows, dates = _development_inputs(df, cfg)
    registry = model_registry(cfg["project"]["seed"])
    declared = cfg.get("_search_spaces", {}).get("hyperparameter_sensitivity_levels", {})
    model_ids = list(model_ids or [m for m in declared if m in registry])
    if not model_ids:
        raise ValueError("No sensitivity models are available for the active project registry; this project exposes a CNN_LSTM-only configuration.")
    unknown = sorted(set(model_ids) - set(registry))
    missing = sorted(set(model_ids) - set(declared))
    if unknown or missing:
        raise ValueError(f"Unsupported sensitivity model(s); unavailable={unknown}, no declared key parameter={missing}")

    x = samples[features].to_numpy(dtype=float)
    y = samples.y.to_numpy(dtype=float)
    rows = []
    for fold in folds:
        train, valid = _fold_sample_indices(target_rows, fold)
        for model_id in model_ids:
            _require_sklearn_style_model(model_id, registry, "hyperparameter sensitivity")
            config_spec = declared[model_id]
            parameter = config_spec["parameter"]
            candidates = select_configuration(model_id, x[train], y[train], dates[train], cfg["project"]["seed"], cfg)
            candidates = [candidate for candidate in candidates if np.isfinite(candidate["rmse"])][:3]
            if not candidates:
                raise RuntimeError(f"No valid inner configurations for {model_id}, fold {fold.get('fold')}")
            selected_params = candidates[0]["params"]
            if parameter not in selected_params:
                raise ValueError(f"Sensitivity key {parameter} is not in the selected {model_id} configuration")
            variants = []
            for rank, candidate in enumerate(candidates, 1):
                variants.append((f"top3_rank_{rank}", candidate["params"], rank, candidate["rmse"]))
            for value in adjacent_values(config_spec["values"], selected_params[parameter]):
                params = dict(selected_params)
                params[parameter] = value
                side = "lower" if value < selected_params[parameter] else "upper"
                variants.append((f"one_step_{side}", params, 1, None))

            fold_scores = {}
            for variant, params, rank, inner_rmse in variants:
                if inner_rmse is None:
                    inner_train, inner_valid = chronological_inner_indices(dates[train])
                    inner_estimator = registry[model_id]()
                    inner_estimator.set_params(**params)
                    inner_estimator.fit(x[train[inner_train]], y[train[inner_train]])
                    inner_rmse = regression_metrics(
                        y[train[inner_valid]],
                        inner_estimator.predict(x[train[inner_valid]]),
                    )["rmse"]
                estimator = registry[model_id]()
                estimator.set_params(**params)
                estimator.fit(x[train], y[train])
                metrics = regression_metrics(y[valid], estimator.predict(x[valid]))
                record = {
                    "model_id": model_id,
                    "cv_scheme": "temporal_panel_expanding",
                    "cv_fold": fold["fold"],
                    "variant": variant,
                    "config_rank": rank,
                    "sensitivity_parameter": parameter,
                    "sensitivity_value": params[parameter],
                    "configuration": json.dumps(params, sort_keys=True, default=str),
                    "inner_validation_rmse": inner_rmse,
                    **{key: value for key, value in metrics.items() if key != "n_samples"},
                    "n_samples": metrics["n_samples"],
                    "random_state": cfg["project"]["seed"],
                }
                rows.append(record)
                fold_scores[variant] = metrics["rmse"]
            reference = fold_scores["top3_rank_1"]
            for record in rows[-len(variants):]:
                record["delta_rmse_vs_selected"] = record["rmse"] - reference

    fold_table = pd.DataFrame(rows)
    if fold_table.empty:
        raise ValueError("No sensitivity rows were generated for the selected model registry; this project exposes a CNN_LSTM-only configuration.")
    fold_path = out / "hyperparameter_sensitivity_folds.csv"
    summary_rows = []
    for (model_id, variant), group in fold_table.groupby(["model_id", "variant"], sort=True):
        deltas = group.delta_rmse_vs_selected.to_numpy(dtype=float)
        low, high = paired_bootstrap_ci(deltas, seed=cfg["project"]["seed"]) if len(deltas) > 1 else (np.nan, np.nan)
        baseline = fold_table[(fold_table.model_id == model_id) & (fold_table.variant == "top3_rank_1")]
        reference_rmse = float(baseline.rmse.mean())
        mean_delta = float(deltas.mean())
        summary_rows.append({
            "model_id": model_id,
            "variant": variant,
            "sensitivity_parameter": group.sensitivity_parameter.iloc[0],
            "mean_rmse": float(group.rmse.mean()),
            "mean_delta_rmse_vs_selected": mean_delta,
            "delta_ci_lower": low,
            "delta_ci_upper": high,
            "relative_delta": mean_delta / reference_rmse if reference_rmse else np.nan,
            "stability_verdict": "stable_within_5pct" if abs(mean_delta) <= 0.05 * reference_rmse else "sensitive_over_5pct",
            "n_paired_folds": len(deltas),
        })
    fold_table.to_csv(fold_path, index=False, encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(out / "hyperparameter_sensitivity_summary.csv", index=False, encoding="utf-8")
    return {"models": model_ids, "fold_rows": len(fold_table), "fold_file": fold_path.name}


def _target_transform(values, transform):
    values = np.asarray(values, dtype=float)
    if transform == "raw":
        return values
    if transform == "log1p":
        if np.any(values <= -1):
            raise ValueError("log1p target transform requires all targets > -1")
        return np.log1p(values)
    raise ValueError(f"Unknown target transform: {transform}")


def _target_inverse(values, transform):
    values = np.asarray(values, dtype=float)
    return np.expm1(values) if transform == "log1p" else values


def _require_sklearn_style_model(model_id, registry, study_name):
    estimator = registry[model_id]()
    if not hasattr(estimator, "set_params") or not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
        raise ValueError(
            f"{model_id} is not compatible with the {study_name} study; this project exposes a CNN_LSTM-only model registry."
        )
    return estimator


def _select_original_scale_configuration(model_id, x, y, dates, transform, add_indicator, cfg):
    inner_train, inner_valid = chronological_inner_indices(dates)
    transformed_y = _target_transform(y, transform)
    registry = model_registry(cfg["project"]["seed"])
    _require_sklearn_style_model(model_id, registry, "preprocessing ablation")
    scores = []
    for params in candidate_configs(model_id, cfg["project"]["seed"], cfg):
        estimator = registry[model_id]()
        if params:
            estimator.set_params(**params)
        estimator.set_params(simpleimputer__add_indicator=add_indicator)
        estimator.fit(x[inner_train], transformed_y[inner_train])
        prediction = _target_inverse(estimator.predict(x[inner_valid]), transform)
        score = regression_metrics(y[inner_valid], prediction)["rmse"]
        scores.append((score, params))
    if not scores:
        raise RuntimeError(f"No candidate configurations were declared for {model_id}")
    scores.sort(key=lambda item: item[0])
    return scores[0][1]


def run_preprocessing_ablation(df, cfg, model_ids=None):
    """Compare indicators and target scale with fold-local transforms and shared rows."""
    out, folds, samples, features, target_rows, dates = _development_inputs(df, cfg)
    registry = model_registry(cfg["project"]["seed"])
    model_ids = list(model_ids or [m for m in ("ElasticNet", "SVR_RBF", "XGBoost", "Bagging", "CatBoost") if m in registry])
    if not model_ids:
        raise ValueError("No preprocessing-ablation models are available for the active project registry; this project exposes a CNN_LSTM-only configuration.")
    unknown = sorted(set(model_ids) - set(registry))
    if unknown:
        raise ValueError(f"Unavailable preprocessing-ablation models: {unknown}")
    variants = (
        ("median_indicator_on_raw", True, "raw"),
        ("median_indicator_off_raw", False, "raw"),
        ("median_indicator_on_log1p", True, "log1p"),
    )
    x = samples[features].to_numpy(dtype=float)
    y = samples.y.to_numpy(dtype=float)
    rows = []
    for fold in folds:
        train, valid = _fold_sample_indices(target_rows, fold)
        for model_id in model_ids:
            _require_sklearn_style_model(model_id, registry, "preprocessing ablation")
            for variant, add_indicator, transform in variants:
                transformed_y = _target_transform(y, transform)
                selected_params = _select_original_scale_configuration(
                    model_id, x[train], y[train], dates[train], transform, add_indicator, cfg
                )
                estimator = registry[model_id]()
                estimator.set_params(**selected_params, simpleimputer__add_indicator=add_indicator)
                estimator.fit(x[train], transformed_y[train])
                prediction = _target_inverse(estimator.predict(x[valid]), transform)
                metrics = regression_metrics(y[valid], prediction)
                for metric in ("rmse", "mae", "r2"):
                    rows.append({
                        "model_id": model_id,
                        "model_group": "comparators",
                        "ablation_axis": "preprocessing",
                        "variant": variant,
                        "cv_scheme": "temporal_panel_expanding",
                        "cv_fold": fold["fold"],
                        "metric": metric,
                        "value": metrics[metric],
                        "delta_vs_full": np.nan,
                        "random_state": cfg["project"]["seed"],
                        "notes": f"fold-local {transform} target; inner selection and outer metrics use original-scale PM2.5; median-imputation indicator={add_indicator}",
                    })

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No preprocessing-ablation rows were generated for the selected model registry; this project exposes a CNN_LSTM-only configuration.")
    reference = table[table.variant.eq("median_indicator_on_raw")].set_index(["model_id", "cv_fold", "metric"])
    for index, row in table.iterrows():
        table.loc[index, "delta_vs_full"] = row.value - reference.loc[(row.model_id, row.cv_fold, row.metric), "value"]
    interval_rows = []
    rmse_table = table[(table.metric == "rmse") & ~table.variant.eq("median_indicator_on_raw")]
    for (model_id, variant), group in rmse_table.groupby(["model_id", "variant"]):
        deltas = group.delta_vs_full.to_numpy(dtype=float)
        low, high = paired_bootstrap_ci(deltas, seed=cfg["project"]["seed"])
        interval_rows.append({"model_id": model_id, "variant": variant, "mean_delta_rmse": float(deltas.mean()), "ci_lower": low, "ci_upper": high, "n_paired_folds": len(deltas)})
        table.loc[group.index, "delta_ci_lower"] = low
        table.loc[group.index, "delta_ci_upper"] = high

    ablation_path = out / "ablation.csv"
    if ablation_path.exists():
        prior = pd.read_csv(ablation_path)
        prior = prior[~(prior.ablation_axis.eq("preprocessing") & prior.model_id.isin(model_ids))]
        table = pd.concat([prior, table.reindex(columns=ABLATION)], ignore_index=True)
    else:
        table = table.reindex(columns=ABLATION)
    table.to_csv(ablation_path, index=False, encoding="utf-8")
    pd.DataFrame(interval_rows).to_csv(out / "preprocessing_ablation_summary.csv", index=False, encoding="utf-8")
    return {"models": model_ids, "variants": [item[0] for item in variants], "n_rows": len(rows)}
