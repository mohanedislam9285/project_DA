"""Development-fold local error cases and feature-ranking stability."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .benchmark import supervised_frame


def run_local_explanations(df, cfg, model_ids=("Bagging", "CatBoost", "CNN_LSTM", "LSTM")):
    """Export matched development cases, their features, predictions, and residuals."""
    root = Path(cfg["_config_dir"])
    out = root / cfg["paths"]["results_root"] / cfg["project"]["group_id"]
    prediction_paths = [
        out / "predictions" / "temporal_oof.csv",
        out / "predictions" / "cnn_lstm_temporal_oof.csv",
        out / "predictions" / "lstm_temporal_oof.csv",
    ]
    frames = [pd.read_csv(path) for path in prediction_paths if path.exists()]
    if not frames:
        raise FileNotFoundError("Development temporal out-of-fold predictions are required")
    predictions = pd.concat(frames, ignore_index=True)
    predictions = predictions[
        predictions.dataset_split.eq("validation")
        & predictions.cv_scheme.eq("temporal_panel_expanding")
        & predictions.model_id.isin(model_ids)
    ].copy()
    absent_models = sorted(set(model_ids) - set(predictions.model_id))
    if absent_models:
        raise ValueError(f"No development predictions for requested local-explanation models: {absent_models}")
    predictions["sample_id"] = pd.to_numeric(predictions.sample_id, errors="raise").astype(int)
    coverage = predictions.groupby("sample_id").model_id.nunique()
    shared_ids = coverage[coverage == len(model_ids)].index
    shared = predictions[predictions.sample_id.isin(shared_ids)].copy()
    if shared.empty:
        raise ValueError("Selected models have no common development validation samples")
    samples, features = supervised_frame(df, cfg)
    expected_targets = samples.drop_duplicates("target_row").set_index("target_row").y
    predicted_targets = shared.groupby("sample_id").y_true.mean()
    expected_shared = expected_targets.reindex(predicted_targets.index)
    if expected_shared.isna().any() or not np.isclose(
        predicted_targets.to_numpy(dtype=float),
        expected_shared.to_numpy(dtype=float),
        rtol=1e-7,
        atol=1e-5,
    ).all():
        raise ValueError("Model prediction artifacts disagree with source targets for shared samples")
    wide = shared.pivot(index="sample_id", columns="model_id", values="y_pred")
    targets = expected_shared
    abs_errors = wide.sub(targets, axis=0).abs()
    representative = {}
    bands = (
        ("low_lt10", targets < 10, 5.0),
        ("medium_10_25", (targets >= 10) & (targets < 25), 17.5),
        ("high_25_50", (targets >= 25) & (targets <= 50), 37.5),
        ("extreme_gt50", targets > 50, None),
    )
    for label, mask, midpoint in bands:
        candidates = targets[mask]
        if candidates.empty:
            continue
        reference = float(candidates.median()) if midpoint is None else midpoint
        sample_id = int((candidates - reference).abs().idxmin())
        representative[label] = sample_id
    representative["largest_error"] = int(abs_errors.max(axis=1).idxmax())
    reasons = {}
    for reason, sample_id in representative.items():
        reasons.setdefault(sample_id, []).append(reason)

    feature_values = samples.set_index("target_row")[features]
    output = []
    for sample_id, case_reasons in reasons.items():
        if sample_id not in feature_values.index:
            raise ValueError(f"No origin features found for development sample {sample_id}")
        values = feature_values.loc[sample_id]
        if isinstance(values, pd.DataFrame):
            values = values.iloc[0]
        for model_id in model_ids:
            prediction = float(wide.loc[sample_id, model_id])
            actual = float(targets.loc[sample_id])
            output.append({
                "sample_id": sample_id,
                "case_reasons": "|".join(sorted(case_reasons)),
                "model_id": model_id,
                "target": actual,
                "prediction": prediction,
                "residual": actual - prediction,
                "absolute_error": abs(actual - prediction),
                "feature_values_json": json.dumps(values.to_dict(), sort_keys=True, default=str),
                "interpretation_note": "Descriptive local case; feature values and residuals are not causal explanations.",
            })
    pd.DataFrame(output).to_csv(out / "local_explanations.csv", index=False, encoding="utf-8")
    pd.DataFrame([
        {"case": label, "sample_id": sample_id, "target": float(targets.loc[sample_id]), "n_shared_models": len(model_ids)}
        for label, sample_id in representative.items()
    ]).to_csv(out / "local_explanation_cases.csv", index=False, encoding="utf-8")
    absent_bands = sorted({label for label, _, _ in bands} - set(representative))
    return {"models": list(model_ids), "shared_samples": int(len(shared_ids)), "cases": len(representative), "bands_absent": absent_bands}


def run_cross_scheme_ranking_stability(cfg):
    """Compare mean validation permutation-importance ranks across saved CV schemes."""
    root = Path(cfg["_config_dir"])
    out = root / cfg["paths"]["results_root"] / cfg["project"]["group_id"]
    sources = {
        "temporal_panel_expanding": out / "feature_importance.csv",
        "spatial_group_kfold": out / "logs" / "feature_importance_spatial_group_kfold.csv",
        "random_diagnostic": out / "logs" / "feature_importance_random_diagnostic.csv",
    }
    tables = []
    for scheme, path in sources.items():
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if {"model_id", "feature", "rank"}.issubset(table.columns):
            tables.append(table.assign(cv_scheme=scheme))
    if not tables:
        raise FileNotFoundError("No saved development permutation-importance tables were found")
    importance = pd.concat(tables, ignore_index=True)
    summary = importance.groupby(["model_id", "cv_scheme", "feature"], as_index=False).agg(mean_rank=("rank", "mean"))
    rows = []
    for model_id, model in summary.groupby("model_id", sort=True):
        schemes = sorted(model.cv_scheme.unique())
        for left_index, left_scheme in enumerate(schemes):
            for right_scheme in schemes[left_index + 1:]:
                left = model[model.cv_scheme.eq(left_scheme)].set_index("feature").mean_rank
                right = model[model.cv_scheme.eq(right_scheme)].set_index("feature").mean_rank
                common = left.index.intersection(right.index)
                if len(common) < 2:
                    continue
                rho = float(spearmanr(left.loc[common], right.loc[common]).statistic)
                top_left = set(left.sort_values().head(5).index)
                top_right = set(right.sort_values().head(5).index)
                union = top_left | top_right
                jaccard = len(top_left & top_right) / len(union) if union else 1.0
                rows.append({
                    "model_id": model_id,
                    "scheme_a": left_scheme,
                    "scheme_b": right_scheme,
                    "spearman_rho": rho,
                    "jaccard_top5": jaccard,
                    "rho_threshold_met": rho > 0.8,
                    "jaccard_threshold_met": jaccard > 0.6,
                    "n_common_features": len(common),
                    "notes": "Ranks are averaged over saved development folds; random CV is diagnostic only.",
                })
    result = pd.DataFrame(rows)
    result.to_csv(out / "ranking_stability_cross_scheme.csv", index=False, encoding="utf-8")
    return {"comparisons": len(result), "models": sorted(result.model_id.unique().tolist()) if len(result) else []}


def run_validation_scheme_ablation(cfg):
    """Report unpaired mean RMSE gaps across CV schemes as diagnostics, not selection evidence."""
    root = Path(cfg["_config_dir"])
    out = root / cfg["paths"]["results_root"] / cfg["project"]["group_id"]
    path = out / "results.csv"
    if not path.exists():
        raise FileNotFoundError("Run the development benchmark before validation-scheme comparison")
    results = pd.read_csv(path)
    eligible = results[
        results.dataset_split.eq("validation")
        & results.niveau.eq("fold")
        & results.metric.eq("rmse")
        & results.cv_scheme.isin(["temporal_panel_expanding", "spatial_group_kfold", "random_diagnostic"])
    ].copy()
    if "cv_purpose" in eligible:
        eligible = eligible[eligible.cv_purpose.isin(["development_selection", "development", "diagnostic_only"])]
    means = eligible.groupby(["model_id", "cv_scheme"], as_index=False).agg(
        mean_rmse=("value", "mean"), folds=("cv_fold", "nunique"), mean_samples=("n_samples", "mean")
    )
    wide = means.pivot(index="model_id", columns="cv_scheme", values="mean_rmse")
    rows = []
    for model_id, row in wide.iterrows():
        for comparison, left, right in (
            ("random_minus_temporal", "random_diagnostic", "temporal_panel_expanding"),
            ("random_minus_spatial", "random_diagnostic", "spatial_group_kfold"),
        ):
            if left not in row or right not in row or not np.isfinite(row[left]) or not np.isfinite(row[right]):
                continue
            left_meta = means[(means.model_id == model_id) & (means.cv_scheme == left)].iloc[0]
            right_meta = means[(means.model_id == model_id) & (means.cv_scheme == right)].iloc[0]
            rows.append({
                "model_id": model_id,
                "comparison": comparison,
                "rmse_scheme_a": float(row[left]),
                "rmse_scheme_b": float(row[right]),
                "delta_rmse": float(row[left] - row[right]),
                "folds_scheme_a": int(left_meta.folds),
                "folds_scheme_b": int(right_meta.folds),
                "mean_samples_scheme_a": float(left_meta.mean_samples),
                "mean_samples_scheme_b": float(right_meta.mean_samples),
                "interpretation": "Unpaired scheme-level diagnostic; not a model-selection estimate.",
            })
    table = pd.DataFrame(rows)
    table.to_csv(out / "validation_scheme_ablation.csv", index=False, encoding="utf-8")
    return {"comparisons": len(table), "models": sorted(table.model_id.unique().tolist()) if len(table) else []}
