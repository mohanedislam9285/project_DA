"""Paired metrics, cluster bootstrap intervals, and tail diagnostics for OOF data."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "16"
EXP = OUT / "architecture_experiment"
AUDIT = OUT / "reproducibility_audit"
AUDIT.mkdir(parents=True, exist_ok=True)
base = pd.read_csv(AUDIT / "reloaded_temporal_oof_and_persistence.csv")
arch = pd.read_csv(EXP / "architecture_oof_predictions.csv")
if arch.duplicated(["variant", "sample_id"]).any():
    raise ValueError("Architecture OOF file has duplicate variant/sample IDs")
reference_ids = set(base.sample_id.astype(int))
reference = base.set_index("sample_id").sort_index()
variants = sorted(arch.variant.unique())
predictions: dict[str, pd.Series] = {
    "CNN_LSTM_OPTUNA_reference": reference.cnn_lstm_prediction_reloaded.astype(float),
    "persistence": reference.persistence_prediction_origin_pm25.astype(float),
}
for variant in variants:
    g = arch[arch.variant.eq(variant)].copy().set_index("sample_id").sort_index()
    if set(g.index.astype(int)) != reference_ids:
        raise ValueError(f"Variant {variant} does not cover the same OOF sample IDs")
    if not np.allclose(g.y_true.to_numpy(), reference.y_true.to_numpy(), atol=1e-6, rtol=0):
        raise ValueError(f"Variant {variant} target arrays differ from reference")
    predictions[variant] = g.y_pred.astype(float)

chosen_names = pd.read_csv(EXP / "selected_variant_by_fold.csv").set_index("cv_fold").selected_variant.to_dict()
chosen = arch[arch.apply(lambda r: str(chosen_names[int(r.cv_fold)]) == r.variant, axis=1)].copy()
chosen = chosen.sort_values("sample_id").set_index("sample_id")
if set(chosen.index.astype(int)) != reference_ids:
    raise ValueError("Nested inner-selected architecture does not cover reference OOF IDs")
predictions["nested_inner_selected_CNN_LSTM"] = chosen.y_pred.astype(float)

y = reference.y_true.to_numpy(dtype=float)
station = reference.station_id.astype(str).to_numpy()
fold = reference.cv_fold.to_numpy(dtype=int)
stations, station_code = np.unique(station, return_inverse=True)
fold_ids = np.sort(np.unique(fold))
rng = np.random.default_rng(42016)
B = 20_000

# Precompute station-cluster sufficient statistics for all predictions.
cluster_n = np.bincount(station_code, minlength=len(stations)).astype(float)
cluster_y = np.bincount(station_code, weights=y, minlength=len(stations))
cluster_y2 = np.bincount(station_code, weights=y * y, minlength=len(stations))
draws = rng.multinomial(len(stations), np.full(len(stations), 1 / len(stations)), size=B)
denom = draws @ cluster_n
sum_y = draws @ cluster_y
sum_y2 = draws @ cluster_y2
sst = sum_y2 - sum_y * sum_y / denom


def metrics(yy: np.ndarray, pp: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(mean_squared_error(yy, pp) ** 0.5),
        "mae": float(mean_absolute_error(yy, pp)),
        "r2": float(r2_score(yy, pp)),
    }


def boot_mean_ci(values: np.ndarray) -> tuple[float, float]:
    samples = rng.choice(values, size=(B, len(values)), replace=True).mean(axis=1)
    return tuple(float(v) for v in np.quantile(samples, [0.025, 0.975]))


def exact_sign_p(values: np.ndarray) -> float:
    observed = abs(float(np.mean(values)))
    permuted = [abs(float(np.mean(values * np.asarray(signs)))) for signs in itertools.product([-1.0, 1.0], repeat=len(values))]
    return float(np.mean(np.asarray(permuted) >= observed - 1e-12))


fold_rows = []
for model_id, pred_series in predictions.items():
    pred = pred_series.to_numpy(dtype=float)
    for f in fold_ids:
        ix = np.flatnonzero(fold == f)
        scored = metrics(y[ix], pred[ix])
        fold_rows.append({"model_id": model_id, "cv_fold": int(f), "n": int(len(ix)), **scored})
pd.DataFrame(fold_rows).to_csv(EXP / "paired_metrics_by_fold.csv", index=False)

summary_rows = []
bootstrap_distributions: dict[str, dict[str, np.ndarray]] = {}
for model_id, pred_series in predictions.items():
    pred = pred_series.to_numpy(dtype=float)
    point = metrics(y, pred)
    errors = pred - y
    cluster_sse = np.bincount(station_code, weights=errors**2, minlength=len(stations))
    cluster_abs = np.bincount(station_code, weights=np.abs(errors), minlength=len(stations))
    mse_boot = (draws @ cluster_sse) / denom
    mae_boot = (draws @ cluster_abs) / denom
    r2_boot = 1.0 - (draws @ cluster_sse) / sst
    # Fold bootstrap reports uncertainty of the fold-mean metric, with only five dependent folds.
    fold_values = pd.DataFrame(fold_rows)
    f_rmse = fold_values[fold_values.model_id.eq(model_id)].rmse.to_numpy()
    f_mae = fold_values[fold_values.model_id.eq(model_id)].mae.to_numpy()
    rmse_fold_ci = boot_mean_ci(f_rmse)
    mae_fold_ci = boot_mean_ci(f_mae)
    summary_rows.append({
        "model_id": model_id, "n_oof": len(y), "n_stations": len(stations),
        "pooled_rmse": point["rmse"], "pooled_rmse_station_cluster_ci_low": float(np.quantile(np.sqrt(mse_boot), .025)),
        "pooled_rmse_station_cluster_ci_high": float(np.quantile(np.sqrt(mse_boot), .975)),
        "pooled_mae": point["mae"], "pooled_mae_station_cluster_ci_low": float(np.quantile(mae_boot, .025)),
        "pooled_mae_station_cluster_ci_high": float(np.quantile(mae_boot, .975)),
        "pooled_r2": point["r2"], "pooled_r2_station_cluster_ci_low": float(np.quantile(r2_boot, .025)),
        "pooled_r2_station_cluster_ci_high": float(np.quantile(r2_boot, .975)),
        "mean_fold_rmse": float(f_rmse.mean()), "sd_fold_rmse": float(f_rmse.std(ddof=1)),
        "mean_fold_rmse_fold_bootstrap_ci_low": rmse_fold_ci[0], "mean_fold_rmse_fold_bootstrap_ci_high": rmse_fold_ci[1],
        "mean_fold_mae": float(f_mae.mean()), "sd_fold_mae": float(f_mae.std(ddof=1)),
        "mean_fold_mae_fold_bootstrap_ci_low": mae_fold_ci[0], "mean_fold_mae_fold_bootstrap_ci_high": mae_fold_ci[1],
    })
    bootstrap_distributions[model_id] = {"mse": mse_boot, "mae": mae_boot}
pd.DataFrame(summary_rows).to_csv(EXP / "oof_metric_summary.csv", index=False)

comparison_rows = []
baseline = "persistence"
for model_id in [m for m in predictions if m != baseline]:
    model_pred = predictions[model_id].to_numpy(dtype=float)
    base_pred = predictions[baseline].to_numpy(dtype=float)
    fold_delta = []
    for f in fold_ids:
        ix = np.flatnonzero(fold == f)
        fold_delta.append(float(mean_squared_error(y[ix], model_pred[ix]) ** 0.5 - mean_squared_error(y[ix], base_pred[ix]) ** 0.5))
    fold_delta = np.asarray(fold_delta)
    fold_ci = boot_mean_ci(fold_delta)
    cluster_sse_model = np.bincount(station_code, weights=(model_pred-y)**2, minlength=len(stations))
    cluster_sse_base = np.bincount(station_code, weights=(base_pred-y)**2, minlength=len(stations))
    cluster_abs_model = np.bincount(station_code, weights=np.abs(model_pred-y), minlength=len(stations))
    cluster_abs_base = np.bincount(station_code, weights=np.abs(base_pred-y), minlength=len(stations))
    rmse_delta_boot = np.sqrt((draws @ cluster_sse_model) / denom) - np.sqrt((draws @ cluster_sse_base) / denom)
    mae_delta_boot = (draws @ cluster_abs_model) / denom - (draws @ cluster_abs_base) / denom
    p_exact = exact_sign_p(fold_delta)
    try:
        p_wilcoxon = float(wilcoxon(fold_delta, alternative="two-sided", method="exact").pvalue)
    except ValueError:
        p_wilcoxon = 1.0
    comparison_rows.append({
        "model_id": model_id, "baseline_id": baseline,
        "mean_fold_rmse_delta_model_minus_persistence": float(fold_delta.mean()),
        "fold_rmse_differences": json.dumps(fold_delta.tolist()),
        "fold_bootstrap_ci_low_descriptive": fold_ci[0], "fold_bootstrap_ci_high_descriptive": fold_ci[1],
        "exact_sign_flip_p_two_sided_n5": p_exact, "wilcoxon_exact_p_two_sided_n5": p_wilcoxon,
        "mean_fold_mae_delta_model_minus_persistence": float(np.mean([
            mean_absolute_error(y[fold==f], model_pred[fold==f]) - mean_absolute_error(y[fold==f], base_pred[fold==f]) for f in fold_ids
        ])),
        "pooled_rmse_delta": float(mean_squared_error(y, model_pred)**0.5 - mean_squared_error(y, base_pred)**0.5),
        "pooled_rmse_delta_station_cluster_ci_low": float(np.quantile(rmse_delta_boot, .025)),
        "pooled_rmse_delta_station_cluster_ci_high": float(np.quantile(rmse_delta_boot, .975)),
        "pooled_mae_delta": float(mean_absolute_error(y, model_pred)-mean_absolute_error(y, base_pred)),
        "pooled_mae_delta_station_cluster_ci_low": float(np.quantile(mae_delta_boot, .025)),
        "pooled_mae_delta_station_cluster_ci_high": float(np.quantile(mae_delta_boot, .975)),
    })
comparisons = pd.DataFrame(comparison_rows)
comparisons["holm_p_across_architecture_comparisons"] = multipletests(comparisons.wilcoxon_exact_p_two_sided_n5.to_numpy(), method="holm")[1]
comparisons.to_csv(EXP / "paired_comparisons_vs_persistence.csv", index=False)

# Concentration bands, calibrated bias, extreme-event underprediction, and squared-error shares.
bands = [(0,10,"0-10"),(10,25,"10-25"),(25,50,"25-50"),(50,np.inf,">=50")]
band_rows=[]
for model_id, (lo, hi, band_name) in itertools.product(predictions.keys(), bands):
    mask = (y >= lo) & (y < hi)
    if not mask.any():
        continue
    p = predictions[model_id].to_numpy(dtype=float)[mask]
    true = y[mask]
    err = p-true
    squared = (p-true)**2
    all_err=(predictions[model_id].to_numpy(dtype=float)-y)**2
    band_rows.append({
        "model_id": model_id, "concentration_band_ug_m3": band_name, "n": int(mask.sum()),
        "target_mean": float(true.mean()), "prediction_mean": float(p.mean()),
        "rmse": float(np.mean(squared)**.5), "mae": float(np.mean(np.abs(err))),
        "prediction_minus_actual_bias": float(err.mean()), "underprediction_rate": float(np.mean(err<0)),
        "share_of_model_pooled_squared_error": float(squared.sum()/all_err.sum()),
    })
pd.DataFrame(band_rows).to_csv(EXP / "concentration_band_error_analysis.csv", index=False)

methods = {
    "n_outer_folds": int(len(fold_ids)),
    "paired_unit_primary": "fold-level RMSE difference, exact sign-flip and exact Wilcoxon; low power with five dependent folds",
    "fold_intervals": "bootstrap resampling of the five fold values; descriptive only because folds are dependent",
    "cluster_intervals": "20,000 paired bootstrap resamples of station IDs, preserving all within-station OOF days; conditional on the observed time folds and not a time-regime bootstrap",
    "holms_family": ["all reported CNN-LSTM variants and the nested inner-selected procedure versus persistence"],
    "random_seed": 42016,
    "future_test_labels_or_metrics_used": False,
}
(EXP / "uncertainty_method.json").write_text(json.dumps(methods, indent=2), encoding="utf-8")
print(json.dumps({
    "summary": pd.DataFrame(summary_rows)[["model_id","pooled_rmse","pooled_mae","pooled_r2","mean_fold_rmse","mean_fold_mae"]].to_dict(orient="records"),
    "comparisons": comparisons[["model_id","mean_fold_rmse_delta_model_minus_persistence","fold_bootstrap_ci_low_descriptive","fold_bootstrap_ci_high_descriptive","exact_sign_flip_p_two_sided_n5","wilcoxon_exact_p_two_sided_n5","holm_p_across_architecture_comparisons","pooled_rmse_delta_station_cluster_ci_low","pooled_rmse_delta_station_cluster_ci_high"]].to_dict(orient="records"),
}, indent=2))
