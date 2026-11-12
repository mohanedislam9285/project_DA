"""Generate additional rubric evidence from development rows and OOF scores only."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pm25_project.config import load_config
from pm25_project.features import build_origin_features


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "16"
cfg, _ = load_config(ROOT / "configs" / "project.yaml")
df = pd.read_csv(ROOT / "data" / "processed" / "cleaned_daily.csv", parse_dates=["timestamp"])
df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
partition = json.loads((OUT / "future_test.json").read_text(encoding="utf-8"))
dev = df.iloc[np.asarray(partition["development_indices"], dtype=int)].copy()
cutoff = pd.to_datetime(partition["development_end"], utc=True)
assert (dev.timestamp <= cutoff).all()

# Fine-grained station/month/pollutant missingness, based only on development.
pollutants = [c for c in ["no2", "co", "o3"] if c in dev]
dev["month"] = dev.timestamp.dt.strftime("%Y-%m")
rows = []
for (station, month), group in dev.groupby(["station_id", "month"], sort=True):
    for variable in pollutants:
        rows.append({
            "station_id": station, "month": month, "variable": variable.upper(),
            "n_station_days": int(len(group)), "n_missing": int(group[variable].isna().sum()),
            "pct_missing": float(group[variable].isna().mean() * 100),
        })
missingness = pd.DataFrame(rows)
missingness.to_csv(OUT / "pollutant_missingness_station_month_development.csv", index=False, encoding="utf-8")

# Formal daily-series tests use each station's longest uninterrupted development segment.
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

diag_rows = []
for station, group in dev.sort_values("timestamp").groupby("station_id", sort=True):
    group = group.sort_values("timestamp")
    run_id = group.timestamp.diff().dt.days.ne(1).cumsum()
    runs = [g for _, g in group.groupby(run_id) if len(g) >= 14]
    if not runs:
        diag_rows.append({"station_id": station, "diagnostics_status": "no_contiguous_segment_of_14_days"})
        continue
    seg = max(runs, key=len)
    y = seg.pm25.to_numpy(dtype=float)
    max_acf_lag = min(30, len(y) // 2 - 1)
    max_pacf_lag = min(20, len(y) // 2 - 1)
    row = {
        "station_id": station, "segment_start": seg.timestamp.min().isoformat(),
        "segment_end": seg.timestamp.max().isoformat(), "n_daily_observations": len(y),
        "acf_lag1": float(acf(y, nlags=max_acf_lag, fft=False)[1]),
        "acf_lag7": float(acf(y, nlags=max_acf_lag, fft=False)[7]) if max_acf_lag >= 7 else np.nan,
        "pacf_lag1": float(pacf(y, nlags=max_pacf_lag, method="ywm")[1]),
    }
    row["diagnostics_status"] = "acf_pacf_only_short_segment" if len(y) < 30 else "ok"
    row["interpretation_note"] = "ACF/PACF use the longest uninterrupted daily development segment. Formal tests are reported only for segments of at least 30 days; results remain sensitive to length and structural breaks."
    if len(y) >= 30:
        lb_lag = min(7, max(1, len(y) // 5))
        row["ljung_box_lag"] = lb_lag
        row["ljung_box_p"] = float(acorr_ljungbox(y, lags=[lb_lag], return_df=True).lb_pvalue.iloc[0])
        try:
            adf_result = adfuller(y, autolag="AIC", result_object=True)
            row["adf_stat"], row["adf_p"] = float(adf_result.statistic), float(adf_result.pvalue)
        except (ValueError, ZeroDivisionError):
            row["adf_stat"], row["adf_p"] = np.nan, np.nan
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                row["kpss_stat"], row["kpss_p"] = map(float, kpss(y, regression="c", nlags="auto")[:2])
            except (ValueError, ZeroDivisionError):
                row["kpss_stat"], row["kpss_p"] = np.nan, np.nan
    diag_rows.append(row)
stationarity = pd.DataFrame(diag_rows)
stationarity.to_csv(OUT / "stationarity_acf_development.csv", index=False, encoding="utf-8")

# Put development-only segment diagnostics in the teacher's station-level EDA table.
temporal_path = OUT / "Exploratory_temporal.csv"
temporal = pd.read_csv(temporal_path)
diagnostic_fields = ["acf_lag1", "acf_lag7", "pacf_lag1", "ljung_box_p", "adf_p", "kpss_p"]
stationarity_by_id = stationarity.set_index("station_id")
for field in diagnostic_fields:
    if field in stationarity_by_id:
        temporal[field] = temporal.node_id.map(stationarity_by_id[field])
temporal["notes"] = temporal.notes.fillna("").astype(str).str.strip()
temporal["notes"] = temporal.notes.map(
    lambda note: (note + " " if note else "")
    + "Time-series diagnostics use the longest uninterrupted daily development segment; see stationarity_acf_development.csv."
)
temporal.to_csv(temporal_path, index=False, encoding="utf-8")

pd.DataFrame([
    {"axis":"full_feature_set","status":"blocked","available_ablation_variant":"","reason":"Meteorology, trace gases, and AOD are excluded because availability at forecast origin is unverified."},
    {"axis":"without_AOD","status":"blocked","available_ablation_variant":"","reason":"Requires an otherwise valid full exogenous-feature model; AOD availability at forecast origin is unverified."},
    {"axis":"without_NO2_CO_O3","status":"blocked","available_ablation_variant":"","reason":"Requires an otherwise valid full exogenous-feature model; trace-gas availability at forecast origin is unverified."},
    {"axis":"without_Meteorological","status":"blocked","available_ablation_variant":"","reason":"Requires an otherwise valid full exogenous-feature model; meteorology availability at forecast origin is unverified."},
    {"axis":"without_Calendar","status":"evaluated","available_ablation_variant":"without_calendar","reason":"Evaluated against the full currently approved feature set."},
    {"axis":"without_Lag_and_Rolling","status":"evaluated","available_ablation_variant":"without_lag_and_rolling","reason":"Evaluated against the full currently approved feature set."},
]).to_csv(OUT / "feature_group_ablation_status.csv", index=False, encoding="utf-8")

# Local cases selected only from already-saved temporal development OOF predictions.
optuna_oof = OUT / "predictions" / "cnn_lstm_optuna_temporal_oof.csv"
oof = pd.read_csv(optuna_oof if optuna_oof.is_file() else OUT / "predictions" / "cnn_lstm_temporal_oof.csv")
oof["sample_id"] = pd.to_numeric(oof.sample_id, errors="raise").astype(int)
oof["absolute_error"] = (oof.y_true - oof.y_pred).abs()
band_rows=[]
for label,mask in [
    ("low_lt10",oof.y_true<10),
    ("medium_10_25",(oof.y_true>=10)&(oof.y_true<25)),
    ("high_25_50",(oof.y_true>=25)&(oof.y_true<=50)),
    ("extreme_gt50",oof.y_true>50),
]:
    g=oof.loc[mask]
    if len(g):
        residual=g.y_true-g.y_pred
        band_rows.append({"band":label,"n":len(g),"fraction":len(g)/len(oof),"mean_actual":float(g.y_true.mean()),"mean_prediction":float(g.y_pred.mean()),"rmse":float(np.sqrt(np.mean(residual**2))),"mae":float(np.mean(np.abs(residual))),"mean_bias_actual_minus_prediction":float(residual.mean()),"underprediction_rate":float((residual>0).mean())})
pd.DataFrame(band_rows).to_csv(OUT/"error_by_concentration_band_development_optuna.csv",index=False,encoding="utf-8")
pd.DataFrame([{"n":len(oof),"actual_mean":float(oof.y_true.mean()),"actual_sd":float(oof.y_true.std(ddof=1)),"actual_min":float(oof.y_true.min()),"actual_max":float(oof.y_true.max()),"prediction_mean":float(oof.y_pred.mean()),"prediction_sd":float(oof.y_pred.std(ddof=1)),"prediction_min":float(oof.y_pred.min()),"prediction_max":float(oof.y_pred.max()),"sd_ratio_prediction_to_actual":float(oof.y_pred.std(ddof=1)/oof.y_true.std(ddof=1))}]).to_csv(OUT/"prediction_variance_diagnostics_development_optuna.csv",index=False,encoding="utf-8")
case_specs = [
    ("low_lt10", oof.y_true < 10, 5.0),
    ("medium_10_25", (oof.y_true >= 10) & (oof.y_true < 25), 17.5),
    ("high_25_50", (oof.y_true >= 25) & (oof.y_true <= 50), 37.5),
    ("extreme_gt50", oof.y_true > 50, None),
]
selected = {}
for label, mask, reference in case_specs:
    cohort = oof.loc[mask]
    if cohort.empty:
        continue
    target = float(cohort.y_true.median()) if reference is None else reference
    row = cohort.loc[(cohort.y_true - target).abs().idxmin()]
    selected.setdefault(int(row.sample_id), []).append(label)
worst = oof.loc[oof.absolute_error.idxmax()]
selected.setdefault(int(worst.sample_id), []).append("largest_absolute_error")

feature_table = build_origin_features(dev, cfg)
features = list(cfg["features"].get("calendar", [])) + ["pm25_lag1"] + [
    f"pm25_roll{w}_mean" for w in cfg["features"].get("rolling_windows_days", [3, 7])
]
features = [f for f in features if f in feature_table]
feature_values = feature_table.set_index("clean_index")
actual_lookup = {
    (str(r.station_id), pd.Timestamp(r.timestamp).tz_localize(None)): float(r.pm25)
    for r in dev.itertuples()
}
case_rows = []
for sample_id, reasons in selected.items():
    prediction = oof.loc[oof.sample_id.eq(sample_id)].iloc[0]
    inputs = feature_values.loc[sample_id, features].to_dict()
    sid = str(prediction.station_id)
    target_date = pd.Timestamp(prediction.target_timestamp)
    baseline_predictions = {
        "persistence": float(feature_values.loc[sample_id, "pm25_lag1"]),
        "seasonal_naive": float(actual_lookup.get((sid, target_date - pd.Timedelta(days=7)), np.nan)),
    }
    candidates = {"CNN_LSTM_OPTUNA": float(prediction.y_pred), **baseline_predictions}
    for model_id, predicted in candidates.items():
        case_rows.append({
            "sample_id": sample_id, "case": "|".join(sorted(set(reasons))), "model_id": model_id,
            "model_kind": "learned CNN-LSTM" if model_id == "CNN_LSTM_OPTUNA" else "simple forecast baseline",
            "station_id": prediction.station_id, "origin_timestamp": prediction.origin_timestamp,
            "target_timestamp": prediction.target_timestamp, "actual": float(prediction.y_true),
            "prediction": predicted, "residual_actual_minus_predicted": float(prediction.y_true - predicted),
            "absolute_error": abs(float(prediction.y_true - predicted)),
            "input_values_json": json.dumps(inputs, sort_keys=True, default=str),
            "scope": "development temporal out-of-fold only",
            "interpretation": "Descriptive case review; recorded inputs and residuals are not causal explanations.",
        })
pd.DataFrame(case_rows).to_csv(OUT / "local_development_cases.csv", index=False, encoding="utf-8")

(OUT / "review_evidence_metadata.json").write_text(json.dumps({
    "development_cutoff_exclusive": "2024-12-08T00:00:00Z",
    "station_month_pollutant_missingness_rows": len(missingness),
    "stations_with_acf_pacf": int(sum(r.get("diagnostics_status") in {"ok", "acf_pacf_only_short_segment"} for r in diag_rows)),
    "stations_with_formal_stationarity_tests": int(sum(r.get("diagnostics_status") == "ok" for r in diag_rows)),
    "local_case_rows": len(case_rows),
    "local_case_categories_available": sorted({x for row in case_rows for x in row["case"].split("|")}),
    "future_test_used": False,
}, indent=2), encoding="utf-8")
print(json.dumps(json.loads((OUT / "review_evidence_metadata.json").read_text(encoding="utf-8")), indent=2))
