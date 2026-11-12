"""Traceable helpers for report tables. No hand-entered findings are generated."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import paired_bootstrap_ci

def summarize_fold_scores(results:pd.DataFrame)->pd.DataFrame:
    required={"model_id","cv_scheme","metric","value","cv_fold"}
    if not required.issubset(results): raise ValueError(f"Missing result columns: {sorted(required-set(results))}")
    dev=results[(results.dataset_split=="validation")&(results.niveau=="fold")]
    if "cv_purpose" in dev: dev=dev[dev.cv_purpose.ne("training_size_robustness")]
    if "robustness_variant" in dev: dev=dev[dev.robustness_variant.isna()|dev.robustness_variant.isin(["","reference"])]
    return dev.groupby(["model_id","cv_scheme","metric"],as_index=False).agg(mean=("value","mean"),std=("value","std"),n_folds=("cv_fold","nunique"),n_samples=("n_samples","sum"))

def require_final_test_evidence(results:pd.DataFrame)->None:
    if "dataset_split" not in results or not results.dataset_split.eq("future_test").any(): raise ValueError("No independent future-test scores exist; do not state final test findings")

def _prediction_error_summary(predictions:pd.DataFrame,group_columns:list[str])->pd.DataFrame:
    required={"y_true","y_pred","residual",*group_columns}
    if not required.issubset(predictions): raise ValueError(f"Prediction data missing columns: {sorted(required-set(predictions))}")
    rows=[]
    for keys,group in predictions.groupby(group_columns,dropna=False):
        if not isinstance(keys,tuple): keys=(keys,)
        residual=group.residual.to_numpy(dtype=float); valid=np.isfinite(residual)&np.isfinite(group.y_true)&np.isfinite(group.y_pred)
        if not valid.any(): continue
        row=dict(zip(group_columns,keys))
        row.update(n_samples=int(valid.sum()),rmse=float(np.sqrt(np.mean(residual[valid]**2))),mae=float(np.mean(np.abs(residual[valid]))),mean_error=float(np.mean(residual[valid])))
        rows.append(row)
    return pd.DataFrame(rows,columns=[*group_columns,"n_samples","rmse","mae","mean_error"])

def generate_development_assets(results_dir:str|Path,seed:int=42)->dict:
    """Generate report-ready development tables/figures without touching the future holdout."""
    root=Path(results_dir); results_path=root/"results.csv"
    if not results_path.is_file(): raise FileNotFoundError(f"Development results not found: {results_path}")
    results=pd.read_csv(results_path)
    selection=results[
        results.dataset_split.eq("validation")
        & results.cv_scheme.eq("temporal_panel_expanding")
        & results.niveau.eq("fold")
        & results.cv_purpose.isin(["development_selection","development"])
    ].copy()
    if "robustness_variant" in selection:
        selection=selection[selection.robustness_variant.isna()|selection.robustness_variant.isin(["","reference"])]
    if selection.empty: raise ValueError("No eligible full-data temporal validation folds are available for report assets")

    table_rows=[]
    for (model_id,metric),group in selection.groupby(["model_id","metric"]):
        values=group.value.to_numpy(dtype=float); lower,upper=paired_bootstrap_ci(values,seed=seed)
        table_rows.append({"model_id":model_id,"metric":metric,"mean":float(np.mean(values)),"std":float(np.std(values,ddof=1)) if len(values)>1 else np.nan,"ci_lower":lower,"ci_upper":upper,"n_folds":int(group.cv_fold.nunique()),"n_samples_total":int(group.n_samples.sum())})
    tables=root/"tables"; figures=root/"figures"; tables.mkdir(parents=True,exist_ok=True); figures.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(table_rows).to_csv(tables/"development_metrics.csv",index=False,encoding="utf-8")

    temporal=selection[selection.metric.eq("rmse")].groupby("model_id").agg(mean=("value","mean"),std=("value","std")).sort_values("mean")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(max(9,len(temporal)*.55),5))
    ax.bar(temporal.index,temporal["mean"],yerr=temporal["std"],capsize=3,color="#3977a8")
    ax.set_ylabel("Temporal validation RMSE (ug/m3; mean +/- SD across folds)")
    ax.set_title("Development-fold model comparison (not future-test performance)")
    ax.tick_params(axis="x",rotation=55); fig.tight_layout(); fig.savefig(figures/"temporal_rmse_by_model.png",dpi=180); plt.close(fig)

    prediction_paths=[
        root/"predictions"/"temporal_oof.csv",
        root/"predictions"/"cnn_lstm_temporal_oof.csv",
        root/"predictions"/"lstm_temporal_oof.csv",
        root/"predictions"/"baseline_oof.csv",
    ]
    available=[pd.read_csv(path) for path in prediction_paths if path.is_file()]
    if not available: raise FileNotFoundError("No saved temporal out-of-fold predictions are available for error analysis")
    predictions=pd.concat(available,ignore_index=True)
    if "cv_scheme" not in predictions:
        raise ValueError("Saved predictions must identify their cross-validation scheme")
    predictions=predictions[
        predictions.dataset_split.eq("validation")
        & predictions.cv_scheme.eq("temporal_panel_expanding")
    ].copy()
    predictions["residual"]=predictions.y_true-predictions.y_pred
    predictions["pm25_band"]=np.select([predictions.y_true<10,predictions.y_true<=25,predictions.y_true<=50],["low (<10)","medium (10-25)","high (25-50)"],default="extreme (>50)")
    by_band=_prediction_error_summary(predictions,["model_id","pm25_band"])
    by_band.to_csv(tables/"error_by_pm25_band.csv",index=False,encoding="utf-8")
    predictions["month"]=pd.to_datetime(predictions.target_timestamp,utc=True,format="mixed").dt.month
    _prediction_error_summary(predictions,["model_id","month"]).to_csv(tables/"error_by_month.csv",index=False,encoding="utf-8")
    _prediction_error_summary(predictions,["model_id","station_id"]).to_csv(tables/"error_by_station.csv",index=False,encoding="utf-8")

    top_models=temporal.head(5).index.tolist()
    if "CNN_LSTM" in temporal.index and "CNN_LSTM" not in top_models: top_models.append("CNN_LSTM")
    band_order=["low (<10)","medium (10-25)","high (25-50)","extreme (>50)"]
    fig,ax=plt.subplots(figsize=(9,5))
    for model_id,group in by_band[by_band.model_id.isin(top_models)].groupby("model_id"):
        ordered=group.set_index("pm25_band").reindex(band_order)
        ax.plot(band_order,ordered.rmse,marker="o",label=model_id)
    ax.set_ylabel("Out-of-fold RMSE (ug/m3)"); ax.set_xlabel("Observed PM2.5 band")
    ax.set_title("Temporal out-of-fold error by concentration band")
    ax.tick_params(axis="x",rotation=15); ax.legend(ncol=2); fig.tight_layout(); fig.savefig(figures/"error_by_pm25_band.png",dpi=180); plt.close(fig)
    return {"models":int(temporal.shape[0]),"prediction_rows":int(len(predictions)),"tables":str(tables),"figures":str(figures)}
