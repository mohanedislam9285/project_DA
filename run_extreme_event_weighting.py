"""Compare ordinary and threshold-weighted CNN-LSTM fits on temporal development folds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from pm25_project.config import load_config
from pm25_project.extreme_events import extreme_event_sample_weights
from pm25_project.features import build_origin_features
from pm25_project.metrics import regression_metrics,paired_bootstrap_ci
from pm25_project.models.cnn_lstm import build_cnn_lstm
from pm25_project.preprocessing import SequencePreprocessor
from pm25_project.sequences import make_sequences


ROOT=Path(__file__).resolve().parent


def _band_mask(target:pd.Series,band:str)->pd.Series:
    if band=="all":
        return pd.Series(True,index=target.index)
    if band=="low_le25":
        return target.le(25)
    if band=="high_25_50":
        return target.gt(25)&target.le(50)
    return target.gt(50)


cfg,_=load_config(ROOT/"configs"/"project.yaml")
out=ROOT/cfg["paths"]["results_root"]/cfg["project"]["group_id"]
data=pd.read_csv(ROOT/"data"/"processed"/"cleaned_daily.csv",parse_dates=["timestamp"])
data["timestamp"]=pd.to_datetime(data.timestamp,utc=True)
partition=json.loads((out/"future_test.json").read_text(encoding="utf-8"))
dev=data.iloc[np.asarray(partition["development_indices"],dtype=int)].copy()
features=build_origin_features(dev,cfg)
columns=list(cfg["features"].get("calendar",[]))+["pm25_lag1"]+[
    f"pm25_roll{window}_mean" for window in cfg["features"].get("rolling_windows_days",[3,7])
]
columns=list(dict.fromkeys(column for column in columns if column in features))
folds=json.loads((out/"folds_temporal.json").read_text(encoding="utf-8"))["folds"]
optuna=pd.read_csv(out/"temporal_cv_optuna_results.csv")
reference=pd.read_csv(out/"predictions"/"cnn_lstm_optuna_temporal_oof.csv")
prediction_rows=[]
metric_rows=[]
for fold in folds:
    fold_id=int(fold["fold"])
    selected=optuna.loc[optuna.cv_fold.eq(fold_id)].iloc[0]
    config=json.loads(selected.configuration)
    X,y,keys=make_sequences(features,columns,lookback=int(config["lookback_days"]),horizon=1)
    train=np.flatnonzero(keys.target_row.isin(fold["train_indices"]).to_numpy())
    valid=np.flatnonzero(keys.target_row.isin(fold["validation_indices"]).to_numpy())
    reference_fold=reference.loc[reference.cv_fold.eq(fold_id)].copy()
    expected=set(pd.to_numeric(reference_fold.sample_id).astype(int))
    valid=valid[keys.iloc[valid].target_row.astype(int).isin(expected).to_numpy()]
    if set(keys.iloc[valid].target_row.astype(int))!=expected:
        raise ValueError(f"Weighted fold {fold_id} does not match the saved reference cohort")

    prep=SequencePreprocessor().fit(X[train])
    x_train=prep.transform(X[train])
    x_valid=prep.transform(X[valid])
    model_config={key:config[key] for key in ("filters","kernel_size","dropout","lstm_units","dense_units","learning_rate","loss") if key in config}
    model_config["seed"]=int(selected.seed)
    tf.keras.backend.clear_session()
    model=build_cnn_lstm(int(config["lookback_days"]),x_train.shape[-1],model_config)
    model.fit(
        x_train,y[train],
        sample_weight=extreme_event_sample_weights(y[train]),
        epochs=int(selected.best_epoch),
        batch_size=int(cfg["model"]["cnn_lstm"].get("batch_size",64)),
        verbose=0,
    )
    weighted_prediction=model.predict(x_valid,verbose=0).reshape(-1)
    weighted=pd.DataFrame({
        "sample_id":keys.iloc[valid].target_row.astype(int).to_numpy(),
        "station_id":keys.iloc[valid].station_id.to_numpy(),
        "origin_timestamp":keys.iloc[valid].origin_timestamp.astype(str).to_numpy(),
        "target_timestamp":keys.iloc[valid].target_timestamp.astype(str).to_numpy(),
        "y_true":y[valid],
        "y_pred":weighted_prediction,
        "cv_fold":fold_id,
        "model_id":"CNN_LSTM_WEIGHTED_MSE",
    })
    reference_rows=reference_fold.set_index("sample_id").loc[weighted.sample_id].reset_index()
    reference_rows=reference_rows[["sample_id","station_id","origin_timestamp","target_timestamp","y_true","y_pred"]].copy()
    if not np.allclose(reference_rows.y_true.to_numpy(dtype=float),weighted.y_true.to_numpy(dtype=float)):
        raise ValueError(f"Saved reference targets disagree in fold {fold_id}")
    reference_rows["cv_fold"]=fold_id
    reference_rows["model_id"]="CNN_LSTM_OPTUNA_REFERENCE"
    for model_id,frame in (("CNN_LSTM_OPTUNA_REFERENCE",reference_rows),("CNN_LSTM_WEIGHTED_MSE",weighted)):
        target=frame.y_true.to_numpy(dtype=float)
        prediction=frame.y_pred.to_numpy(dtype=float)
        bands={
            "all":np.ones(len(frame),dtype=bool),
            "low_le25":target<=25,
            "high_25_50":(target>25)&(target<=50),
            "extreme_gt50":target>50,
        }
        for band,mask in bands.items():
            if not mask.any():
                continue
            scores=regression_metrics(target[mask],prediction[mask])
            for metric,value in scores.items():
                if metric!="n_samples":
                    metric_rows.append({
                        "model_id":model_id,"cv_scheme":"temporal_panel_expanding","cv_fold":fold_id,
                        "concentration_band":band,"metric":metric,"value":value,
                        "n_samples":int(scores["n_samples"]),
                        "weight_rule":"1x <=25; 2x >25-50; 4x >50" if model_id=="CNN_LSTM_WEIGHTED_MSE" else "unweighted MSE reference",
                    })
    prediction_rows.extend((reference_rows,weighted))
    tf.keras.backend.clear_session()

predictions=pd.concat(prediction_rows,ignore_index=True)
metrics=pd.DataFrame(metric_rows)
predictions.to_csv(out/"predictions"/"cnn_lstm_extreme_weighting_oof.csv",index=False,encoding="utf-8")
metrics.to_csv(out/"extreme_event_weighting_folds.csv",index=False,encoding="utf-8")

summary_rows=[]
for (model_id,band,metric),group in metrics.groupby(["model_id","concentration_band","metric"],sort=True):
    values=group.value.to_numpy(dtype=float)
    interval=paired_bootstrap_ci(values,seed=42)
    model_predictions=predictions.loc[predictions.model_id.eq(model_id)].copy()
    selected=model_predictions.loc[_band_mask(model_predictions.y_true,band)]
    pooled_value=regression_metrics(selected.y_true.to_numpy(dtype=float),selected.y_pred.to_numpy(dtype=float))[metric]
    paired=metrics.loc[metrics.concentration_band.eq(band)&metrics.metric.eq(metric)].pivot(
        index="cv_fold",columns="model_id",values="value"
    ).dropna()
    delta=paired["CNN_LSTM_WEIGHTED_MSE"]-paired["CNN_LSTM_OPTUNA_REFERENCE"] if model_id=="CNN_LSTM_WEIGHTED_MSE" and {"CNN_LSTM_WEIGHTED_MSE","CNN_LSTM_OPTUNA_REFERENCE"}.issubset(paired.columns) else pd.Series(dtype=float)
    delta_interval=paired_bootstrap_ci(delta.to_numpy(dtype=float),seed=42)
    summary_rows.append({
        "model_id":model_id,"concentration_band":band,"metric":metric,
        "fold_mean":float(np.nanmean(values)),
        "fold_sd":float(np.nanstd(values,ddof=1)) if len(values)>1 else np.nan,
        "fold_bootstrap_95_ci_lower":interval[0],
        "fold_bootstrap_95_ci_upper":interval[1],
        "pooled_oof_metric":pooled_value,
        "mean_delta_vs_reference":float(delta.mean()) if len(delta) else np.nan,
        "delta_ci_lower":delta_interval[0],
        "delta_ci_upper":delta_interval[1],
        "n_folds":int(len(values)),"total_oof_samples":int(len(selected)),
    })
summary=pd.DataFrame(summary_rows)
summary.to_csv(out/"extreme_event_weighting_summary.csv",index=False,encoding="utf-8")
print(summary.to_string(index=False))
