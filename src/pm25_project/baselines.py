from __future__ import annotations
import numpy as np
import pandas as pd
from .metrics import regression_metrics

def predict_baselines(train:pd.DataFrame,valid:pd.DataFrame,seasonal_period:int=7):
    """One day ahead persistence and weekly seasonal naive, using prior observations only."""
    lookup={(str(r.station_id),pd.Timestamp(r.timestamp)):float(r.pm25) for r in train.itertuples()}
    all_lookup=lookup|{(str(r.station_id),pd.Timestamp(r.timestamp)):float(r.pm25) for r in valid.itertuples()}
    outs={"persistence":[],"seasonal_naive":[]}
    for r in valid.itertuples():
        station=str(r.station_id); t=pd.Timestamp(r.timestamp); prev=all_lookup.get((station,t-pd.Timedelta(days=1)),np.nan); seas=all_lookup.get((station,t-pd.Timedelta(days=seasonal_period)),np.nan)
        outs["persistence"].append(prev); outs["seasonal_naive"].append(seas)
    return {k:np.asarray(v,dtype=float) for k,v in outs.items()}

def predict_origin_baselines(observations:pd.DataFrame,keys:pd.DataFrame,seasonal_period_days:int=7):
    """Predict sequence targets from observations at their origin and prior seasonal dates."""
    if seasonal_period_days<1: raise ValueError("seasonal_period_days must be positive")

    def as_utc_naive(value):
        timestamp=pd.Timestamp(value)
        return timestamp.tz_convert("UTC").tz_localize(None) if timestamp.tz is not None else timestamp

    lookup={(str(row.station_id),as_utc_naive(row.timestamp)):float(row.pm25) for row in observations.itertuples()}
    persistence=[]; seasonal=[]
    for key in keys.itertuples():
        station=str(key.station_id)
        origin=as_utc_naive(key.origin_timestamp)
        target=as_utc_naive(key.target_timestamp)
        persistence.append(lookup.get((station,origin),np.nan))
        seasonal.append(lookup.get((station,target-pd.Timedelta(days=seasonal_period_days)),np.nan))
    return {"persistence":np.asarray(persistence,dtype=float),"seasonal_naive":np.asarray(seasonal,dtype=float)}

def evaluate_baseline_fold(train,valid,seasonal_period=7):
    predictions=predict_baselines(train,valid,seasonal_period); rows=[]
    for model,pred in predictions.items():
        m=regression_metrics(valid.pm25.to_numpy(),pred); rows.append({"model_id":model,**m})
    return rows,predictions
