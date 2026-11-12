from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

def development_future_split(df:pd.DataFrame,fraction:float=.2,min_test_days:int=30)->dict:
    """Reserve the latest contiguous dates before constructing any development folds."""
    if not 0<fraction<.5: raise ValueError("future_test_fraction must be in (0, 0.5)")
    date_series=pd.to_datetime(df.timestamp,utc=True).dt.tz_localize(None)
    dates=np.array(sorted(date_series.dropna().unique()),dtype="datetime64[ns]")
    n_test=max(min_test_days,int(np.ceil(len(dates)*fraction)))
    if len(dates)-n_test<8: raise ValueError("Insufficient dates for an independent future holdout plus development folds")
    cut=len(dates)-n_test; dev_dates=set(dates[:cut]); test_dates=set(dates[cut:]); dt=date_series.to_numpy(dtype="datetime64[ns]")
    return {"scheme":"independent_future_holdout","status":"defined_not_evaluated","development_indices":np.flatnonzero(np.isin(dt,list(dev_dates))).tolist(),"future_test_indices":np.flatnonzero(np.isin(dt,list(test_dates))).tolist(),"development_end":str(dates[cut-1]),"future_test_start":str(dates[cut]),"future_test_end":str(dates[-1]),"n_development_dates":int(len(dev_dates)),"n_future_test_dates":int(len(test_dates)),"fraction_requested":fraction,"note":"Future test reserved before selection; do not use until model/configuration choices are locked."}

def temporal_panel_folds(df: pd.DataFrame, n_splits: int=5, embargo_days: int=1) -> dict:
    date_series=pd.to_datetime(df.timestamp,utc=True).dt.tz_localize(None)
    dates=np.array(sorted(date_series.dropna().unique()),dtype="datetime64[ns]")
    if len(dates)<n_splits+embargo_days+2: raise ValueError(f"Need at least {n_splits+embargo_days+2} unique dates for temporal CV; found {len(dates)}")
    # Expanding chronological folds with a validation block; gap excludes embargo dates before validation.
    test_size=max(1,len(dates)//(n_splits+1)); folds=[]; first=len(dates)-n_splits*test_size
    for k in range(n_splits):
        va0=first+k*test_size; va1=min(len(dates),va0+test_size); tr1=max(0,va0-embargo_days)
        if tr1==0: continue
        trdates=set(dates[:tr1]); vadates=set(dates[va0:va1])
        # Date-based assignment keeps all stations on a date together.
        dt=date_series.to_numpy(dtype="datetime64[ns]"); tr=np.flatnonzero(np.isin(dt, list(trdates))); va=np.flatnonzero(np.isin(dt,list(vadates)))
        folds.append({"fold":k+1,"train_indices":tr.tolist(),"validation_indices":va.tolist(),"train_start":str(dates[0]),"train_end":str(dates[tr1-1]),"validation_start":str(dates[va0]),"validation_end":str(dates[va1-1]),"embargo_days":embargo_days})
    return {"scheme":"temporal_panel_expanding","n_splits":len(folds),"date_unit":"UTC day","folds":folds}

def spatial_folds(df: pd.DataFrame,n_splits:int=5,seed:int=42)->dict:
    groups=df.station_id.astype(str).to_numpy(); unique=np.unique(groups)
    if len(unique)<n_splits: return {"scheme":"spatial_group_kfold","status":"not_applicable","reason":f"Need {n_splits} stations; found {len(unique)}"}
    folds=[]
    for i,(tr,va) in enumerate(GroupKFold(n_splits=n_splits).split(df,groups=groups),1): folds.append({"fold":i,"train_indices":tr.tolist(),"validation_indices":va.tolist(),"train_groups":sorted(np.unique(groups[tr]).tolist()),"validation_groups":sorted(np.unique(groups[va]).tolist())})
    return {"scheme":"spatial_group_kfold","n_splits":n_splits,"folds":folds,"seed":seed}

def random_diagnostic_folds(df:pd.DataFrame,n_splits:int=5,seed:int=42)->dict:
    folds=[]
    for i,(tr,va) in enumerate(KFold(n_splits=n_splits,shuffle=True,random_state=seed).split(df),1): folds.append({"fold":i,"train_indices":tr.tolist(),"validation_indices":va.tolist()})
    return {"scheme":"random_diagnostic","cv_purpose":"optimism diagnostic only; never model selection/generalization evidence","n_splits":n_splits,"seed":seed,"folds":folds}

def save_folds(df,cfg):
    root=Path(cfg["_config_dir"])/cfg["paths"]["results_root"]/cfg["project"]["group_id"]; root.mkdir(parents=True,exist_ok=True)
    future_path=root/"future_test.json"
    if future_path.exists():
        # An existing reservation/evaluation record is immutable. Reuse its partition
        # when regenerating development folds; never rewrite historical holdout metadata.
        future=json.loads(future_path.read_text(encoding="utf-8"))
        if not future.get("development_indices") or not future.get("future_test_indices"):
            raise ValueError("Existing future_test.json lacks saved partition indices; refusing to replace locked metadata")
        future["status"]=future.get("status","existing_locked_partition")
    else:
        future=development_future_split(df,cfg["project"].get("future_test_fraction",.2),cfg["project"].get("future_test_min_days",30))
    dev=df.iloc[future["development_indices"]].reset_index(drop=True)
    temporal=temporal_panel_folds(dev,cfg["project"]["temporal_folds"],cfg["project"]["embargo_days"])
    # Map fold positions back to the original cleaned table for traceable shared indices.
    original=np.asarray(future["development_indices"])
    for fold in temporal["folds"]:
        fold["train_indices"]=original[np.asarray(fold["train_indices"],dtype=int)].tolist()
        fold["validation_indices"]=original[np.asarray(fold["validation_indices"],dtype=int)].tolist()
    temporal["future_holdout_file"]="future_test.json"; temporal["development_end"]=future["development_end"]
    spatial=spatial_folds(dev,cfg["project"]["temporal_folds"],cfg["project"]["seed"])
    random=random_diagnostic_folds(dev,cfg["project"]["temporal_folds"],cfg["project"]["seed"])
    for scheme in (spatial,random):
        if "folds" in scheme:
            for fold in scheme["folds"]:
                fold["train_indices"]=original[np.asarray(fold["train_indices"],dtype=int)].tolist(); fold["validation_indices"]=original[np.asarray(fold["validation_indices"],dtype=int)].tolist()
    definitions={"folds_temporal.json":temporal,"folds_spatial.json":spatial,"folds_standard.json":random,"future_test.json":future}
    for name,data in definitions.items():
        if name=="future_test.json" and future_path.exists():
            continue
        (root/name).write_text(json.dumps(data,indent=2),encoding="utf-8")
    return definitions
