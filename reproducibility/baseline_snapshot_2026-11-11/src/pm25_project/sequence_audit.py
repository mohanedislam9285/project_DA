"""Check lookback coverage against saved outer folds before any model tuning."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .features import build_origin_features
from .sequences import make_sequences

def sequence_coverage_table(df,cfg,lookbacks=(7,14,28,56)):
    root=Path(cfg["_config_dir"]); out=root/cfg["paths"]["results_root"]/cfg["project"]["group_id"]; fold_path=out/"folds_temporal.json"
    if not fold_path.exists(): raise FileNotFoundError("Run folds before sequence coverage analysis")
    folds=json.loads(fold_path.read_text(encoding="utf-8"))["folds"]; features=build_origin_features(df,cfg)
    cols=list(cfg["features"].get("predictors",[]))+list(cfg["features"].get("calendar",[]))
    if cfg["features"].get("include_target_lags",True): cols.append("pm25_lag1")
    if cfg["features"].get("include_past_rolling",True): cols += [f"pm25_roll{w}_mean" for w in cfg["features"].get("rolling_windows_days",[3,7])]
    cols=list(dict.fromkeys(cols)); rows=[]
    for lookback in lookbacks:
        _,_,keys=make_sequences(features,cols,int(lookback),1); rows.append({"lookback_days":int(lookback),"fold":"all","partition":"all_observations","n_sequences":len(keys),"n_stations":int(keys.station_id.nunique()),"target_start":str(keys.target_timestamp.min()) if len(keys) else None,"target_end":str(keys.target_timestamp.max()) if len(keys) else None})
        target_rows=keys.target_row.astype(int)
        for fold in folds:
            for partition,index_key in (("train","train_indices"),("validation","validation_indices")):
                selected=keys.loc[target_rows.isin(fold[index_key])]
                rows.append({"lookback_days":int(lookback),"fold":fold["fold"],"partition":partition,"n_sequences":len(selected),"n_stations":int(selected.station_id.nunique()),"target_start":str(selected.target_timestamp.min()) if len(selected) else None,"target_end":str(selected.target_timestamp.max()) if len(selected) else None})
    table=pd.DataFrame(rows); table["eligible_primary_candidate"]=table.apply(lambda r: r.partition=="all_observations" or (r.partition=="validation" and r.n_sequences>=100),axis=1)
    table.to_csv(out/"lookback_coverage.csv",index=False,encoding="utf-8"); return table
