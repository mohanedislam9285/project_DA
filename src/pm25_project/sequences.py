from __future__ import annotations
import numpy as np
import pandas as pd

def make_sequences(df:pd.DataFrame, feature_columns:list[str], lookback:int=7,horizon:int=1,max_gap_days:int=1)->tuple[np.ndarray,np.ndarray,pd.DataFrame]:
    """Create station-contained daily windows with vectorized gap checks and fold-locally imputable NaNs."""
    if lookback<1 or horizon<1: raise ValueError("lookback and horizon must be positive")
    all_x=[]; all_y=[]; all_keys=[]; one_day=pd.Timedelta(days=1).value
    row_key="clean_index" if "clean_index" in df else "source_row" if "source_row" in df else None
    for station,g in df.sort_values(["station_id","timestamp"]).groupby("station_id",sort=True):
        g=g.reset_index(drop=True); dates=pd.to_datetime(g.timestamp,utc=True).dt.tz_localize(None); date_ns=dates.to_numpy(dtype="datetime64[ns]").astype("int64")
        values=g[feature_columns].to_numpy(dtype=float); target=g.pm25.to_numpy(dtype=float); n=len(g)
        origins=np.arange(lookback-1,n-horizon,dtype=int)
        if not len(origins): continue
        target_indices=origins+horizon; starts=origins-lookback+1
        windows=np.lib.stride_tricks.sliding_window_view(values,window_shape=lookback,axis=0).transpose(0,2,1)[origins-lookback+1]
        delta=np.diff(date_ns); bad_gap=(delta!=one_day).astype(np.int64); gap_prefix=np.concatenate(([0],np.cumsum(bad_gap)))
        history_has_gap=(gap_prefix[origins]-gap_prefix[starts])>0
        horizon_ok=(date_ns[target_indices]-date_ns[origins])==horizon*one_day
        valid=(~history_has_gap)&horizon_ok&np.isfinite(target[target_indices])&(~np.isinf(windows).any(axis=(1,2)))
        if not valid.any(): continue
        selected_origins=origins[valid]; selected_targets=target_indices[valid]; selected_windows=windows[valid]
        all_x.append(selected_windows.astype(np.float32)); all_y.append(target[selected_targets].astype(np.float32))
        target_rows=g.iloc[selected_targets][row_key].to_numpy() if row_key else selected_targets
        all_keys.append(pd.DataFrame({"station_id":str(station),"origin_timestamp":dates.iloc[selected_origins].to_numpy(),"target_timestamp":dates.iloc[selected_targets].to_numpy(),"target_row":target_rows}))
    n_features=len(feature_columns)
    X=np.concatenate(all_x,axis=0) if all_x else np.empty((0,lookback,n_features),dtype=np.float32)
    y=np.concatenate(all_y) if all_y else np.empty((0,),dtype=np.float32)
    keys=pd.concat(all_keys,ignore_index=True) if all_keys else pd.DataFrame(columns=["station_id","origin_timestamp","target_timestamp","target_row"])
    return X,y,keys
