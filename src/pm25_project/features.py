from __future__ import annotations
import numpy as np
import pandas as pd

def build_origin_features(df: pd.DataFrame,cfg:dict)->pd.DataFrame:
    """Build known-at-origin daily predictors; never look forward or bridge a gap/station."""
    f=cfg["features"]; predictors=f.get("predictors",[]); specs=cfg["data"].get("predictors",{})
    unavailable=[p for p in predictors if p not in specs or not specs[p].get("available_at_origin",False)]
    if f.get("disallow_future_exogenous_values",True) and unavailable: raise ValueError(f"Predictors not certified available at origin: {unavailable}")
    out=df.sort_values(["station_id","timestamp"]).copy(); dt=out.timestamp.dt
    if "day_of_week_sin" in f.get("calendar",[]): out["day_of_week_sin"]=np.sin(2*np.pi*dt.dayofweek/7); out["day_of_week_cos"]=np.cos(2*np.pi*dt.dayofweek/7)
    if "day_of_year_sin" in f.get("calendar",[]): out["day_of_year_sin"]=np.sin(2*np.pi*(dt.dayofyear-1)/365.25); out["day_of_year_cos"]=np.cos(2*np.pi*(dt.dayofyear-1)/365.25)
    if f.get("include_target_lags",True) or f.get("include_past_rolling",True):
        for st,idx in out.groupby("station_id",sort=False).groups.items():
            g=out.loc[idx].sort_values("timestamp"); y=g.pm25; segment=g.timestamp.diff().dt.days.ne(1).cumsum()
            if f.get("include_target_lags",True): out.loc[g.index,"pm25_lag1"]=y  # value observed at origin t; label is t+1
            if f.get("include_past_rolling",True):
                for w in f.get("rolling_windows_days",[3,7]): out.loc[g.index,f"pm25_roll{w}_mean"]=y.groupby(segment).transform(lambda s:s.rolling(w,min_periods=w).mean())
    return out
