"""Descriptive lead-lag correlations using development data only.

For each station's longest uninterrupted development segment, correlate
available pollutant observations at t + lag with PM2.5 at t. Results are
associational diagnostics, not causal effects or evidence that a feature is
available at forecast issue time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "16"
df = pd.read_csv(ROOT / "data" / "processed" / "cleaned_daily.csv", parse_dates=["timestamp"])
df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
partition = json.loads((OUT / "future_test.json").read_text(encoding="utf-8"))
dev = df.iloc[np.asarray(partition["development_indices"], dtype=int)].copy()
cutoff = pd.to_datetime(partition["development_end"], utc=True)
assert (dev.timestamp <= cutoff).all()

pollutants = [c for c in ("no2", "co", "o3") if c in dev.columns]
lags = range(-7, 8)
rows: list[dict[str, object]] = []
for station, group in dev.sort_values("timestamp").groupby("station_id", sort=True):
    group = group.sort_values("timestamp").copy()
    group["run_id"] = group.timestamp.diff().dt.days.ne(1).cumsum()
    runs = [g for _, g in group.groupby("run_id", sort=True) if len(g) >= 30]
    if not runs:
        for pollutant in pollutants:
            for lag in lags:
                rows.append({"station_id": station, "pollutant": pollutant.upper(), "lag_days": lag,
                             "n_pairs": 0, "pearson_r": np.nan,
                             "status": "no_uninterrupted_segment_of_30_days"})
        continue
    segment = max(runs, key=len)
    target = segment.pm25.to_numpy(dtype=float)
    for pollutant in pollutants:
        values = segment[pollutant].to_numpy(dtype=float)
        for lag in lags:
            if lag < 0:
                x, y = values[:lag], target[-lag:]
            elif lag > 0:
                x, y = values[lag:], target[:-lag]
            else:
                x, y = values, target
            valid = np.isfinite(x) & np.isfinite(y)
            n = int(valid.sum())
            r = float(np.corrcoef(x[valid], y[valid])[0, 1]) if n >= 3 and np.std(x[valid]) and np.std(y[valid]) else np.nan
            rows.append({"station_id": station, "pollutant": pollutant.upper(), "segment_start": segment.timestamp.min().isoformat(),
                         "segment_end": segment.timestamp.max().isoformat(), "segment_n_days": len(segment),
                         "lag_days": lag, "n_pairs": n, "pearson_r": r,
                         "status": "ok" if n >= 30 and np.isfinite(r) else "insufficient_valid_pairs_or_variation"})

result = pd.DataFrame(rows)
path = OUT / "lead_lag_correlations_development.csv"
result.to_csv(path, index=False, encoding="utf-8")
summary = (result[result.status.eq("ok")].groupby(["pollutant", "lag_days"], as_index=False)
           .agg(stations=("station_id", "nunique"), median_station_r=("pearson_r", "median"),
                mean_station_r=("pearson_r", "mean"), total_pairs=("n_pairs", "sum")))
summary["interpretation"] = "Descriptive station-level correlations on uninterrupted development segments; no causal or forecast-time availability claim."
summary.to_csv(OUT / "lead_lag_summary_development.csv", index=False, encoding="utf-8")
print(f"Saved {len(result)} station-lag rows and {len(summary)} pollutant-lag summaries to results/16")
