"""Regenerate only the remediation evidence requested by the review: seasonal PM2.5
variation with exceedance frequencies (W2) and source-to-station identity reconciliation (W3).

Writes new artifacts only; existing locked future-test and development artifacts are
never touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pm25_project.config import load_config
from pm25_project.data_audit import canonicalize, read_source, station_identity_detail, station_identity_reconciliation, validate_source_manifest
from pm25_project.eda import seasonal_table

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "16"
cfg, _ = load_config(ROOT / "configs" / "project.yaml")

# W2: seasonal PM2.5 levels and exceedance frequencies above 25 and 50 ug/m3, full cleaned series.
df = pd.read_csv(ROOT / "data" / "processed" / "cleaned_daily.csv", parse_dates=["timestamp"])
df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
seasonal = seasonal_table(df)
seasonal.to_csv(OUT / "seasonal_summary.csv", index=False, encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

figure_dir = OUT / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
ax[0].bar(seasonal.season, seasonal["mean"], color="#4c72b0")
ax[0].set(xlabel="Season", ylabel="PM2.5 mean (µg/m³)", title="Seasonal PM2.5 level")
ax[1].bar(seasonal.season, seasonal.pct_days_above_25, color="#dd8452", label=">25 µg/m³")
ax[1].bar(seasonal.season, seasonal.pct_days_above_50, color="#c44e52", label=">50 µg/m³")
ax[1].set(xlabel="Season", ylabel="% of days", title="Seasonal exceedance frequency")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(figure_dir / "pm25_by_season.png", dpi=150)
plt.close(fig)

# W3: reconcile source sensor names, numeric source ids and composite station keys.
raw, file_meta = read_source(cfg)
validate_source_manifest(cfg, file_meta)
canonical, _ = canonicalize(raw, cfg)
detail = station_identity_detail(canonical)
detail.to_csv(OUT / "station_identity_reconciliation.csv", index=False, encoding="utf-8")
summary = station_identity_reconciliation(canonical)
raw_stations = set(detail.station_id)
clean_stations = set(df.station_id.astype(str))
summary["composite_stations_in_cleaned_data"] = len(clean_stations)
summary["composite_stations_removed_by_target_cleaning"] = sorted(raw_stations - clean_stations)
(OUT / "station_identity_reconciliation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("seasonal_summary.csv\n", seasonal.to_string(index=False))
print("\nstation identity reconciliation\n", json.dumps(summary, indent=2))
print("\nstations sharing a source sensor name:")
multi = detail[detail.sensor_name.isin(detail.sensor_name.value_counts()[lambda s: s > 1].index)]
print(multi[["station_id", "sensor_name", "n_rows", "n_source_ids", "source_ids"]].to_string(index=False))
print(f"\nSaved {len(detail)} station rows to results/16/station_identity_reconciliation.csv")
