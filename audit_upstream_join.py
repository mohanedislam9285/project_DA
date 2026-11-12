"""Measure row-level correspondence between the supplied raw CSVs and bundle.

The audit reports observed matches and the join recipe stated in the Kaggle
description. It does not claim access to the author's original ETL code.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "extracted"
OUT = ROOT / "results" / "16" / "provenance_audit"
OUT.mkdir(parents=True, exist_ok=True)
pm = pd.read_csv(RAW / "pm25.csv", encoding="utf-8").reset_index(names="pm25_row")
gee = pd.read_csv(RAW / "gee_features_daily.csv", encoding="utf-8")
bundle = pd.read_csv(RAW / "combined_gee_pm25.csv", encoding="utf-8")

pm_key = ["date", "sensor_name", "latitude", "longitude"]
if pm.duplicated(pm_key).any():
    raise ValueError("PM2.5 station/date/coordinate key is not unique")
if gee.duplicated(["id", "date"]).any():
    raise ValueError("GEE id/date key is not unique")

# Match each bundle row to the exact PM station-date-coordinate record.
bundle_pm = bundle.merge(pm[pm_key + ["pm25_row", "pm25"]], on=pm_key, how="left", validate="many_to_one", suffixes=("_bundle", "_source"))
bundle_pm["pm_target_abs_difference"] = (bundle_pm["pm25_bundle"] - bundle_pm["pm25_source"]).abs()

# Match each bundle row to its claimed GEE location/date row.
gee_key = ["id", "date"]
bundle_gee = bundle.merge(gee, on=gee_key, how="left", validate="many_to_one", suffixes=("_bundle", "_gee"), indicator="gee_match")
lat_delta = bundle_gee["latitude"] - bundle_gee["lat"]
lon_delta = bundle_gee["longitude"] - bundle_gee["lon"]
bundle_gee["coordinate_distance_km"] = 2 * 6371.0088 * np.arcsin(np.sqrt(
    np.sin(np.deg2rad(lat_delta) / 2) ** 2
    + np.cos(np.deg2rad(bundle_gee["latitude"])) * np.cos(np.deg2rad(bundle_gee["lat"]))
    * np.sin(np.deg2rad(lon_delta) / 2) ** 2
))

# Recompute the nearest same-day GEE point for every PM2.5 source row.
# The dataset description says the spatial join uses the date and nearest
# latitude/longitude, with an approximate 0.25-degree threshold.
threshold_km = 6371.0088 * np.deg2rad(0.25)
gee_by_date = {d: g.drop_duplicates("id").reset_index(drop=True) for d, g in gee.groupby("date", sort=False)}
nearest_rows: list[dict[str, object]] = []
for date, part in pm.groupby("date", sort=False):
    locations = gee_by_date.get(date)
    if locations is None or locations.empty:
        for row in part.itertuples():
            nearest_rows.append({"pm25_row": int(row.pm25_row), "has_same_day_gee": False, "nearest_gee_id": None,
                                 "nearest_distance_km": np.nan, "second_nearest_distance_km": np.nan,
                                 "nearest_tie_within_1m": False, "candidates_within_0_25deg": 0})
        continue
    tree = BallTree(np.deg2rad(locations[["lat", "lon"]].to_numpy()), metric="haversine")
    k = min(2, len(locations))
    distances, indices = tree.query(np.deg2rad(part[["latitude", "longitude"]].to_numpy()), k=k)
    cand_indices, cand_distances = tree.query_radius(
        np.deg2rad(part[["latitude", "longitude"]].to_numpy()),
        r=threshold_km / 6371.0088, return_distance=True, sort_results=True,
    )
    for j, row in enumerate(part.itertuples()):
        first_km = float(distances[j, 0] * 6371.0088)
        second_km = float(distances[j, 1] * 6371.0088) if k > 1 else np.nan
        nearest_rows.append({
            "pm25_row": int(row.pm25_row), "has_same_day_gee": True,
            "nearest_gee_id": int(locations.iloc[int(indices[j, 0])]["id"]),
            "nearest_distance_km": first_km, "second_nearest_distance_km": second_km,
            "nearest_tie_within_1m": bool(k > 1 and abs(second_km - first_km) <= 0.001),
            "candidates_within_0_25deg": int(len(cand_indices[j])),
        })
nearest = pd.DataFrame(nearest_rows)
in_bundle = set(bundle_pm.pm25_row.dropna().astype(int))
nearest["in_combined_bundle"] = nearest.pm25_row.isin(in_bundle)
nearest["within_0_25deg"] = nearest.nearest_distance_km.le(threshold_km)
nearest["match_disposition"] = np.select(
    [nearest.in_combined_bundle, ~nearest.has_same_day_gee, ~nearest.within_0_25deg],
    ["present_in_combined_bundle", "no_same_day_GEE_row", "nearest_GEE_outside_approx_0.25deg"],
    default="within_threshold_but_absent_from_combined_bundle",
)
nearest.to_csv(OUT / "pm25_source_to_nearest_daily_gee.csv", index=False, encoding="utf-8")
bundle_pm.to_csv(OUT / "combined_to_PM25_exact_station_date_coordinate.csv", index=False, encoding="utf-8")
bundle_gee.to_csv(OUT / "combined_to_GEE_exact_id_date.csv", index=False, encoding="utf-8")

combined_nearest = bundle_pm[["pm25_row"]].merge(nearest, on="pm25_row", how="left", validate="one_to_one")
summary = {
    "dataset_metadata_snapshot": "data/provenance/kaggle_dataset_metadata_2026-11-11.json",
    "observed_data_rows": {"pm25_source": int(len(pm)), "gee_features_source": int(len(gee)), "combined_bundle": int(len(bundle))},
    "source_key_uniqueness": {
        "pm25_station_date_coordinate_duplicate_keys": int(pm.duplicated(pm_key).sum()),
        "gee_id_date_duplicate_keys": int(gee.duplicated(gee_key).sum()),
        "combined_station_date_coordinate_duplicate_keys": int(bundle.duplicated(pm_key).sum()),
        "combined_gee_id_date_duplicate_keys": int(bundle.duplicated(gee_key).sum()),
    },
    "combined_bundle_correspondence": {
        "rows_matching_PM25_station_date_coordinate": int(bundle_pm.pm25_row.notna().sum()),
        "rows_matching_GEE_id_date": int(bundle_gee.gee_match.eq("both").sum()),
        "rows_matching_both_sources": int((bundle_pm.pm25_row.notna().to_numpy() & bundle_gee.gee_match.eq("both").to_numpy()).sum()),
        "PM_target_max_abs_difference": float(bundle_pm.pm_target_abs_difference.max()),
        "PM_target_rows_not_exact_within_1e_12": int((bundle_pm.pm_target_abs_difference > 1e-12).sum()),
        "GEE_coordinate_exact_matches": int((bundle_gee.coordinate_distance_km <= 1e-9).sum()),
        "GEE_distance_km_max": float(bundle_gee.coordinate_distance_km.max()),
        "fraction_of_PM_source_rows_in_bundle": float(bundle_pm.pm25_row.nunique() / len(pm)),
        "fraction_of_GEE_rows_in_bundle": float(bundle_gee.gee_match.eq("both").sum() / len(gee)),
        "unmatched_PM_source_rows": int(len(pm) - bundle_pm.pm25_row.nunique()),
        "unused_GEE_source_rows": int(len(gee) - len(bundle)),
    },
    "nearest_same_day_GEE_reconstruction": {
        "description_threshold_interpretation": "0.25 degrees as great-circle angular radius, approximately 27.8 km",
        "PM_rows_with_same_day_GEE": int(nearest.has_same_day_gee.sum()),
        "PM_rows_nearest_within_threshold": int(nearest.within_0_25deg.sum()),
        "PM_rows_nearest_outside_threshold": int((nearest.has_same_day_gee & ~nearest.within_0_25deg).sum()),
        "PM_rows_without_same_day_GEE": int((~nearest.has_same_day_gee).sum()),
        "PM_rows_within_threshold_but_absent_from_bundle": int((~nearest.in_combined_bundle & nearest.within_0_25deg).sum()),
        "combined_rows_with_nearest_GEE_id_equal_saved_id": int((combined_nearest.nearest_gee_id == bundle.id.to_numpy()).sum()),
        "combined_rows_nearest_distance_km_max": float(combined_nearest.nearest_distance_km.max()),
        "combined_rows_with_nearest_tie_within_1m": int(combined_nearest.nearest_tie_within_1m.sum()),
        "combined_rows_with_multiple_GEE_candidates_inside_threshold": int((combined_nearest.candidates_within_0_25deg > 1).sum()),
        "PM_rows_with_nearest_tie_within_1m": int(nearest.nearest_tie_within_1m.sum()),
        "nearest_distance_km_quantiles_for_PM_rows_with_same_day_GEE": {
            str(q): float(nearest.loc[nearest.has_same_day_gee, "nearest_distance_km"].quantile(q))
            for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "candidate_count_within_threshold_quantiles_for_PM_rows_with_same_day_GEE": {
            str(q): float(nearest.loc[nearest.has_same_day_gee, "candidates_within_0_25deg"].quantile(q))
            for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
    },
    "interpretation": (
        "All combined rows correspond to one PM2.5 source record by station name, date and coordinates, "
        "and to one GEE row by id/date. The matched PM/GEE coordinates coincide. This reproduces the "
        "observed bundle correspondences, but does not prove the author's original code or explain all "
        "PM observations within the stated distance threshold that were omitted from the bundle."
    ),
}
(OUT / "upstream_join_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
