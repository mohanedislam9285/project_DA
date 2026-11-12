from __future__ import annotations
from pathlib import Path
import hashlib, json
import yaml
import numpy as np
import pandas as pd
from .config import resolve_path, validate_config

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def inspect_source_file(cfg:dict)->dict:
    """Read a small sample to display headers and a hash for explicit mapping/provenance."""
    p=resolve_path(cfg,cfg["data"]["raw_file"]); fmt=cfg["data"].get("file_format",p.suffix.lstrip(".")).lower()
    if not p.is_file(): raise FileNotFoundError(f"Raw source file not found: {p}")
    if fmt=="csv": sample=pd.read_csv(p,nrows=5)
    elif fmt in {"parquet","pq"}: sample=pd.read_parquet(p).head(5)
    elif fmt in {"xlsx","xls"}: sample=pd.read_excel(p,nrows=5)
    else: raise ValueError(f"Unsupported source format: {fmt}")
    return {"file":p.name,"bytes":p.stat().st_size,"sha256":sha256(p),"columns":list(sample.columns),"sample_dtypes":{str(k):str(v) for k,v in sample.dtypes.items()},"sample_rows":len(sample),"provenance":"Confirm source and license independently; source-info does not verify either."}

def read_source(cfg: dict) -> tuple[pd.DataFrame, dict]:
    validate_config(cfg)
    p = resolve_path(cfg, cfg["data"]["raw_file"])
    if not p.is_file(): raise FileNotFoundError(f"Raw source file not found: {p}. Add the original dataset and update configs/project.yaml")
    fmt = cfg["data"].get("file_format", p.suffix.lstrip(".")).lower()
    if fmt == "csv": raw = pd.read_csv(p)
    elif fmt in {"parquet", "pq"}: raw = pd.read_parquet(p)
    elif fmt in {"xlsx", "xls"}: raw = pd.read_excel(p)
    else: raise ValueError(f"Unsupported source format {fmt!r}; convert only with a documented lossless procedure")
    d = cfg["data"]
    station_columns=d.get("station_id_columns") or ([d["station_column"]] if d.get("station_column") else [])
    needed = [d["date_column"], d["target_column"]]+station_columns
    missing = [x for x in needed if x not in raw]
    missing += [v["column"] for v in d.get("predictors", {}).values() if v.get("column") not in raw]
    missing += [v for v in d.get("metadata_columns", {}).values() if v not in raw]
    if missing: raise ValueError(f"Configured source columns absent: {sorted(set(missing))}; correct explicit mappings")
    meta = {"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p), "rows": len(raw), "columns": list(raw.columns)}
    return raw, meta

def validate_source_manifest(cfg:dict,source_meta:dict)->dict:
    path=Path(cfg["_config_dir"])/"data/manifest.yaml"
    if not path.exists(): raise ValueError("data/manifest.yaml is required; document the verified source URL/title, retrieval date, license, and source checksum before analysis")
    manifest=yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required=("source_url","title","retrieval_date","license")
    missing=[key for key in required if not manifest.get(key)]
    if missing: raise ValueError(f"Dataset provenance is incomplete in data/manifest.yaml: {', '.join(missing)}")
    files=manifest.get("files",[]); entry=next((item for item in files if item.get("file")==source_meta["file"]),None)
    if not entry or not entry.get("sha256"): raise ValueError(f"Record filename and SHA-256 for {source_meta['file']} in data/manifest.yaml")
    if entry["sha256"].lower()!=source_meta["sha256"].lower(): raise ValueError("Source SHA-256 differs from data/manifest.yaml; preserve raw files and resolve the provenance mismatch")
    return manifest

def canonicalize(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    d = cfg["data"]
    frame = pd.DataFrame(index=raw.index)
    parsed = pd.to_datetime(raw[d["date_column"]], format=d.get("date_format"), errors="coerce")
    if parsed.dtype == object:
        raise ValueError("Mixed timestamp timezone formats detected. Normalize the source with a documented timezone rule before mapping it.")
    if getattr(parsed.dt,"tz",None) is None:
        source_tz=d.get("timestamp_timezone")
        if not source_tz: raise ValueError("Source timestamps are timezone-naive. Verify their convention and set data.timestamp_timezone explicitly.")
        frame["timestamp"]=parsed.dt.tz_localize(source_tz,ambiguous="NaT",nonexistent="NaT").dt.tz_convert("UTC")
    else:
        frame["timestamp"]=parsed.dt.tz_convert("UTC")
    frame["pm25"] = pd.to_numeric(raw[d["target_column"]], errors="coerce")
    mapping = {"latitude": d.get("latitude_column"), "longitude": d.get("longitude_column")}
    for name, spec in d.get("predictors", {}).items():
        frame[name] = pd.to_numeric(raw[spec["column"]], errors="coerce")
    for name,column in d.get("metadata_columns",{}).items():
        if column not in raw: raise ValueError(f"Configured metadata column {column!r} not found")
        frame[name]=raw[column]
    for name, col in mapping.items():
        if col:
            if col not in raw: raise ValueError(f"Configured {name} column {col!r} not found")
            frame[name] = pd.to_numeric(raw[col], errors="coerce")
    station_columns=d.get("station_id_columns") or ([d["station_column"]] if d.get("station_column") else [])
    if len(station_columns)>1:
        parts=[]
        for col in station_columns:
            values=raw[col].astype("string")
            if col in {d.get("latitude_column"),d.get("longitude_column")}:
                values=pd.to_numeric(raw[col],errors="coerce").map(lambda v: format(v,".6f") if pd.notna(v) else "NA").astype("string")
            parts.append(values.fillna("NA"))
        frame["station_id"]=parts[0]
        for part in parts[1:]: frame["station_id"]=frame["station_id"]+"::"+part
    elif station_columns: frame["station_id"]=raw[station_columns[0]].astype("string")
    else: frame["station_id"]="single_series"
    frame["source_row"] = np.arange(len(frame), dtype=np.int64)
    frame["date"] = frame["timestamp"].dt.floor("D")
    audit = {"timestamp_parse_failures": int(frame.timestamp.isna().sum()), "target_parse_failures": int(frame.pm25.isna().sum()), "n_rows": len(frame)}
    if "datetime_utc" in d.get("metadata_columns",{}):
        datetime_col=d["metadata_columns"]["datetime_utc"]; parsed_datetime=pd.to_datetime(raw[datetime_col],errors="coerce",utc=True)
        audit["datetime_utc_parse_failures"]=int(parsed_datetime.isna().sum())
        audit["date_vs_datetime_utc_mismatch_rows"]=int((parsed_datetime.dt.floor("D")!=frame.timestamp.dt.floor("D")).fillna(False).sum())
    if "sensor_name" in d.get("metadata_columns",{}):
        names=raw[d["metadata_columns"]["sensor_name"]].astype("string")
        audit["sensor_names_with_replacement_character"]=int(names.str.contains("\ufffd",regex=False,na=False).sum())
    return frame, audit

def station_identity_detail(canonical: pd.DataFrame) -> pd.DataFrame:
    """One row per composite station key, with the source sensor name and numeric source id(s) mapping to it."""
    empty_string=pd.Series(index=canonical.index,dtype="string"); empty_float=pd.Series(index=canonical.index,dtype=float)
    frame=pd.DataFrame({"station_id":canonical.station_id.astype("string"),"sensor_name":canonical.get("sensor_name",empty_string),
                        "source_id":canonical.get("gee_id",empty_string),"latitude":canonical.get("latitude",empty_float),"longitude":canonical.get("longitude",empty_float)})
    aggregated=frame.groupby("station_id",sort=True,dropna=False).agg(sensor_name=("sensor_name","first"),latitude=("latitude","first"),longitude=("longitude","first"),n_rows=("source_id","size"),n_source_ids=("source_id","nunique"),source_ids=("source_id",lambda s:";".join(sorted({str(v) for v in s.dropna()}))))
    return aggregated.reset_index()

def station_identity_reconciliation(canonical: pd.DataFrame) -> dict:
    """Explain how source sensor names, numeric source ids and composite station keys relate."""
    detail=station_identity_detail(canonical)
    if detail.empty: return {}
    source_ids=detail.source_ids.astype("string").str.split(";").explode().replace("",pd.NA).dropna()
    return {"composite_stations":int(len(detail)),"unique_sensor_names":int(detail.sensor_name.nunique()),"unique_source_ids":int(source_ids.nunique()),
            "sensor_names_spanning_multiple_composite_stations":int((detail.sensor_name.value_counts()>1).sum()),
            "source_ids_spanning_multiple_composite_stations":int((source_ids.value_counts()>1).sum()),
            "explanation":"Composite key = source sensor name + latitude + longitude rounded to six decimals, so rows sharing a sensor name or a numeric source id but differing in coordinates remain distinct stations."}

def clean_data(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    c = cfg["cleaning"]
    bad_date = frame.timestamp.isna()
    missing = frame.pm25.isna() & ~bad_date
    low = frame.pm25.le(c["target_min_exclusive"]) & ~bad_date & ~missing
    high = frame.pm25.ge(c["target_max_exclusive"]) & ~bad_date & ~missing
    reasons = {"invalid_timestamp": int(bad_date.sum()), "missing_target": int(missing.sum()), "target_le_min": int(low.sum()), "target_ge_max": int(high.sum())}
    valid = ~(bad_date | missing | low | high)
    out = frame.loc[valid].copy()
    out["timestamp"] = out["timestamp"].dt.floor("D")
    exact_columns=[col for col in out.columns if col not in {"source_row","clean_index"}]
    exact_dups = int(out.duplicated(subset=exact_columns).sum())
    key = ["station_id", "timestamp"]
    key_dups = int(out.duplicated(key, keep=False).sum())
    sensor_name_day_dups=int(out.duplicated(["sensor_name","timestamp"],keep=False).sum()) if "sensor_name" in out else 0
    policy = c.get("duplicate_policy", "error")
    if policy == "error" and key_dups:
        raise ValueError(f"{key_dups} rows belong to duplicate station-days. Review source and set a justified duplicate policy; no automatic aggregation applied")
    duplicate_rows_removed=0
    if policy == "keep_first":
        before=len(out); out = out.sort_values("source_row").drop_duplicates(key, keep="first"); duplicate_rows_removed=before-len(out)
    elif policy not in {"error", "keep_first"}: raise ValueError(f"Unsupported duplicate policy {policy!r}")
    out = out.sort_values(["station_id", "timestamp", "source_row"], kind="stable").reset_index(drop=True)
    out["clean_index"] = np.arange(len(out), dtype=np.int64)
    reasons["duplicate_station_day_dropped"]=int(duplicate_rows_removed)
    report = {"input_rows": len(frame), "clean_rows": len(out), "removed_by_reason_mutually_exclusive": reasons, "removed_total": int(len(frame)-len(out)), "exact_duplicates_before_duplicate_policy": exact_dups, "duplicate_station_day_rows": key_dups, "same_sensor_name_day_rows_separated_by_coordinate": sensor_name_day_dups, "station_identity_note":"Station key combines source sensor name with latitude and longitude rounded to six decimals so distinct nearby sites are not merged.", "duplicate_policy": policy}
    return out, report

def run_audit(cfg: dict) -> tuple[pd.DataFrame, dict]:
    raw, file_meta = read_source(cfg)
    source_manifest=validate_source_manifest(cfg,file_meta)
    canonical, parse = canonicalize(raw, cfg)
    clean, cleaning = clean_data(canonical, cfg)
    outdir = Path(cfg["_config_dir"]) / "data" / "processed"
    outdir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(outdir / "cleaned_daily.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    try: clean.to_parquet(outdir / "cleaned_daily.parquet", index=False)
    except ImportError: pass
    provenance = {"source_manifest":source_manifest,"source": file_meta, "parse": parse, "cleaning": cleaning, "date_min": str(clean.timestamp.min()), "date_max": str(clean.timestamp.max()), "station_count": int(clean.station_id.nunique()), "station_identity_reconciliation": station_identity_reconciliation(canonical), "rows_per_station": {str(k): int(v) for k,v in clean.groupby("station_id").size().items()}, "data_dictionary": {"timestamp": {"source": cfg["data"]["date_column"], "unit": "UTC daily timestamp"}, "station_id": {"source": cfg["data"].get("station_column") or cfg["data"].get("station_id_columns") or "synthetic single-series ID", "unit": "identifier", "construction": "joined configured station_id_columns with coordinates rounded to six decimals" if cfg["data"].get("station_id_columns") else "direct source ID"}, "pm25": {"source": cfg["data"]["target_column"], "unit": cfg["data"]["target_unit"]}, **{k: {"source": v, "type": "metadata"} for k,v in cfg["data"].get("metadata_columns",{}).items()}, **{k: {"source": v["column"], "unit": v.get("unit"), "family": v.get("family"), "available_at_origin": bool(v.get("available_at_origin", False))} for k,v in cfg["data"].get("predictors", {}).items()}}}
    (outdir / "audit.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return clean, provenance
