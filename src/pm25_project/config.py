from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or not {"project", "data", "cleaning"}.issubset(cfg):
        raise ValueError("Config must define project, data, and cleaning sections")
    cfg["_config_dir"] = str(path.parent.parent)
    search_path=path.parent/"search_spaces.yaml"
    cfg["_search_spaces"]=yaml.safe_load(search_path.read_text(encoding="utf-8")) if search_path.exists() else {}
    return cfg, path.parent.parent

def resolve_path(cfg: dict[str, Any], value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else Path(cfg["_config_dir"]) / p

def validate_config(cfg: dict[str, Any]) -> None:
    d = cfg["data"]
    if not d.get("raw_file") or "REPLACE_WITH_REAL_DATA" in d["raw_file"]:
        raise ValueError("Set data.raw_file to the actual source file")
    for k in ("date_column", "target_column", "target_unit"):
        if not d.get(k):
            raise ValueError(f"Set data.{k} explicitly after inspecting the source")
    if d["target_unit"] != cfg["project"].get("target_unit"):
        raise ValueError("Source target unit must match project.target_unit or be explicitly converted")
    if d.get("station_column") is None and not d.get("station_id_columns") and not d.get("single_series_confirmed",False):
        raise ValueError("Set data.station_column or confirm data.single_series_confirmed for a genuine single-series source")
    if cfg["project"].get("primary_horizon_days") != 1:
        raise ValueError("This implementation currently supports primary horizon 1 day")
