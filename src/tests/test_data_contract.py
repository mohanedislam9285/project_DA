import pandas as pd
import pytest
from pm25_project.config import validate_config
from pm25_project.data_audit import canonicalize,read_source,validate_source_manifest,sha256
from pathlib import Path

def test_config_requires_explicit_column_mapping(config):
    incomplete={**config,"data":{**config["data"],"target_column":None}}
    with pytest.raises(ValueError,match="data.target_column"): validate_config(incomplete)

def test_source_schema_mapping_is_explicit(config):
    raw=pd.DataFrame({"date":["2020-01-01"],"site":["A"],"fine_pm":[12.5],"x_obs":[1.0]})
    canonical,stats=canonicalize(raw,config)
    assert canonical.loc[0,"pm25"]==12.5
    assert canonical.loc[0,"station_id"]=="A"
    assert canonical.loc[0,"x"]==1.0
    assert stats["timestamp_parse_failures"]==0

def test_naive_timestamps_require_declared_timezone(config):
    config["data"]["timestamp_timezone"]=None
    raw=pd.DataFrame({"date":["2020-01-01"],"site":["A"],"fine_pm":[12.5]})
    with pytest.raises(ValueError,match="timezone-naive"): canonicalize(raw,config)

def test_missing_source_has_clear_failure(config):
    with pytest.raises(FileNotFoundError,match="Raw source file not found"): read_source(config)

def test_audit_requires_verified_license_provenance_and_checksum(config,tmp_path):
    p=tmp_path/"data/raw/source.csv"; p.parent.mkdir(parents=True); p.write_text("source",encoding="utf-8")
    meta={"file":p.name,"sha256":sha256(p)}
    manifest=tmp_path/"data/manifest.yaml"; manifest.parent.mkdir(exist_ok=True); manifest.write_text("status: awaiting_source_files\n",encoding="utf-8")
    with pytest.raises(ValueError,match="provenance is incomplete"): validate_source_manifest(config,meta)
    manifest.write_text(f"source_url: https://example.test/data\ntitle: Verified dataset\nretrieval_date: '2026-11-09'\nlicense: CC-BY\nfiles:\n  - file: source.csv\n    sha256: {meta['sha256']}\n",encoding="utf-8")
    result=validate_source_manifest(config,meta)
    assert result["license"]=="CC-BY"
