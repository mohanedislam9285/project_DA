import numpy as np
import json
from pathlib import Path
from pm25_project.splits import temporal_panel_folds,spatial_folds,random_diagnostic_folds,development_future_split,save_folds

def test_temporal_folds_are_chronological_and_date_grouped(panel):
    result=temporal_panel_folds(panel,n_splits=5,embargo_days=1)
    assert len(result["folds"])==5
    for f in result["folds"]:
        tr=panel.iloc[f["train_indices"]]; va=panel.iloc[f["validation_indices"]]
        assert tr.timestamp.max()<va.timestamp.min()
        assert tr.timestamp.max()<=va.timestamp.min()-__import__("pandas").Timedelta(days=2)
        assert set(tr.timestamp).isdisjoint(va.timestamp)

def test_spatial_groups_are_disjoint(panel):
    result=spatial_folds(panel,2)
    for f in result["folds"]: assert set(f["train_groups"]).isdisjoint(f["validation_groups"])

def test_random_is_labelled_diagnostic_only(panel):
    assert "only" in random_diagnostic_folds(panel,5)["cv_purpose"]

def test_future_holdout_is_latest_and_disjoint(panel):
    hold=development_future_split(panel,.2,10)
    dev=panel.iloc[hold["development_indices"]]; future=panel.iloc[hold["future_test_indices"]]
    assert dev.timestamp.max()<future.timestamp.min()
    assert future.timestamp.nunique()==hold["n_future_test_dates"]

def test_saved_temporal_folds_exclude_future_test(panel,config):
    config["project"].update(temporal_folds=5,embargo_days=1,future_test_fraction=.2,future_test_min_days=10)
    defs=save_folds(panel,config); future=set(defs["future_test.json"]["future_test_indices"])
    for f in defs["folds_temporal.json"]["folds"]:
        assert not future.intersection(f["train_indices"])
        assert not future.intersection(f["validation_indices"])

def test_regenerating_development_folds_does_not_rewrite_locked_future_record(panel,config,tmp_path):
    config['_config_dir']=str(tmp_path)
    out=tmp_path/'results'/config['project']['group_id']; out.mkdir(parents=True)
    hold=development_future_split(panel,.2,10); hold['status']='evaluated_once'
    marker=out/'future_test.json'; marker.write_text(json.dumps(hold,sort_keys=True),encoding='utf-8')
    before=marker.read_bytes(); save_folds(panel,config)
    assert marker.read_bytes()==before
