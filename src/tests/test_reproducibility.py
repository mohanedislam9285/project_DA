from pm25_project.splits import temporal_panel_folds

def test_temporal_fold_definitions_repeat(panel):
    a=temporal_panel_folds(panel,n_splits=5,embargo_days=1)
    b=temporal_panel_folds(panel,n_splits=5,embargo_days=1)
    assert a==b
