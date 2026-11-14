import numpy as np
from pm25_project.features import build_origin_features

def test_feature_transform_is_past_only(panel,config):
    config["features"]["rolling_windows_days"]=[3]
    a=build_origin_features(panel,config)
    changed=panel.copy(); changed.loc[changed.timestamp>="2020-02-15","pm25"]+=10000
    b=build_origin_features(changed,config)
    early=a.timestamp<"2020-02-15"
    assert np.array_equal(a.loc[early,"pm25_roll3_mean"].to_numpy(),b.loc[early,"pm25_roll3_mean"].to_numpy(),equal_nan=True)

def test_unavailable_exogenous_feature_is_rejected(panel,config):
    config["features"]["predictors"]=["x"]
    config["data"]["predictors"]["x"]["available_at_origin"]=False
    with __import__("pytest").raises(ValueError,match="not certified available at origin"):
        build_origin_features(panel,config)
