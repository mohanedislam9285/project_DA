import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def panel():
    rows=[]
    for station in ["A","B"]:
        for i,date in enumerate(pd.date_range("2020-01-01",periods=80)):
            rows.append({"timestamp":date,"station_id":station,"pm25":float(5+i%20),"x":float(i),"source_row":len(rows)})
    return pd.DataFrame(rows)

@pytest.fixture
def config(tmp_path):
    return {"_config_dir":str(tmp_path),"project":{"group_id":"TEST-00","target_unit":"ug/m3","temporal_folds":5,"embargo_days":1,"seed":42,"primary_horizon_days":1,"seasons_naive_period_days":7},"data":{"raw_file":"data/raw/missing.csv","file_format":"csv","timestamp_timezone":"UTC","date_column":"date","station_column":"site","target_column":"fine_pm","target_unit":"ug/m3","predictors":{"x":{"column":"x_obs","unit":"unit","family":"AOD","available_at_origin":True}}},"cleaning":{"target_min_exclusive":0,"target_max_exclusive":500,"duplicate_policy":"error","leakage_columns":[]},"features":{"predictors":[],"calendar":["day_of_week_sin","day_of_week_cos","day_of_year_sin","day_of_year_cos"],"include_target_lags":True,"include_past_rolling":True,"rolling_windows_days":[3,7],"disallow_future_exogenous_values":True},"paths":{"results_root":"results"}}
