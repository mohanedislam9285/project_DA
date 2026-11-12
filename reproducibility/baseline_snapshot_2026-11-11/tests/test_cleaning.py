import pandas as pd
import pytest
from pm25_project.data_audit import clean_data

def test_cleaning_counts_mutually_exclusive_and_preserves_valid_extreme(config):
    f=pd.DataFrame({"timestamp":pd.to_datetime(["2020-01-01","2020-01-02",None,"2020-01-04","2020-01-05","2020-01-06"]),"station_id":["a"]*6,"pm25":[None,0,10,500,499,25],"source_row":range(6)})
    out,stats=clean_data(f,config)
    assert out.pm25.tolist()==[499,25]
    assert stats["removed_total"]==4
    assert sum(stats["removed_by_reason_mutually_exclusive"].values())==4

def test_duplicate_station_day_is_explicit_error(config):
    f=pd.DataFrame({"timestamp":pd.to_datetime(["2020-01-01"]*2),"station_id":["a"]*2,"pm25":[5,6],"source_row":[0,1]})
    with pytest.raises(ValueError,match="duplicate station-days"): clean_data(f,config)
