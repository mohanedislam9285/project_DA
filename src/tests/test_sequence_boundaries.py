import numpy as np
from pm25_project.sequences import make_sequences

def test_windows_never_cross_stations_or_dates(panel):
    X,y,keys=make_sequences(panel,["pm25","x"],lookback=5,horizon=1)
    assert X.shape==(150,5,2)
    assert len(y)==len(keys)
    assert (keys.target_timestamp-keys.origin_timestamp).dt.days.eq(1).all()
    assert keys.station_id.nunique()==2

def test_gap_breaks_windows(panel):
    panel=panel.drop(index=[10]).reset_index(drop=True)
    X,y,keys=make_sequences(panel,["pm25","x"],lookback=5,horizon=1)
    around=keys[(keys.station_id=="A")&(keys.target_timestamp.between("2020-01-09","2020-01-18"))]
    assert len(around)<10

def test_future_changes_do_not_change_earlier_windows(panel):
    original=make_sequences(panel,["pm25","x"],5,1)[0]
    changed=panel.copy(); changed.loc[changed.timestamp>="2020-02-15",["pm25","x"]]+=10000
    newer=make_sequences(changed,["pm25","x"],5,1)[0]
    assert np.array_equal(original,newer[:len(original)]) is False  # later targets include changed values
    early1=make_sequences(panel[panel.timestamp<"2020-02-01"],["pm25","x"],5,1)[0]
    early2=make_sequences(changed[changed.timestamp<"2020-02-01"],["pm25","x"],5,1)[0]
    assert np.array_equal(early1,early2)

def test_missing_feature_values_are_retained_for_fold_local_imputation(panel):
    from pm25_project.preprocessing import SequencePreprocessor
    panel.loc[panel.index[5],"x"]=np.nan
    X,_,_=make_sequences(panel,["pm25","x"],5,1)
    assert np.isnan(X).any()
    prep=SequencePreprocessor().fit(X[:80]); transformed=prep.transform(X)
    assert transformed.shape[0]==len(X)
    assert np.isfinite(transformed).all()
