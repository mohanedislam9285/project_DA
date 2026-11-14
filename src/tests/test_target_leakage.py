"""Explicit leakage checks for lag features, rolling features and sequence construction."""
import numpy as np
import pandas as pd
from pm25_project.features import build_origin_features
from pm25_project.sequences import make_sequences
from pm25_project.preprocessing import SequencePreprocessor

CUTOFF = "2020-02-15"


def test_lag1_is_the_origin_day_observation_not_a_future_value(panel, config):
    features = build_origin_features(panel, config)
    observed = panel[(panel.station_id == "A") & (panel.timestamp == "2020-01-10")].iloc[0]
    row = features[(features.station_id == "A") & (features.timestamp == "2020-01-10")].iloc[0]
    assert row.pm25_lag1 == observed.pm25
    next_day = panel[(panel.station_id == "A") & (panel.timestamp == "2020-01-11")].iloc[0]
    assert row.pm25_lag1 != next_day.pm25


def test_mutating_future_pm25_does_not_change_earlier_lag_or_rolling_features(panel, config):
    config["features"]["rolling_windows_days"] = [3]
    original = build_origin_features(panel, config)
    changed = panel.copy()
    changed.loc[changed.timestamp >= CUTOFF, "pm25"] += 10000.0
    mutated = build_origin_features(changed, config)
    for column in ("pm25_lag1", "pm25_roll3_mean"):
        early_original = original.loc[original.timestamp < CUTOFF, column].to_numpy()
        early_mutated = mutated.loc[mutated.timestamp < CUTOFF, column].to_numpy()
        assert np.array_equal(early_original, early_mutated, equal_nan=True)


def test_rolling_mean_equals_manual_past_only_computation(panel, config):
    config["features"]["rolling_windows_days"] = [3]
    features = build_origin_features(panel, config)
    for station, group in panel.sort_values(["station_id", "timestamp"]).groupby("station_id"):
        expected = group.pm25.rolling(3, min_periods=3).mean().to_numpy()
        got = features.loc[features.station_id == station].sort_values("timestamp")["pm25_roll3_mean"].to_numpy()
        assert np.allclose(got, expected, equal_nan=True)


def test_rolling_window_resets_across_a_time_gap(panel, config):
    config["features"]["rolling_windows_days"] = [3]
    gapped = panel.drop(index=[10]).reset_index(drop=True)
    features = build_origin_features(gapped, config).sort_values(["station_id", "timestamp"])
    station_a = features.loc[features.station_id == "A"].reset_index(drop=True)
    assert np.isnan(station_a.loc[0, "pm25_roll3_mean"])
    assert np.isnan(station_a.loc[1, "pm25_roll3_mean"])
    assert np.isfinite(station_a.loc[2, "pm25_roll3_mean"])
    after_gap = station_a.loc[station_a.timestamp == "2020-01-13"].iloc[0]
    assert np.isnan(after_gap.pm25_roll3_mean)


def test_sequence_window_ends_at_the_origin_and_label_is_one_day_ahead(panel):
    X, y, keys = make_sequences(panel, ["pm25"], lookback=5, horizon=1)
    lookup = {(str(r.station_id), r.timestamp): float(r.pm25) for r in panel.itertuples()}
    for i in range(len(keys)):
        row = keys.iloc[i]
        assert X[i, -1, 0] == lookup[(row.station_id, row.origin_timestamp)]
        assert y[i] == lookup[(row.station_id, row.target_timestamp)]
        assert (row.target_timestamp - row.origin_timestamp).days == 1


def test_mutating_future_rows_leaves_earlier_sequence_inputs_identical(panel):
    original_x, _, original_keys = make_sequences(panel, ["pm25"], lookback=5, horizon=1)
    changed = panel.copy()
    changed.loc[changed.timestamp >= CUTOFF, "pm25"] += 10000.0
    mutated_x, _, mutated_keys = make_sequences(changed, ["pm25"], lookback=5, horizon=1)
    cutoff = pd.Timestamp(CUTOFF)
    keep_original = (original_keys.target_timestamp < cutoff).to_numpy()
    keep_mutated = (mutated_keys.target_timestamp < cutoff).to_numpy()
    assert keep_original.sum() == keep_mutated.sum()
    assert np.array_equal(original_x[keep_original], mutated_x[keep_mutated])


def test_sequence_inputs_never_span_stations_or_extend_past_the_label(panel):
    X, y, keys = make_sequences(panel, ["pm25"], lookback=5, horizon=1)
    assert keys.station_id.nunique() == panel.station_id.nunique()
    assert np.isfinite(X).all() and np.isfinite(y).all()
    lookup = {(str(r.station_id), r.timestamp): float(r.pm25) for r in panel.itertuples()}
    future_rows = {(row.station_id, row.origin_timestamp + pd.Timedelta(days=1)) for row in keys.itertuples()}
    assert future_rows == {(row.station_id, row.target_timestamp) for row in keys.itertuples()}
    assert all(key in lookup for key in future_rows)


def test_preprocessor_statistics_are_fitted_on_the_training_fold_only(panel):
    X, _, _ = make_sequences(panel, ["pm25"], lookback=5, horizon=1)
    train = X[:80]
    preprocessor = SequencePreprocessor().fit(train)
    fitted_before = preprocessor.imputer.statistics_.copy()
    assert np.allclose(fitted_before, np.median(train.reshape(-1, 1), axis=0))
    preprocessor.transform(X)
    assert np.array_equal(preprocessor.imputer.statistics_, fitted_before)
