import pandas as pd
import pytest

from pm25_project.seed_sensitivity import _seed_values, summarize_seed_results


def test_seed_sensitivity_requires_three_distinct_integer_seeds():
    with pytest.raises(ValueError, match="at least three"):
        _seed_values({"_search_spaces": {"cnn_lstm_seed_sensitivity": [42, 43]}})
    with pytest.raises(ValueError, match="unique"):
        _seed_values({"_search_spaces": {"cnn_lstm_seed_sensitivity": [42, 43, 42]}})
    with pytest.raises(ValueError, match="integers"):
        _seed_values({"_search_spaces": {"cnn_lstm_seed_sensitivity": [42, 43, "44"]}})


def test_seed_summary_requires_matched_fold_sample_counts():
    rows = pd.DataFrame(
        [
            {"seed": seed, "cv_fold": fold, "metric": "rmse", "value": 8.0, "n_samples": count}
            for seed, fold, count in [(42, 1, 10), (43, 1, 10), (44, 1, 10),
                                      (42, 2, 8), (43, 2, 8), (44, 2, 8)]
        ]
    )
    summary = summarize_seed_results(rows)
    assert len(summary) == 3
    assert summary.mean_fold_value.eq(8.0).all()

    rows.loc[(rows.seed == 44) & (rows.cv_fold == 2), "n_samples"] = 7
    with pytest.raises(ValueError, match="identical validation sample counts"):
        summarize_seed_results(rows)
