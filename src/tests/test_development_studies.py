import numpy as np
import pytest

from pm25_project.sensitivity import adjacent_values, _target_inverse, _target_transform, run_hyperparameter_sensitivity, run_preprocessing_ablation
from pm25_project import sensitivity
from pm25_project.sequence_studies import _write_model_family_ablation
from pm25_project.splits import save_folds
from pm25_project.development_diagnostics import run_local_explanations, run_cross_scheme_ranking_stability, run_validation_scheme_ablation


def test_hyperparameter_neighbor_selection_is_one_predeclared_step():
    assert adjacent_values([1, 2, 3, 5], 3) == [2, 5]
    assert adjacent_values([1, 2, 3], 1) == [2]
    assert adjacent_values([1, 2, 3], 4) == []


def test_log_target_transform_round_trips_original_scale():
    target = np.array([0.1, 2.0, 25.0, 400.0])
    transformed = _target_transform(target, "log1p")
    np.testing.assert_allclose(_target_inverse(transformed, "log1p"), target)
    np.testing.assert_array_equal(_target_inverse(target, "raw"), target)


def test_log_target_transform_rejects_invalid_domain():
    with pytest.raises(ValueError, match="requires all targets > -1"):
        _target_transform([-1.0, 1.0], "log1p")


def test_log_target_configuration_is_selected_on_original_scale(monkeypatch):
    class Estimator:
        def set_params(self, **params):
            self.tag = params.get("tag", getattr(self, "tag", "A"))
            return self

        def fit(self, x, y):
            return self

        def predict(self, x):
            values = {
                "A": [0.6931471806, 6.408768],
                "B": [1.6931471806, 6.908755],
            }
            return np.asarray(values[self.tag][: len(x)])

    monkeypatch.setattr(sensitivity, "model_registry", lambda seed: {"Fake": Estimator})
    monkeypatch.setattr(sensitivity, "candidate_configs", lambda *args: [{"tag": "A"}, {"tag": "B"}])
    dates = np.arange("2020-01-01", "2020-01-11", dtype="datetime64[D]")
    target = np.ones(10)
    target[-2:] = [1.0, 1000.0]
    selected = sensitivity._select_original_scale_configuration(
        "Fake", np.zeros((10, 1)), target, dates, "log1p", True, {"project": {"seed": 42}}
    )
    assert selected == {"tag": "B"}




def test_sequence_model_family_ablation_requires_identical_predictions(tmp_path):
    import pandas as pd

    lstm = pd.DataFrame([
        {"sample_id": sample, "cv_fold": 1, "y_true": target, "y_pred": target + 1}
        for sample, target in [(1, 10.0), (2, 20.0)]
    ])
    cnn = pd.DataFrame([
        {"sample_id": sample, "cv_fold": 1, "y_true": target, "y_pred": target + 2, "model_id": "CNN_LSTM"}
        for sample, target in [(1, 10.0), (2, 20.0)]
    ])
    _write_model_family_ablation(tmp_path, lstm, cnn, {"project": {"seed": 42}})
    summary = pd.read_csv(tmp_path / "model_family_ablation_summary.csv")
    assert summary.n_paired_folds.item() == 1
    cnn.loc[0, "y_true"] = 11.0
    with pytest.raises(ValueError, match="disagree on observed targets"):
        _write_model_family_ablation(tmp_path, lstm, cnn, {"project": {"seed": 42}})


def test_local_cases_use_shared_development_samples_and_report_absent_bands(panel, config):
    import pandas as pd

    panel = panel.copy()
    panel["clean_index"] = np.arange(len(panel))
    out = __import__("pathlib").Path(config["_config_dir"]) / "results" / config["project"]["group_id"]
    prediction_dir = out / "predictions"
    prediction_dir.mkdir(parents=True)
    sample_ids = panel.clean_index.iloc[12:22].to_numpy()
    targets = panel.set_index("clean_index").loc[sample_ids, "pm25"].to_numpy()
    rows = []
    for model, error in [("Bagging", 1.0), ("CatBoost", 2.0), ("CNN_LSTM", 3.0), ("LSTM", 2.5)]:
        for sample_id, target in zip(sample_ids, targets):
            rows.append({
                "sample_id": sample_id,
                "station_id": "A",
                "target_timestamp": "2020-01-01",
                "y_true": float(np.float32(target)) if model in {"CNN_LSTM", "LSTM"} else target,
                "y_pred": target + error,
                "dataset_split": "validation",
                "cv_scheme": "temporal_panel_expanding",
                "cv_fold": 1,
                "model_id": model,
            })
    pd.DataFrame(rows).to_csv(prediction_dir / "temporal_oof.csv", index=False)
    result = run_local_explanations(panel, config)
    assert result["shared_samples"] == len(sample_ids)
    assert "LSTM" in result["models"]
    assert set(result["bands_absent"]) == {"high_25_50", "extreme_gt50"}
    cases = pd.read_csv(out / "local_explanation_cases.csv")
    assert "largest_error" in set(cases.case)
    local = pd.read_csv(out / "local_explanations.csv")
    assert local.feature_values_json.notna().all()


def test_cross_scheme_rank_and_validation_gap_outputs_are_unpaired(tmp_path):
    import pandas as pd

    out = tmp_path / "results" / "TEST-00"
    (out / "logs").mkdir(parents=True)
    config = {"_config_dir": str(tmp_path), "paths": {"results_root": "results"}, "project": {"group_id": "TEST-00"}}
    for scheme, ranks in [
        ("feature_importance.csv", [("a", 1), ("b", 2)]),
        ("logs/feature_importance_spatial_group_kfold.csv", [("a", 2), ("b", 1)]),
        ("logs/feature_importance_random_diagnostic.csv", [("a", 1), ("b", 2)]),
    ]:
        pd.DataFrame([{"model_id": "M", "feature": feature, "rank": rank} for feature, rank in ranks]).to_csv(out / scheme, index=False)
    stability = run_cross_scheme_ranking_stability(config)
    assert stability["comparisons"] == 3

    results = []
    for scheme, values in [
        ("temporal_panel_expanding", [8.0, 9.0]),
        ("spatial_group_kfold", [9.0, 10.0]),
        ("random_diagnostic", [7.0, 8.0]),
    ]:
        for fold, value in enumerate(values, 1):
            results.append({
                "model_id": "M", "dataset_split": "validation", "niveau": "fold", "metric": "rmse",
                "cv_scheme": scheme, "cv_fold": fold, "cv_purpose": "development_selection",
                "value": value, "n_samples": 10,
            })
    pd.DataFrame(results).to_csv(out / "results.csv", index=False)
    gaps = run_validation_scheme_ablation(config)
    assert gaps["comparisons"] == 2
    table = pd.read_csv(out / "validation_scheme_ablation.csv")
    assert table.interpretation.str.contains("Unpaired").all()
