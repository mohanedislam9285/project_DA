from pathlib import Path
import numpy as np
from pm25_project.metrics import regression_metrics
from pm25_project.baselines import predict_origin_baselines
from pm25_project.extreme_events import extreme_event_sample_weights
from pm25_project.schemas import RESULTS,PREDICTIONS,EXPLORATORY_DATA,EXPLORATORY_TEMPORAL
from pm25_project.comparisons import comparison_table
from pm25_project.summaries import add_fold_summaries
from pm25_project.ablations import feature_family_variants,require_same_samples
import pytest
import pandas as pd

def test_metrics_known_values():
    m=regression_metrics([1,2,3],[1,2,5]); assert m["rmse"]==np.sqrt(4/3); assert m["mae"]==2/3; assert m["n_samples"]==3

def test_origin_persistence_uses_last_observation_not_target():
    dates=pd.date_range("2020-01-01",periods=8,tz="UTC")
    observations=pd.DataFrame({"station_id":"A","timestamp":dates,"pm25":[2,3,4,5,6,7,8,9]})
    keys=pd.DataFrame({"station_id":["A"],"origin_timestamp":[dates[6]],"target_timestamp":[dates[7]]})
    baselines=predict_origin_baselines(observations,keys)
    assert baselines["persistence"].tolist()==[8]
    assert baselines["seasonal_naive"].tolist()==[2]

def test_extreme_event_weights_use_strict_thresholds():
    actual=extreme_event_sample_weights([25,25.01,50,50.01])
    np.testing.assert_array_equal(actual,[1,2,2,4])

def test_extreme_event_weights_reject_invalid_targets_and_weights():
    with pytest.raises(ValueError,match="targets must be finite"):
        extreme_event_sample_weights([1,np.nan])
    with pytest.raises(ValueError,match="weights must be positive"):
        extreme_event_sample_weights([1],moderate_weight=0)

def test_teacher_schemas_include_required_columns():
    assert {"model_id","variable","n_missing","mean","is_leakage_suspect"}.issubset(EXPLORATORY_DATA)
    assert {"model_id","node_id","acf_lag1","cv_strategy_recommended"}.issubset(EXPLORATORY_TEMPORAL)
    assert {"dataset_split","cv_scheme","cv_fold","metric","is_primary","value"}.issubset(RESULTS)
    assert {"station_id","origin_timestamp","target_timestamp","y_true","y_pred"}.issubset(PREDICTIONS)

def test_paired_comparisons_apply_familywise_correction():
    rows=[]
    for model,values in {"A":[1.,2.,1.5],"B":[2.,3.,2.5],"persistence":[3.,3.5,4.]}.items():
        for fold,value in enumerate(values,1): rows.append({"model_id":model,"cv_scheme":"temporal_panel_expanding","cv_fold":fold,"metric":"rmse","niveau":"fold","dataset_split":"validation","value":value})
    result=comparison_table(pd.DataFrame(rows),n_perm=200)
    assert len(result)==3
    assert result.p_value_corrected.notna().all()
    assert result.correction_method.eq("Holm-Bonferroni").all()

def test_training_size_robustness_rows_do_not_enter_paired_model_tests():
    rows=[]
    for model,values in {"A":[1.,2.,1.5],"B":[2.,3.,2.5]}.items():
        for fold,value in enumerate(values,1):
            rows.append({"model_id":model,"cv_scheme":"temporal_panel_expanding","cv_fold":fold,"metric":"rmse","niveau":"fold","dataset_split":"validation","value":value,"cv_purpose":"development_selection"})
    ordinary=comparison_table(pd.DataFrame(rows),n_perm=200)
    rows.append({"model_id":"A","cv_scheme":"temporal_panel_expanding","cv_fold":1,"metric":"rmse","niveau":"fold","dataset_split":"validation","value":0.01,"cv_purpose":"training_size_robustness","robustness_variant":"train_10%"})
    with_robustness=comparison_table(pd.DataFrame(rows),n_perm=200)
    assert with_robustness.delta.tolist()==ordinary.delta.tolist()

def test_paired_model_tests_require_equal_fold_sample_counts():
    rows=[]
    for model,(values,n_samples) in {"A":([1.,2.,1.5],100),"B":([2.,3.,2.5],100),"CNN_LSTM":([3.,4.,3.5],25)}.items():
        for fold,value in enumerate(values,1):
            rows.append({"model_id":model,"cv_scheme":"temporal_panel_expanding","cv_fold":fold,"metric":"rmse","niveau":"fold","dataset_split":"validation","value":value,"n_samples":n_samples})
    result=comparison_table(pd.DataFrame(rows),n_perm=200)
    assert result.binome_id.tolist()==["A|B"]
    assert result.n_paired_folds.tolist()==[3]

def test_fold_summary_adds_uncertainty_without_removing_fold_rows():
    frame=pd.DataFrame([{"model_id":"m","cv_scheme":"temporal","dataset_split":"validation","niveau":"fold","metric":"rmse","cv_fold":i,"value":float(i),"n_samples":10} for i in range(1,6)])
    result=add_fold_summaries(frame,n_boot=500)
    summary=result[result.niveau=="global"].iloc[0]
    assert len(result[result.niveau=="fold"])==5
    assert summary.value==3.0
    assert summary.ci_lower<summary.value<summary.ci_upper

def test_feature_ablation_variants_are_family_based_and_require_same_samples():
    variants=feature_family_variants(["pm25_lag1","temp","sin"],{"pm25_lag1":"Lag","temp":"Meteorological","sin":"Calendar"})
    assert variants["without_Lag"]==["temp","sin"]
    require_same_samples([1,2],[2,1])
    with pytest.raises(ValueError,match="identical held-out samples"): require_same_samples([1,2],[2,3])
