import pytest
import pandas as pd
import json
from pm25_project.final_evaluation import _fit_final_cnn,evaluate_future_once,select_development_winner

def test_future_test_requires_explicit_finalization_lock(panel,config):
    with pytest.raises(ValueError,match="finalization_locked"):
        evaluate_future_once(panel,config)

def test_model_selection_ignores_diagnostic_and_training_size_rows():
    rows=[
        {"model_id":"persistence","dataset_split":"validation","cv_scheme":"temporal_panel_expanding","niveau":"fold","metric":"rmse","value":8.0,"cv_purpose":"development"},
        {"model_id":"ExtraTrees","dataset_split":"validation","cv_scheme":"temporal_panel_expanding","niveau":"fold","metric":"rmse","value":7.0,"cv_purpose":"development_selection","robustness_variant":""},
        {"model_id":"ExtraTrees","dataset_split":"validation","cv_scheme":"temporal_panel_expanding","niveau":"fold","metric":"rmse","value":1.0,"cv_purpose":"training_size_robustness","robustness_variant":"train_10%"},
        {"model_id":"RandomForest","dataset_split":"validation","cv_scheme":"random_diagnostic","niveau":"fold","metric":"rmse","value":0.1,"cv_purpose":"diagnostic_only"},
    ]
    assert select_development_winner(pd.DataFrame(rows))=="ExtraTrees"

def test_locked_finalists_require_temporal_development_evidence():
    from pm25_project.final_evaluation import validate_locked_finalists
    rows=[
        {"model_id":"Bagging","dataset_split":"validation","cv_scheme":"temporal_panel_expanding","niveau":"fold","metric":"rmse","value":8.0,"cv_purpose":"development_selection"},
        {"model_id":"CatBoost","dataset_split":"validation","cv_scheme":"spatial_group_kfold","niveau":"fold","metric":"rmse","value":7.0,"cv_purpose":"diagnostic_only"},
    ]
    assert validate_locked_finalists(pd.DataFrame(rows),["Bagging"])==["Bagging"]
    with pytest.raises(ValueError,match="lack temporal development evidence"):
        validate_locked_finalists(pd.DataFrame(rows),["CatBoost"])


def test_model_id_aliases_are_normalized_for_selection_and_locking():
    from pm25_project.final_evaluation import validate_locked_finalists
    rows=[
        {"model_id":"CNN-LSTM","dataset_split":"validation","cv_scheme":"temporal_panel_expanding","niveau":"fold","metric":"rmse","value":2.0,"cv_purpose":"development_selection"},
        {"model_id":"Bagging","dataset_split":"validation","cv_scheme":"temporal_panel_expanding","niveau":"fold","metric":"rmse","value":4.0,"cv_purpose":"development_selection"},
    ]
    assert select_development_winner(pd.DataFrame(rows)) == "CNN_LSTM"
    assert validate_locked_finalists(pd.DataFrame(rows), ["CNN-LSTM"]) == ["CNN-LSTM"]


def test_future_evaluation_scores_locked_models_once(panel,config,tmp_path,monkeypatch):
    import pm25_project.final_evaluation as final_evaluation
    panel=panel.copy()
    panel["clean_index"]=range(len(panel))
    panel["timestamp"]=pd.to_datetime(panel.timestamp,utc=True)
    test_start=panel.timestamp.max()-pd.Timedelta(days=9)
    dev_indices=panel.index[panel.timestamp<test_start].tolist()
    test_indices=panel.index[panel.timestamp>=test_start].tolist()
    config["project"].update(finalization_locked=True,finalist_models=["CNN_LSTM"])
    out=tmp_path/"results"/config["project"]["group_id"]
    (out/"predictions").mkdir(parents=True)
    pd.DataFrame([{
        "model_id":"CNN_LSTM","dataset_split":"validation","cv_scheme":"temporal_panel_expanding",
        "niveau":"fold","metric":"rmse","value":8.0,"cv_purpose":"development_selection",
    }]).to_csv(out/"results.csv",index=False)
    (out/"future_test.json").write_text(json.dumps({
        "development_indices":dev_indices,"future_test_indices":test_indices,
        "future_test_start":str(test_start.date()),"future_test_end":str(panel.timestamp.max().date()),
    }),encoding="utf-8")

    def fake_cnn(frame,cfg,devset,testset,out):
        scored=frame.iloc[sorted(testset)]
        (out/"models").mkdir(parents=True,exist_ok=True)
        (out/"models"/"cnn_lstm_final_config.json").write_text(json.dumps({"lookback_days":7,"filters":32}),encoding="utf-8")
        return pd.DataFrame({
            "sample_id":scored.clean_index.to_numpy(),"station_id":scored.station_id.to_numpy(),
            "origin_timestamp":scored.timestamp-pd.Timedelta(days=1),"target_timestamp":scored.timestamp,
            "y_true":scored.pm25.to_numpy(),"y_pred":scored.pm25.to_numpy(),
        }),7,10
    monkeypatch.setattr(final_evaluation,"_fit_final_cnn",fake_cnn)
    monkeypatch.setattr(final_evaluation,"predict_baselines",lambda train,test,period:{
        "persistence":test.pm25.to_numpy(),"seasonal_naive":test.pm25.to_numpy(),
    })

    result=final_evaluation.evaluate_future_once(panel,config)
    assert set(result["models_scored"])=={"CNN_LSTM","persistence","seasonal_naive"}
    assert json.loads((out/"future_test_evaluation.json").read_text(encoding="utf-8"))["status"]=="completed"
    assert set(pd.read_csv(out/"monthly_future_rmse.csv").columns)=={"model_id","year_month","n_samples","rmse","mae","r2"}
    with pytest.raises(RuntimeError,match="already started or completed"):
        final_evaluation.evaluate_future_once(panel,config)

def test_final_cnn_tunes_distinct_architectures_with_same_lookback(panel,config,tmp_path,monkeypatch):
    pytest.importorskip("tensorflow")
    from pm25_project import cnn_lstm_benchmark
    panel=panel.copy()
    panel["clean_index"]=panel["source_row"]
    panel["timestamp"]=pd.to_datetime(panel.timestamp)
    test_start=panel.timestamp.max()-pd.Timedelta(days=9)
    devset=set(panel.loc[panel.timestamp<test_start,"clean_index"].astype(int))
    testset=set(panel.loc[panel.timestamp>=test_start,"clean_index"].astype(int))
    configs=[
        {"lookback_days":7,"filters":4,"kernel_size":2,"lstm_units":4,"dropout":0.0,"learning_rate":0.001,"epochs":2,"batch_size":8,"patience":1},
        {"lookback_days":7,"filters":8,"kernel_size":2,"lstm_units":4,"dropout":0.0,"learning_rate":0.001,"epochs":2,"batch_size":8,"patience":1},
        {"lookback_days":7,"filters":4,"kernel_size":3,"lstm_units":8,"dropout":0.1,"learning_rate":0.0005,"epochs":2,"batch_size":8,"patience":1},
    ]
    monkeypatch.setattr(cnn_lstm_benchmark,"cnn_lstm_candidates",lambda _:configs)
    config["features"]["predictors"]=[]
    config["model"]={"cnn_lstm":{}}
    out=tmp_path/"results"
    predictions,lookback,epochs=_fit_final_cnn(panel,config,devset,testset,out)
    assert lookback==7
    assert 1<=epochs<=2
    assert len(predictions)>0
    selected=pd.read_json(out/"models"/"cnn_lstm_final_config.json",typ="series")
    assert any(selected["filters"]==candidate["filters"] and selected["kernel_size"]==candidate["kernel_size"] for candidate in configs)
