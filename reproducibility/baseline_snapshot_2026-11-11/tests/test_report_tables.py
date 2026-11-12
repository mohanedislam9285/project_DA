import pandas as pd

from pm25_project.report_tables import generate_development_assets,summarize_fold_scores

def test_report_assets_are_development_only_and_trace_predictions(tmp_path):
    results=[]
    for model,values in {"A":[2.0,4.0],"B":[3.0,5.0]}.items():
        for fold,value in enumerate(values,1):
            results.append({"model_id":model,"dataset_split":"validation","cv_scheme":"temporal_panel_expanding","niveau":"fold","metric":"rmse","value":value,"cv_fold":fold,"n_samples":10,"cv_purpose":"development_selection","robustness_variant":""})
    results.append({"model_id":"A","dataset_split":"validation","cv_scheme":"temporal_panel_expanding","niveau":"fold","metric":"rmse","value":0.1,"cv_fold":1,"n_samples":100,"cv_purpose":"training_size_robustness","robustness_variant":"train_10%"})
    result_frame=pd.DataFrame(results)
    summary=summarize_fold_scores(result_frame)
    assert summary.loc[summary.model_id.eq("A"),"mean"].iloc[0]==3.0
    result_frame.to_csv(tmp_path/"results.csv",index=False)
    predictions_dir=tmp_path/"predictions"
    predictions_dir.mkdir()
    predictions=pd.DataFrame([
        {"model_id":"A","dataset_split":"validation","cv_scheme":"temporal_panel_expanding","station_id":"S1","target_timestamp":"2024-01-01","y_true":v,"y_pred":v+1,"residual":-1}
        for v in [5,15,35,60]
    ])
    predictions.to_csv(predictions_dir/"temporal_oof.csv",index=False)
    lstm_predictions = predictions.assign(model_id="LSTM")
    lstm_predictions.to_csv(predictions_dir/"lstm_temporal_oof.csv",index=False)
    baselines=pd.DataFrame([
        {"model_id":"persistence","dataset_split":"validation","cv_scheme":"temporal_panel_expanding","station_id":"S1","target_timestamp":"2024-01-01","y_true":v,"y_pred":v+2,"residual":-2}
        for v in [5,15,35,60]
    ]+[{"model_id":"persistence","dataset_split":"validation","cv_scheme":"random_diagnostic","station_id":"S1","target_timestamp":"2024-01-01","y_true":5,"y_pred":500,"residual":-495}])
    baselines.to_csv(predictions_dir/"baseline_oof.csv",index=False)
    generated=generate_development_assets(tmp_path)
    assert generated["models"]==2
    bands=pd.read_csv(tmp_path/"tables"/"error_by_pm25_band.csv")
    assert set(bands.pm25_band)=={"low (<10)","medium (10-25)","high (25-50)","extreme (>50)"}
    assert "persistence" in set(bands.model_id)
    assert "LSTM" in set(bands.model_id)
    assert bands.loc[bands.model_id.eq("persistence"),"n_samples"].sum()==4
    assert (tmp_path/"figures"/"temporal_rmse_by_model.png").is_file()
    assert (tmp_path/"figures"/"error_by_pm25_band.png").is_file()
