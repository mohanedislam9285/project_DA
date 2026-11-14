"""Single-use final future evaluation, strictly after development decisions are locked."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
from .benchmark import supervised_frame
from .metrics import regression_metrics
from .models.model_registry import model_registry
from .tuning import select_configuration
from .schemas import RESULTS
from .sequences import make_sequences
from .features import build_origin_features
from .preprocessing import SequencePreprocessor
from .tuning import chronological_inner_indices
from .models.cnn_lstm import build_cnn_lstm
from .baselines import predict_baselines
from .model_inventory import normalize_model_id

def select_development_winner(results:pd.DataFrame)->str:
    """Select only full-development temporal fold scores, never robustness or diagnostic rows."""
    required={"dataset_split","cv_scheme","niveau","metric","model_id","value","cv_purpose"}
    missing=required-set(results.columns)
    if missing: raise ValueError(f"Development results are missing columns: {sorted(missing)}")
    eligible=results[
        results.dataset_split.eq("validation")
        & results.cv_scheme.eq("temporal_panel_expanding")
        & results.niveau.eq("fold")
        & results.metric.eq("rmse")
        & results.cv_purpose.isin(["development_selection","development"])
    ].copy()
    if "robustness_variant" in eligible:
        eligible=eligible[eligible.robustness_variant.isna()|eligible.robustness_variant.isin(["","reference"])]
    if eligible.empty: raise ValueError("No eligible full-data temporal development scores are available for model selection")
    eligible=eligible.copy()
    eligible["model_id"]=eligible["model_id"].map(normalize_model_id)
    scores=eligible.groupby("model_id").value.mean().sort_values()
    if not np.isfinite(scores.iloc[0]): raise ValueError("Eligible temporal development RMSE scores are not finite")
    return str(scores.index[0])

def validate_locked_finalists(results:pd.DataFrame,model_ids:list[str])->list[str]:
    """Validate a predeclared candidate set using development rows only."""
    normalized_ids=[normalize_model_id(model_id) for model_id in model_ids]
    if not model_ids or len(set(normalized_ids))!=len(normalized_ids):
        raise ValueError("finalist_models must be a non-empty list of unique model IDs")
    required={"dataset_split","cv_scheme","niveau","metric","model_id","cv_purpose"}
    missing=required-set(results.columns)
    if missing: raise ValueError(f"Development results are missing columns: {sorted(missing)}")
    eligible=results[
        results.dataset_split.eq("validation")
        & results.cv_scheme.eq("temporal_panel_expanding")
        & results.niveau.eq("fold")
        & results.metric.eq("rmse")
        & results.cv_purpose.isin(["development_selection","development"])
    ].copy()
    if "robustness_variant" in eligible:
        eligible=eligible[eligible.robustness_variant.isna()|eligible.robustness_variant.isin(["","reference"])]
    eligible["model_id"]=eligible["model_id"].map(normalize_model_id)
    available=set(eligible.model_id.dropna().astype(str))
    absent=sorted(set(normalized_ids)-available)
    if absent: raise ValueError(f"Locked finalists lack temporal development evidence: {absent}")
    return list(model_ids)

def _fit_final_cnn(df,cfg,devset,testset,out):
    import tensorflow as tf
    from .cnn_lstm_benchmark import cnn_lstm_candidates
    base=build_origin_features(df,cfg); names=list(cfg["features"].get("predictors",[]))+list(cfg["features"].get("calendar",[]))
    if cfg["features"].get("include_target_lags",True): names.append("pm25_lag1")
    if cfg["features"].get("include_past_rolling",True): names += [f"pm25_roll{w}_mean" for w in cfg["features"].get("rolling_windows_days",[3,7])]
    names=list(dict.fromkeys(names)); candidates=cnn_lstm_candidates(cfg); cache=[]
    for candidate in candidates:
        window=int(candidate["lookback_days"]); x,y,keys=make_sequences(base,names,window,1); dev=np.flatnonzero(keys.target_row.isin(devset).to_numpy()); test=np.flatnonzero(keys.target_row.isin(testset).to_numpy())
        cache.append((candidate,x,y,keys,dev,test))
    common=set.intersection(*(set(keys.iloc[dev].target_row.astype(int)) for _,_,_,keys,dev,_ in cache))
    if not common: raise ValueError("CNN-LSTM final lookback candidates share no development samples")
    scores=[]
    for candidate,x,y,keys,dev,test in cache:
        window=int(candidate["lookback_days"])
        dev=dev[keys.iloc[dev].target_row.astype(int).isin(common).to_numpy()]; it,iv=chronological_inner_indices(keys.iloc[dev].target_timestamp.to_numpy())
        conf={**candidate,"seed":cfg["project"]["seed"]}; prep=SequencePreprocessor().fit(x[dev[it]]); xi=prep.transform(x[dev[it]]); xv=prep.transform(x[dev[iv]])
        tf.keras.backend.clear_session(); model=build_cnn_lstm(int(window),xi.shape[-1],conf); hist=model.fit(xi,y[dev[it]],validation_data=(xv,y[dev[iv]]),epochs=int(conf.get("epochs",100)),batch_size=int(conf.get("batch_size",64)),callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=int(conf.get("patience",10)),restore_best_weights=True)],verbose=0)
        score=regression_metrics(y[dev[iv]],model.predict(xv,verbose=0).reshape(-1))["rmse"]; best_epoch=int(np.argmin(hist.history["val_loss"])+1); scores.append((score,candidate,best_epoch))
    _,candidate,best_epoch=min(scores,key=lambda item:item[0]); window=int(candidate["lookback_days"])
    _,x,y,keys,dev,test=next(item for item in cache if item[0]==candidate)
    dev=dev[keys.iloc[dev].target_row.astype(int).isin(common).to_numpy()]
    conf={**candidate,"seed":cfg["project"]["seed"]}; prep=SequencePreprocessor().fit(x[dev]); xt=prep.transform(x[dev]); xv=prep.transform(x[test]); tf.keras.backend.clear_session(); model=build_cnn_lstm(int(window),xt.shape[-1],conf)
    model.fit(xt,y[dev],epochs=best_epoch,batch_size=int(conf.get("batch_size",64)),verbose=0); pred=model.predict(xv,verbose=0).reshape(-1)
    model_dir=out/"models"; model_dir.mkdir(parents=True,exist_ok=True); model.save(model_dir/"cnn_lstm_final.keras")
    import joblib
    joblib.dump(prep,model_dir/"cnn_lstm_final_preprocessor.joblib"); (model_dir/"cnn_lstm_final_config.json").write_text(json.dumps({**conf,"lookback_days":window,"feature_columns":names,"feature_columns_after_preprocessing":prep.feature_names_out(names)},indent=2),encoding="utf-8")
    frame=pd.DataFrame({"sample_id":keys.iloc[test].target_row.to_numpy(),"station_id":keys.iloc[test].station_id.to_numpy(),"origin_timestamp":keys.iloc[test].origin_timestamp.to_numpy(),"target_timestamp":keys.iloc[test].target_timestamp.to_numpy(),"y_true":y[test],"y_pred":pred})
    return frame,window,best_epoch

def _fit_final_tabular(df,cfg,devset,testset,out,model_id):
    frame,features=supervised_frame(df,cfg); row_ids=frame.target_row.to_numpy(dtype=int)
    dev=np.flatnonzero(np.isin(row_ids,list(devset))); test=np.flatnonzero(np.isin(row_ids,list(testset)))
    if not len(dev) or not len(test): raise ValueError(f"No samples align to the reserved future partition for {model_id}")
    X=frame[features].to_numpy(dtype=float); y=frame.y.to_numpy(dtype=float)
    dates=pd.to_datetime(frame.target_timestamp,utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    configurations=select_configuration(model_id,X[dev],y[dev],dates[dev],cfg["project"]["seed"],cfg)
    params=configurations[0]["params"]; model=model_registry(cfg["project"]["seed"])[model_id]()
    if params: model.set_params(**params)
    model.fit(X[dev],y[dev]); predictions=np.asarray(model.predict(X[test])).reshape(-1)
    model_dir=out/"models"; model_dir.mkdir(parents=True,exist_ok=True)
    import joblib
    joblib.dump(model,model_dir/f"{model_id.lower()}_final.joblib")
    (model_dir/f"{model_id.lower()}_final_config.json").write_text(
        json.dumps({"model_id":model_id,"params":params,"feature_columns":features,"seed":cfg["project"]["seed"]},indent=2,default=str),
        encoding="utf-8",
    )
    selected=frame.iloc[test]
    return pd.DataFrame({
        "sample_id":selected.target_row.to_numpy(),
        "station_id":selected.station_id.to_numpy(),
        "origin_timestamp":selected.origin_timestamp.to_numpy(),
        "target_timestamp":selected.target_timestamp.to_numpy(),
        "y_true":y[test],
        "y_pred":predictions,
    }),configurations[0]

def evaluate_future_once(df,cfg):
    if not cfg["project"].get("finalization_locked",False): raise ValueError("Set project.finalization_locked=true only after model and feature choices are recorded in experiment notes")
    root=Path(cfg["_config_dir"]); out=root/cfg["paths"]["results_root"]/cfg["project"]["group_id"]; results_path=out/"results.csv"; foldpath=out/"future_test.json"
    if not results_path.exists() or not foldpath.exists(): raise FileNotFoundError("Need saved development results and future_test.json before final evaluation")
    old=pd.read_csv(results_path)
    marker_path=out/"future_test_evaluation.json"
    if marker_path.exists() or old.dataset_split.eq("future_test").any() or (out/"predictions/future_test.csv").exists():
        raise RuntimeError("Future test evaluation has already started or completed; refusing to reuse it")
    future=json.loads(foldpath.read_text(encoding="utf-8")); devset=set(future["development_indices"]); testset=set(future["future_test_indices"])
    locked_models=validate_locked_finalists(old,list(cfg["project"].get("finalist_models",[])))
    available_models={normalize_model_id(name) for name in model_registry(cfg["project"]["seed"]).keys()}
    unavailable=sorted({normalize_model_id(name) for name in locked_models}-available_models-{"CNN_LSTM"})
    if unavailable: raise RuntimeError(f"Locked finalists are unavailable in this environment: {unavailable}")
    marker={"status":"started","started_at":pd.Timestamp.now(tz="UTC").isoformat(),"locked_models":locked_models,"baselines":["persistence","seasonal_naive"],"future_period":[future["future_test_start"],future["future_test_end"]],"note":"Written before scoring to prevent accidental repeat evaluation."}
    marker_path.write_text(json.dumps(marker,indent=2),encoding="utf-8")
    try:
        prediction_frames=[]; result_rows=[]; final_hyperparameters=[]
        for model_id in locked_models:
            if normalize_model_id(model_id)=="CNN_LSTM":
                predframe,selected_window,selected_epochs=_fit_final_cnn(df,cfg,devset,testset,out)
                config_path=out/"models"/"cnn_lstm_final_config.json"
                final_note=f"Locked primary model; lookback={selected_window}; epochs={selected_epochs}; rolling one-day-ahead forecast."
                config=json.loads(config_path.read_text(encoding="utf-8"))
                for key,value in config.items():
                    if key in {"feature_columns","feature_columns_after_preprocessing"}: continue
                    final_hyperparameters.append({"model_id":model_id,"model_group":"G21","cv_repetition":"final_refit","cv_fold":0,"config_rank":1,"is_selected":True,"hyperparameter":key,"value":json.dumps(value) if isinstance(value,(list,dict)) else value})
            else:
                predframe,selected_config=_fit_final_tabular(df,cfg,devset,testset,out,model_id)
                final_note="Locked finalist; configuration selected using chronological inner validation on development data only."
                for key,value in selected_config["params"].items():
                    final_hyperparameters.append({"model_id":model_id,"model_group":"final_refit","cv_repetition":"final_refit","cv_fold":0,"config_rank":1,"is_selected":True,"hyperparameter":key,"value":value})
            metrics=regression_metrics(predframe.y_true,predframe.y_pred)
            predframe=predframe.copy()
            predframe["residual"]=predframe.y_true-predframe.y_pred
            predframe["dataset_split"]="future_test"
            predframe["cv_scheme"]="independent_future_holdout"
            predframe["cv_fold"]=0
            predframe["model_id"]=model_id
            predframe["config_id"]="locked_final"
            predframe["seed"]=cfg["project"]["seed"]
            prediction_frames.append(predframe[["sample_id","station_id","origin_timestamp","target_timestamp","y_true","y_pred","residual","dataset_split","cv_scheme","cv_fold","model_id","config_id","seed"]])
            for metric,value in metrics.items():
                if metric=="n_samples": continue
                result_rows.append({"model_id":model_id,"model_group":"finalist","model_family":model_id,"dataset_split":"future_test","cv_scheme":"independent_future_holdout","cv_purpose":"final_locked_estimate","cv_fold":0,"niveau":"global","metric":metric,"is_primary":metric=="rmse","value":value,"n_samples":metrics["n_samples"],"train_size":len(devset),"horizon_step":1,"forecast_origin":"end_of_day_t","random_state":cfg["project"]["seed"],"fold_file":"future_test.json","notes":final_note})

        test=df.iloc[sorted(testset)]
        baselines=predict_baselines(df.iloc[sorted(devset)],test,cfg["project"]["seasons_naive_period_days"])
        for model_id,predictions in baselines.items():
            available=np.isfinite(predictions)
            if not available.any(): raise ValueError(f"No forecastable future-test samples for baseline {model_id}")
            scored=test.iloc[np.flatnonzero(available)]
            y_true=scored.pm25.to_numpy(dtype=float); y_pred=predictions[available]
            metrics=regression_metrics(y_true,y_pred)
            predframe=pd.DataFrame({"sample_id":scored.clean_index.to_numpy(),"station_id":scored.station_id.to_numpy(),"origin_timestamp":scored.timestamp-pd.Timedelta(days=1),"target_timestamp":scored.timestamp.to_numpy(),"y_true":y_true,"y_pred":y_pred,"residual":y_true-y_pred,"dataset_split":"future_test","cv_scheme":"independent_future_holdout","cv_fold":0,"model_id":model_id,"config_id":"fixed_baseline","seed":cfg["project"]["seed"]})
            prediction_frames.append(predframe)
            for metric,value in metrics.items():
                if metric=="n_samples": continue
                result_rows.append({"model_id":model_id,"model_group":"baseline","model_family":"naive","dataset_split":"future_test","cv_scheme":"independent_future_holdout","cv_purpose":"final_locked_estimate","cv_fold":0,"niveau":"global","metric":metric,"is_primary":metric=="rmse","value":value,"n_samples":metrics["n_samples"],"train_size":len(devset),"horizon_step":1,"forecast_origin":"end_of_day_t","random_state":cfg["project"]["seed"],"fold_file":"future_test.json","notes":f"Fixed rolling one-day-ahead baseline; forecastable {metrics['n_samples']}/{len(testset)} target rows."})

        prediction_output=pd.concat(prediction_frames,ignore_index=True)
        result_output=pd.DataFrame(result_rows,columns=RESULTS)
        (out/"predictions").mkdir(parents=True,exist_ok=True)
        prediction_output.to_csv(out/"predictions/future_test.csv",index=False,encoding="utf-8")
        result_output.to_csv(out/"future_test_results.csv",index=False,encoding="utf-8")
        combined=pd.concat([old,result_output],ignore_index=True); combined.to_csv(results_path,index=False,encoding="utf-8")
        if final_hyperparameters:
            hp_path=out/"hyperparameters.csv"
            existing=pd.read_csv(hp_path) if hp_path.exists() else pd.DataFrame(columns=["model_id","model_group","cv_repetition","cv_fold","config_rank","is_selected","hyperparameter","value"])
            final_hp=pd.DataFrame(final_hyperparameters)
            existing=existing[~(existing.model_id.isin(locked_models)&existing.cv_repetition.eq("final_refit"))]
            pd.concat([existing,final_hp],ignore_index=True).to_csv(hp_path,index=False,encoding="utf-8")
        scored_predictions=prediction_output.copy()
        scored_predictions["year_month"]=pd.to_datetime(scored_predictions.target_timestamp,utc=True).dt.strftime("%Y-%m")
        monthly_rows=[]
        for (model_id,year_month),group in scored_predictions.groupby(["model_id","year_month"],sort=True):
            scores=regression_metrics(group.y_true,group.y_pred)
            monthly_rows.append({"model_id":model_id,"year_month":year_month,"n_samples":scores["n_samples"],"rmse":scores["rmse"],"mae":scores["mae"],"r2":scores["r2"]})
        pd.DataFrame(monthly_rows).to_csv(out/"monthly_future_rmse.csv",index=False,encoding="utf-8")
        future["status"]="evaluated_once"
        future["evaluation_file"]="future_test_evaluation.json"
        foldpath.write_text(json.dumps(future,indent=2),encoding="utf-8")
        marker.update({"status":"completed","completed_at":pd.Timestamp.now(tz="UTC").isoformat(),"models_scored":result_output.model_id.drop_duplicates().tolist(),"metrics_file":"future_test_results.csv","predictions_file":"predictions/future_test.csv"})
        marker_path.write_text(json.dumps(marker,indent=2),encoding="utf-8")
        return {"models_scored":marker["models_scored"],"results":result_output.to_dict(orient="records"),"n_predictions_by_model":prediction_output.groupby("model_id").size().to_dict(),"future_period":marker["future_period"]}
    except Exception as exc:
        marker.update({"status":"failed_no_rerun","failed_at":pd.Timestamp.now(tz="UTC").isoformat(),"error":f"{type(exc).__name__}: {exc}"})
        marker_path.write_text(json.dumps(marker,indent=2),encoding="utf-8")
        raise
