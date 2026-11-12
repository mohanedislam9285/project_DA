"""CNN-LSTM outer temporal CV. TensorFlow is loaded lazily and never required for audits."""
from __future__ import annotations
import json,platform,time
from pathlib import Path
import numpy as np,pandas as pd
from .features import build_origin_features
from .sequences import make_sequences
from .preprocessing import SequencePreprocessor
from .tuning import chronological_inner_indices
from .metrics import regression_metrics
from .models.cnn_lstm import build_cnn_lstm
from .explainability import sequence_channel_permutation,fold_ranking_stability
from .comparisons import comparison_table
from .schemas import RESULTS,HYPERPARAMETERS,FEATURE_IMPORTANCE,EXPERIMENTS
from .summaries import add_fold_summaries
from .artifacts import environment_metadata,pipeline_fingerprint,utc_now

def cnn_lstm_candidates(cfg):
    base=cfg["model"]["cnn_lstm"]
    configured=cfg.get("_search_spaces",{}).get("implemented_cnn_lstm_candidates")
    if configured: return [{**base,**candidate} for candidate in configured]
    windows=cfg["project"].get("lookback_candidates_days",[7,14,28,56])
    # Fixed, predeclared three-way lookback search; architecture values come from config.
    return [{**base,"lookback_days":int(w)} for w in windows[:3]]

def _prepare(feature_table,feature_columns,lookback,train_rows,valid_rows):
    x,y,keys=make_sequences(feature_table,feature_columns,lookback=lookback,horizon=1)
    tr=np.flatnonzero(keys.target_row.isin(train_rows).to_numpy()); va=np.flatnonzero(keys.target_row.isin(valid_rows).to_numpy())
    if not len(tr) or not len(va): raise ValueError(f"Insufficient complete {lookback}-day windows in a fold")
    return x,y,keys,tr,va

def run_cnn_lstm_temporal(df,cfg):
    try: import tensorflow as tf
    except ImportError as exc: raise RuntimeError("The primary CNN-LSTM benchmark requires TensorFlow. Install the [deep] extra in a supported environment.") from exc
    started_at=utc_now(); root=Path(cfg["_config_dir"]); out=root/cfg["paths"]["results_root"]/cfg["project"]["group_id"]; fold_path=out/"folds_temporal.json"
    if not fold_path.exists(): raise FileNotFoundError("Run the folds stage before CNN-LSTM training")
    folds=json.loads(fold_path.read_text(encoding="utf-8"))["folds"]; feature_table=build_origin_features(df,cfg)
    feature_columns=list(cfg["features"].get("predictors",[]))+list(cfg["features"].get("calendar",[]))
    if cfg["features"].get("include_target_lags",True): feature_columns.append("pm25_lag1")
    if cfg["features"].get("include_past_rolling",True): feature_columns += [f"pm25_roll{w}_mean" for w in cfg["features"].get("rolling_windows_days",[3,7])]
    feature_columns=list(dict.fromkeys(feature_columns)); candidates=cnn_lstm_candidates(cfg)
    result_rows=[]; pred_rows=[]; hp_rows=[]; imp_rows=[]; top3_rows=[]; timing_rows=[]
    for fold in folds:
        # Tune with a chronological inner split of outer training only.
        scores=[]; cached={}; target_sets=[]
        for candidate in candidates:
            x,y,keys,tr,va=_prepare(feature_table,feature_columns,candidate["lookback_days"],set(fold["train_indices"]),set(fold["validation_indices"]))
            cached[candidate["lookback_days"]]=(x,y,keys,tr,va); target_sets.append(set(keys.target_row.astype(int)))
        common_targets=set.intersection(*target_sets) if target_sets else set()
        if not common_targets: raise ValueError("The predeclared lookback candidates share no fold samples; inspect temporal coverage or narrow the search before running.")
        for rank,candidate in enumerate(candidates,1):
            x,y,keys,tr,va=cached[candidate["lookback_days"]]
            tr=tr[keys.iloc[tr].target_row.astype(int).isin(common_targets).to_numpy()]; va=va[keys.iloc[va].target_row.astype(int).isin(common_targets).to_numpy()]
            inner_train,inner_val=chronological_inner_indices(keys.iloc[tr].target_timestamp.to_numpy())
            t0=time.perf_counter(); prep=SequencePreprocessor().fit(x[tr[inner_train]]); xit=prep.transform(x[tr[inner_train]]); xiv=prep.transform(x[tr[inner_val]])
            tf.keras.backend.clear_session(); tuning_config={k:v for k,v in candidate.items() if k!="lookback_days"}; tuning_config["seed"]=cfg["project"]["seed"]
            model=build_cnn_lstm(candidate["lookback_days"],xit.shape[-1],tuning_config); history=model.fit(xit,y[tr[inner_train]],validation_data=(xiv,y[tr[inner_val]]),epochs=int(candidate.get("epochs",100)),batch_size=int(candidate.get("batch_size",64)),callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=int(candidate.get("patience",10)),restore_best_weights=True)],verbose=0)
            pred=model.predict(xiv,verbose=0).reshape(-1); score=regression_metrics(y[tr[inner_val]],pred)["rmse"]; best_epoch=int(np.argmin(history.history["val_loss"])+1)
            scores.append({"config":candidate,"lookback":candidate["lookback_days"],"rmse":score,"best_epoch":best_epoch,"search_time":time.perf_counter()-t0})
        scores.sort(key=lambda r:r["rmse"]); chosen=scores[0]; cand=chosen["config"]
        x,y,keys,tr,va=_prepare(feature_table,feature_columns,cand["lookback_days"],set(fold["train_indices"]),set(fold["validation_indices"]))
        prep=SequencePreprocessor().fit(x[tr]); xt=prep.transform(x[tr]); xv=prep.transform(x[va]); tf.keras.backend.clear_session()
        fit_config={k:v for k,v in cand.items() if k!="lookback_days"}; fit_config["seed"]=cfg["project"]["seed"]+int(fold["fold"])
        model=build_cnn_lstm(cand["lookback_days"],xt.shape[-1],fit_config); t0=time.perf_counter(); outer_history=model.fit(xt,y[tr],epochs=chosen["best_epoch"],batch_size=int(cand.get("batch_size",64)),verbose=0); fit_time=time.perf_counter()-t0
        t0=time.perf_counter(); pred=model.predict(xv,verbose=0).reshape(-1); inference_time=time.perf_counter()-t0; metrics=regression_metrics(y[va],pred)
        (out/"histories").mkdir(parents=True,exist_ok=True)
        pd.DataFrame(outer_history.history).to_csv(out/"histories"/f"cnn_lstm_temporal_fold_{fold['fold']}_outer_training.csv",index=False)
        timing_rows.append({"cv_fold":int(fold["fold"]),"train_start":fold.get("train_start"),"train_end":fold.get("train_end"),"validation_start":fold.get("validation_start"),"validation_end":fold.get("validation_end"),"train_rows":int(len(fold["train_indices"])),"validation_rows":int(len(fold["validation_indices"])),"train_stations":int(df.iloc[fold["train_indices"]].station_id.nunique()),"validation_stations":int(df.iloc[fold["validation_indices"]].station_id.nunique()),"training_sequences":int(len(tr)),"validation_sequences":int(len(va)),"fit_time_sec":fit_time,"search_time_sec":sum(s["search_time"] for s in scores),"inference_time_sec":inference_time,"n_parameters":int(model.count_params()),"model_size_mb":float((out/"models"/f"cnn_lstm_temporal_fold_{fold['fold']}.keras").stat().st_size/1024**2) if (out/"models"/f"cnn_lstm_temporal_fold_{fold['fold']}.keras").exists() else np.nan,"seed":int(fit_config["seed"])})
        if any(item["config"]["lookback_days"]!=cand["lookback_days"] for item in scores[:3]):
            raise ValueError("Top CNN-LSTM configuration comparison requires a matched lookback cohort")
        for rank,item in enumerate(scores[:3],1):
            if rank==1:
                candidate_prediction=pred
            else:
                tf.keras.backend.clear_session()
                candidate_config={k:v for k,v in item["config"].items() if k!="lookback_days"}
                candidate_config["seed"]=fit_config["seed"]
                candidate_model=build_cnn_lstm(item["config"]["lookback_days"],xt.shape[-1],candidate_config)
                candidate_model.fit(xt,y[tr],epochs=item["best_epoch"],batch_size=int(item["config"].get("batch_size",64)),verbose=0)
                candidate_prediction=candidate_model.predict(xv,verbose=0).reshape(-1)
            candidate_metrics=regression_metrics(y[va],candidate_prediction)
            top3_rows.append({
                "model_id":"CNN_LSTM","cv_fold":fold["fold"],"config_rank":rank,
                "inner_validation_rmse":item["rmse"],"best_epoch":item["best_epoch"],
                "outer_validation_rmse":candidate_metrics["rmse"],
                "outer_validation_mae":candidate_metrics["mae"],
                "outer_validation_r2":candidate_metrics["r2"],
                "n_samples":candidate_metrics["n_samples"],
                "configuration":json.dumps(item["config"],sort_keys=True),
            })
        for metric,value in metrics.items():
            if metric!="n_samples": result_rows.append({"model_id":"CNN_LSTM","model_group":"Group_16","model_family":"CNN-LSTM","dataset_split":"validation","cv_scheme":"temporal_panel_expanding","cv_purpose":"development_selection","cv_fold":fold["fold"],"niveau":"fold","metric":metric,"is_primary":metric=="rmse","value":value,"n_samples":metrics["n_samples"],"train_size":len(tr),"horizon_step":1,"forecast_origin":"end_of_day_t","config_rank":1,"random_state":fit_config["seed"],"fit_time_sec":fit_time,"search_time_sec":sum(s["search_time"] for s in scores),"inference_time_sec":inference_time,"n_parameters":model.count_params(),"hardware":platform.platform(),"n_features":len(feature_columns),"fold_file":"folds_temporal.json","notes":"Inner tuning and early stopping used outer training data only."})
        for rank,s in enumerate(scores[:3],1):
            hp_rows.extend({"model_id":"CNN_LSTM","model_group":"Group_16","cv_repetition":1,"cv_fold":fold["fold"],"config_rank":rank,"is_selected":rank==1,"hyperparameter":k,"value":v} for k,v in s["config"].items())
        for i in range(len(va)):
            key=keys.iloc[va[i]]; pred_rows.append({"sample_id":key.target_row,"station_id":key.station_id,"origin_timestamp":key.origin_timestamp,"target_timestamp":key.target_timestamp,"y_true":y[va[i]],"y_pred":pred[i],"residual":y[va[i]]-pred[i],"dataset_split":"validation","cv_scheme":"temporal_panel_expanding","cv_fold":fold["fold"],"model_id":"CNN_LSTM","config_id":f"fold{fold['fold']}_rank1","seed":fit_config["seed"]})
        importance=sequence_channel_permutation(model,xv,y[va],prep.feature_names_out(feature_columns),seed=fit_config["seed"],n_repeats=10)
        for row in importance.itertuples(): imp_rows.append({"model_id":"CNN_LSTM","model_group":"Group_16","cv_repetition":1,"cv_fold":fold["fold"],"method":"sequence_channel_permutation_rmse","is_mandatory":True,"feature":row.feature,"importance":row.importance,"rank":row.rank})
        modeldir=out/"models"; modeldir.mkdir(parents=True,exist_ok=True); modelpath=modeldir/f"cnn_lstm_temporal_fold_{fold['fold']}.keras"; model.save(modelpath); timing_rows[-1]["model_size_mb"]=modelpath.stat().st_size/1024**2
        import joblib
        joblib.dump(prep,modeldir/f"cnn_lstm_temporal_fold_{fold['fold']}_preprocessor.joblib")
        (modeldir/f"cnn_lstm_temporal_fold_{fold['fold']}_config.json").write_text(json.dumps({**cand,"seed":fit_config["seed"],"best_epoch":chosen["best_epoch"],"inner_validation_rmse":chosen["rmse"],"top3_inner_configs":[{"configuration":item["config"],"inner_validation_rmse":item["rmse"],"best_epoch":item["best_epoch"]} for item in scores[:3]],"feature_columns":feature_columns,"feature_columns_after_preprocessing":prep.feature_names_out(feature_columns)},indent=2),encoding="utf-8")
    # Keep the pre-existing mixed historical results.csv unchanged; this run is
    # written to a dedicated CNN-LSTM development file.
    result_path=out/"temporal_cv_results_long.csv"; new=add_fold_summaries(pd.DataFrame(result_rows,columns=RESULTS),seed=cfg["project"]["seed"])
    if result_path.exists():
        old=pd.read_csv(result_path); old=old[~(old.model_id.eq("CNN_LSTM") & old.cv_scheme.eq("temporal_panel_expanding"))]; new=pd.concat([old,new],ignore_index=True)
    new.to_csv(result_path,index=False,encoding="utf-8"); pd.DataFrame(timing_rows).to_csv(out/"temporal_cv_fold_timing.csv",index=False,encoding="utf-8"); predpath=out/"predictions"/"cnn_lstm_temporal_oof.csv"; predpath.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(pred_rows).to_csv(predpath,index=False,encoding="utf-8")
    hp_path=out/"hyperparameters.csv"; hp=pd.DataFrame(hp_rows,columns=HYPERPARAMETERS)
    if hp_path.exists():
        oldhp=pd.read_csv(hp_path); oldhp=oldhp[~oldhp.model_id.eq("CNN_LSTM")]; hp=pd.concat([oldhp,hp],ignore_index=True)
    hp.to_csv(hp_path,index=False,encoding="utf-8"); imp_path=out/"feature_importance.csv"; imp=pd.DataFrame(imp_rows,columns=FEATURE_IMPORTANCE)
    top3_path=out/"cnn_lstm_top3_configs.csv"; top3=pd.DataFrame(top3_rows)
    if top3_path.exists():
        old_top3=pd.read_csv(top3_path); old_top3=old_top3[old_top3.model_id.ne("CNN_LSTM")]; top3=pd.concat([old_top3,top3],ignore_index=True)
    top3.to_csv(top3_path,index=False,encoding="utf-8")
    if imp_path.exists():
        oldimp=pd.read_csv(imp_path); oldimp=oldimp[~oldimp.model_id.eq("CNN_LSTM")]; imp=pd.concat([oldimp,imp],ignore_index=True)
    imp.to_csv(imp_path,index=False,encoding="utf-8"); fold_ranking_stability(imp).to_csv(out/"ranking_stability.csv",index=False,encoding="utf-8")
    comparison_table(new,seed=cfg["project"]["seed"]).to_csv(out/"comparisons.csv",index=False,encoding="utf-8")
    env=environment_metadata(); env["tensorflow"]=tf.__version__; exp_path=out/"experiments.csv"; exp=pd.DataFrame([{"model_id":"CNN_LSTM","model_group":"Group_16","description":"CNN-LSTM temporal outer CV with inner configuration selection and epoch stopping","pipeline_hash":pipeline_fingerprint(cfg),"library_versions":json.dumps(env,sort_keys=True),"python_version":env["python"],"hardware":env["platform"],"search_strategy":"three predeclared CNN-LSTM candidates; chronological inner validation","n_trials":len(candidates),"search_space_id":"configs/search_spaces.yaml","fold_file":"folds_temporal.json","start_time":started_at,"end_time":utc_now(),"git_commit":"","notes":"Seeds fixed per fold; final-test period not used."}],columns=EXPERIMENTS)
    if exp_path.exists():
        oldexp=pd.read_csv(exp_path); exp=pd.concat([oldexp,exp],ignore_index=True)
    exp.to_csv(exp_path,index=False,encoding="utf-8")
    return {"model_id":"CNN_LSTM","n_folds":len(folds),"metrics_rows":len(result_rows),"predictions":len(pred_rows)}
