"""Shared temporal-CV benchmark with training-only tuning and fold artifacts."""
from __future__ import annotations
import json,platform,time
from pathlib import Path
import numpy as np,pandas as pd
from .features import build_origin_features
from .metrics import regression_metrics
from .models.model_registry import model_registry
from .tuning import select_configuration
from .schemas import RESULTS,HYPERPARAMETERS,FEATURE_IMPORTANCE,EXPERIMENTS
from .explainability import tabular_permutation_importance,fold_ranking_stability
from .comparisons import comparison_table
from .summaries import add_fold_summaries
from .artifacts import environment_metadata,pipeline_fingerprint,utc_now
from .tuning import candidate_configs

def supervised_frame(df,cfg):
    feature=build_origin_features(df,cfg); feature_cols=[]
    feature_cols += list(cfg["features"].get("predictors",[]))
    feature_cols += [x for x in cfg["features"].get("calendar",[]) if x in feature]
    if cfg["features"].get("include_target_lags",True): feature_cols += ["pm25_lag1"]
    if cfg["features"].get("include_past_rolling",True): feature_cols += [f"pm25_roll{w}_mean" for w in cfg["features"].get("rolling_windows_days",[3,7])]
    if not feature_cols: raise ValueError("No predictive feature columns configured")
    feature_cols=list(dict.fromkeys(feature_cols)); rows=[]
    for station,g in feature.groupby("station_id",sort=True):
        g=g.sort_values("timestamp").reset_index(drop=True)
        for i in range(len(g)-1):
            if g.loc[i+1,"timestamp"]-g.loc[i,"timestamp"] != pd.Timedelta(days=1): continue
            row={k:g.loc[i,k] for k in feature_cols}; row.update(station_id=station,origin_timestamp=g.loc[i,"timestamp"],target_timestamp=g.loc[i+1,"timestamp"],y=float(g.loc[i+1,"pm25"]),target_row=int(g.loc[i+1,"clean_index"]))
            rows.append(row)
    samples=pd.DataFrame(rows)
    if samples.empty: raise ValueError("No contiguous one-day-ahead samples could be built")
    return samples,feature_cols

def run_temporal_benchmark(df,cfg,model_ids=None,fold_file="folds_temporal.json"):
    started_at=utc_now(); root=Path(cfg["_config_dir"]); group=cfg["project"]["group_id"]; out=root/cfg["paths"]["results_root"]/group; out.mkdir(parents=True,exist_ok=True); (out/"predictions").mkdir(parents=True,exist_ok=True)
    foldfile=out/fold_file
    if not foldfile.exists(): raise FileNotFoundError("Run the folds stage first")
    fold_definition=json.loads(foldfile.read_text(encoding="utf-8")); folds=fold_definition.get("folds",[]); cv_scheme=fold_definition.get("scheme","diagnostic")
    if not folds: return {"status":fold_definition.get("status","no_folds"),"reason":fold_definition.get("reason","No folds in definition"),"fold_file":fold_file}
    samples,features=supervised_frame(df,cfg); target_rows=samples.target_row.to_numpy(dtype=int); X=samples[features].to_numpy(dtype=float); y=samples.y.to_numpy(dtype=float); dates=pd.to_datetime(samples.target_timestamp,utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    registry=model_registry(cfg["project"]["seed"]); model_ids=model_ids or ["Lasso","ElasticNet","Ridge","Huber","KNN","SVR_RBF","DecisionTree","ExtraTrees","RandomForest","GradientBoosting","MLP","Voting","Bagging","AdaBoost"]+[m for m in ("XGBoost","LightGBM","CatBoost") if m in registry]
    unknown=sorted(set(model_ids)-set(registry));
    if unknown: raise ValueError(f"Unknown/unavailable benchmark models: {unknown}")
    result_rows=[]; pred_rows=[]; hp_rows=[]; importance_rows=[]
    for fold in folds:
        trset=set(fold["train_indices"]); vaset=set(fold["validation_indices"])
        tr=np.flatnonzero(np.isin(target_rows,list(trset))); va=np.flatnonzero(np.isin(target_rows,list(vaset)))
        if not len(tr) or not len(va): continue
        for model_id in model_ids:
            started=time.perf_counter(); configs=select_configuration(model_id,X[tr],y[tr],dates[tr],cfg["project"]["seed"],cfg); search_time=time.perf_counter()-started
            best=configs[0]; est=registry[model_id]();
            if best["params"]: est.set_params(**best["params"])
            fitstart=time.perf_counter(); est.fit(X[tr],y[tr]); fit_time=time.perf_counter()-fitstart
            inferstart=time.perf_counter(); pred=np.asarray(est.predict(X[va])).reshape(-1); infer_time=time.perf_counter()-inferstart
            scores=regression_metrics(y[va],pred)
            for metric,val in scores.items():
                if metric=="n_samples": continue
                purpose="development_selection" if fold_file=="folds_temporal.json" else "diagnostic_only"
                result_rows.append({"model_id":model_id,"model_group":"comparators","model_family":model_id,"dataset_split":"validation","cv_scheme":cv_scheme,"cv_purpose":purpose,"cv_fold":fold["fold"],"niveau":"fold","metric":metric,"is_primary":metric=="rmse","value":val,"n_samples":scores["n_samples"],"train_size":len(tr),"horizon_step":1,"forecast_origin":"end_of_day_t","config_rank":1,"random_state":cfg["project"]["seed"],"fit_time_sec":fit_time,"search_time_sec":search_time,"inference_time_sec":infer_time,"hardware":platform.platform(),"n_features":len(features),"fold_file":fold_file,"notes":"inner validation used only outer training data; original µg/m³ scale"})
            for rank,item in enumerate(configs[:3],1):
                for param,value in item["params"].items(): hp_rows.append({"model_id":model_id,"model_group":"comparators","cv_repetition":1,"cv_fold":fold["fold"],"config_rank":rank,"is_selected":rank==1,"hyperparameter":param,"value":value})
            for local_idx,sample_idx in enumerate(va):
                row=samples.iloc[sample_idx]; pred_rows.append({"sample_id":row.target_row,"station_id":row.station_id,"origin_timestamp":row.origin_timestamp,"target_timestamp":row.target_timestamp,"y_true":y[sample_idx],"y_pred":pred[local_idx],"residual":y[sample_idx]-pred[local_idx],"dataset_split":"validation","cv_scheme":cv_scheme,"cv_fold":fold["fold"],"model_id":model_id,"config_id":f"fold{fold['fold']}_rank1","seed":cfg["project"]["seed"]})
            imp=tabular_permutation_importance(est,X[va],y[va],features,seed=cfg["project"]["seed"])
            for r in imp.itertuples(): importance_rows.append({"model_id":model_id,"model_group":"comparators","cv_repetition":1,"cv_fold":fold["fold"],"method":"permutation_rmse","is_mandatory":True,"feature":r.feature,"importance":r.importance,"rank":r.rank})
    current=add_fold_summaries(pd.DataFrame(result_rows,columns=RESULTS),seed=cfg["project"]["seed"]); result_path=out/"results.csv"
    if result_path.exists():
        old=pd.read_csv(result_path)
        if {"model_id","cv_scheme"}.issubset(old):
            old=old[~(old.cv_scheme.eq(cv_scheme)&old.model_id.isin(model_ids))]
        else:
            old=pd.DataFrame(columns=RESULTS)
        current=pd.concat([old,current],ignore_index=True)
    current.to_csv(result_path,index=False,encoding="utf-8")
    comparison_table(current,seed=cfg["project"]["seed"],n_perm=10000).to_csv(out/"comparisons.csv",index=False,encoding="utf-8")
    pred_path=out/"predictions"/(fold_file.replace("folds_","").replace(".json","")+"_oof.csv")
    new_predictions=pd.DataFrame(pred_rows,columns="sample_id,station_id,origin_timestamp,target_timestamp,y_true,y_pred,residual,dataset_split,cv_scheme,cv_fold,model_id,config_id,seed".split(","))
    if pred_path.exists():
        old_predictions=pd.read_csv(pred_path)
        if {"model_id","cv_scheme"}.issubset(old_predictions):
            old_predictions=old_predictions[~(old_predictions.cv_scheme.eq(cv_scheme)&old_predictions.model_id.isin(model_ids))]
        else:
            old_predictions=pd.DataFrame(columns=new_predictions.columns)
        new_predictions=pd.concat([old_predictions,new_predictions],ignore_index=True)
    new_predictions.to_csv(pred_path,index=False,encoding="utf-8")
    logdir=out/"logs"; logdir.mkdir(parents=True,exist_ok=True)
    if fold_file=="folds_temporal.json":
        for filename,columns in (("hyperparameters.csv",HYPERPARAMETERS),("feature_importance.csv",FEATURE_IMPORTANCE)):
            path=out/filename; new_rows=pd.DataFrame(hp_rows if filename=="hyperparameters.csv" else importance_rows,columns=columns)
            if path.exists():
                previous=pd.read_csv(path); previous=previous[~previous.model_id.isin(model_ids)]; new_rows=pd.concat([previous,new_rows],ignore_index=True)
            new_rows.to_csv(path,index=False,encoding="utf-8")
        all_importance=pd.read_csv(out/"feature_importance.csv")
        fold_ranking_stability(all_importance).to_csv(out/"ranking_stability.csv",index=False,encoding="utf-8")
    else:
        (logdir/f"hyperparameters_{cv_scheme}.csv").parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(hp_rows,columns=HYPERPARAMETERS).to_csv(logdir/f"hyperparameters_{cv_scheme}.csv",index=False,encoding="utf-8")
        pd.DataFrame(importance_rows,columns=FEATURE_IMPORTANCE).to_csv(logdir/f"feature_importance_{cv_scheme}.csv",index=False,encoding="utf-8")
    env=environment_metadata(); exp=[]
    for model_id in model_ids:
        exp.append({"model_id":model_id,"model_group":"comparators","description":f"Outer {cv_scheme} CV; training-only chronological inner tuning","pipeline_hash":pipeline_fingerprint(cfg),"library_versions":json.dumps(env,sort_keys=True),"python_version":env["python"],"hardware":env["platform"],"search_strategy":"predeclared finite chronological inner validation","n_trials":len(candidate_configs(model_id,cfg["project"]["seed"],cfg)),"search_space_id":"configs/search_spaces.yaml","fold_file":fold_file,"start_time":started_at,"end_time":utc_now(),"git_commit":"","notes":"No Git commit recorded; random folds are diagnostic only."})
    exp_path=out/"experiments.csv"; exp_new=pd.DataFrame(exp,columns=EXPERIMENTS)
    if exp_path.exists():
        exp_old=pd.read_csv(exp_path)
        exp_old=exp_old[~(exp_old.model_id.isin(model_ids)&exp_old.fold_file.eq(fold_file))]
        exp_new=pd.concat([exp_old,exp_new],ignore_index=True)
    exp_new.to_csv(exp_path,index=False,encoding="utf-8")
    return {"n_metric_rows":len(result_rows),"n_predictions":len(pred_rows),"n_models":len(model_ids),"models":model_ids,"n_features":len(features)}
