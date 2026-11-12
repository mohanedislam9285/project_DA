from __future__ import annotations
import argparse,json,platform,sys
from pathlib import Path
import numpy as np,pandas as pd
from .config import load_config,validate_config
from .data_audit import run_audit,inspect_source_file
from .eda import run_eda
from .temporal_diagnostics import save_temporal
from .splits import save_folds
from .baselines import evaluate_baseline_fold
from .schemas import RESULTS,PREDICTIONS
from .artifacts import environment_metadata,utc_now
from .benchmark import run_temporal_benchmark
from .final_evaluation import evaluate_future_once
from .cnn_lstm_benchmark import run_cnn_lstm_temporal
from .spatial_benchmark import run_cnn_lstm_spatial
from .cnn_development_studies import run_feature_ablation_cnn,run_train_size_cnn
from .development_model import train_development_model
from .model_inventory import inventory_table
from .ablations import run_feature_ablation
from .robustness import run_train_size_analysis
from .summaries import add_fold_summaries
from .report_tables import generate_development_assets
from .seed_sensitivity import run_cnn_lstm_seed_sensitivity
from .sensitivity import run_hyperparameter_sensitivity,run_preprocessing_ablation
from .sequence_studies import run_cnn_lstm_sensitivity
from .development_diagnostics import run_local_explanations,run_cross_scheme_ranking_stability,run_validation_scheme_ablation
import importlib.util

def _load_clean(cfg):
    p=Path(cfg["_config_dir"])/"data/processed/cleaned_daily.csv"
    if not p.exists(): raise FileNotFoundError("Cleaned data missing; run `python -m pm25_project.cli audit` first")
    df=pd.read_csv(p,parse_dates=["timestamp"])
    if df.empty: raise ValueError("No valid PM2.5 observations remain after cleaning")
    return df

def cmd_audit(cfg):
    df,meta=run_audit(cfg); root=Path(cfg["_config_dir"]); dest=root/cfg["paths"]["results_root"]/cfg["project"]["group_id"]; dest.mkdir(parents=True,exist_ok=True)
    from .eda import exploratory_table
    exploratory_table(df,cfg,meta).to_csv(dest/"Exploratory_Data.csv",index=False,encoding="utf-8")
    save_temporal(df,dest/"Exploratory_temporal.csv")
    inventory_table().to_csv(dest/"model_inventory.csv",index=False,encoding="utf-8")
    print(json.dumps(meta,indent=2))

def cmd_folds(cfg):
    definitions=save_folds(_load_clean(cfg),cfg); print(json.dumps({k:(v.get("status","created"),len(v.get("folds",[]))) for k,v in definitions.items()},indent=2))

def cmd_baselines(cfg):
    df=_load_clean(cfg); defs=save_folds(df,cfg)
    outroot=Path(cfg["_config_dir"])/cfg["paths"]["results_root"]/cfg["project"]["group_id"]; (outroot/"predictions").mkdir(parents=True,exist_ok=True); res=[]; preds=[]
    for foldname in ("folds_temporal.json","folds_spatial.json","folds_standard.json"):
        foldfile=defs[foldname]; scheme=foldfile.get("scheme",foldname.removesuffix(".json"))
        for fold in foldfile.get("folds",[]):
            tr=df.iloc[fold["train_indices"]]; va=df.iloc[fold["validation_indices"]]
            scores,pred=evaluate_baseline_fold(tr,va,cfg["project"]["seasons_naive_period_days"])
            for score in scores:
                mid=score.pop("model_id"); m=score
                res.extend({"model_id":mid,"model_group":"baseline","model_family":"naive","dataset_split":"validation","cv_scheme":scheme,"cv_purpose":"development" if foldname=="folds_temporal.json" else "diagnostic_only","cv_fold":fold["fold"],"niveau":"fold","metric":metric,"is_primary":metric=="rmse","value":value,"n_samples":m["n_samples"],"train_size":len(tr),"horizon_step":1,"forecast_origin":"end_of_day_t","random_state":cfg["project"]["seed"],"hardware":platform.platform(),"fold_file":foldname,"notes":"RMSE/MAE/R2 scored on original µg/m³ scale"} for metric,value in m.items() if metric!="n_samples")
                pr=pred[mid]
                for j,(_,row) in enumerate(va.iterrows()):
                    preds.append({"sample_id":int(row.get("clean_index",row.name)),"station_id":row.station_id,"origin_timestamp":row.timestamp-pd.Timedelta(days=1),"target_timestamp":row.timestamp,"y_true":row.pm25,"y_pred":pr[j],"residual":row.pm25-pr[j] if np.isfinite(pr[j]) else np.nan,"dataset_split":"validation","cv_scheme":scheme,"cv_fold":fold["fold"],"model_id":mid,"config_id":"default","seed":cfg["project"]["seed"]})
    result_path=outroot/"results.csv"; new=add_fold_summaries(pd.DataFrame(res,columns=RESULTS),seed=cfg["project"]["seed"])
    if result_path.exists():
        old=pd.read_csv(result_path); old=old[old.model_group.ne("baseline")] if "model_group" in old else pd.DataFrame(columns=RESULTS); new=pd.concat([old,new],ignore_index=True)
    new.to_csv(result_path,index=False,encoding="utf-8"); pd.DataFrame(preds,columns=PREDICTIONS).to_csv(outroot/"predictions"/"baseline_oof.csv",index=False,encoding="utf-8")
    print(f"Wrote {len(res)} actual fold metric rows and {len(preds)} predictions")

def cmd_benchmark(cfg,include_cnn=True):
    df=_load_clean(cfg); outcomes=[]
    for fold_file in ("folds_temporal.json","folds_spatial.json","folds_standard.json"):
        outcomes.append(run_temporal_benchmark(df,cfg,fold_file=fold_file))
    if include_cnn and importlib.util.find_spec("tensorflow"):
        outcomes.append(run_cnn_lstm_temporal(df,cfg))
    elif include_cnn:
        outcomes.append({"model_id":"CNN_LSTM","status":"blocked_dependency","reason":"TensorFlow is not installed; install the [deep] extra to run the primary model."})
    print(json.dumps(outcomes,indent=2))

def main(argv=None):
    ap=argparse.ArgumentParser(prog="pm25-project"); ap.add_argument("command",choices=[
        "validate-config","source-info","inventory","audit","eda","folds","baselines",
        "cnn-lstm","cnn-lstm-spatial","cnn-lstm-seeds","cnn-sensitivity","cnn-feature-ablation",
        "cnn-train-size","cnn-lstm-finalize-dev","local-explanations","ranking-stability",
        "validation-scheme-ablation","report-assets","ablate","robustness","sensitivity",
        "preprocessing-ablation","evaluate-future","run-all"
    ]); ap.add_argument("--config",default="configs/project.yaml"); args=ap.parse_args(argv)
    cfg,_=load_config(args.config)
    if args.command=="run-all":
        print("ERROR: run-all is disabled because it rewrites folds and invokes future evaluation; run explicit development-only stages.",file=sys.stderr)
        return 2
    try:
        if args.command=="validate-config": validate_config(cfg); print("Configuration valid")
        elif args.command=="source-info": print(json.dumps(inspect_source_file(cfg),indent=2))
        elif args.command=="inventory":
            table=inventory_table(); dest=Path(cfg["_config_dir"])/cfg["paths"]["results_root"]/cfg["project"]["group_id"]; dest.mkdir(parents=True,exist_ok=True); table.to_csv(dest/"model_inventory.csv",index=False,encoding="utf-8"); print(table.to_string(index=False))
        elif args.command=="audit": cmd_audit(cfg)
        elif args.command=="eda": run_eda(_load_clean(cfg),cfg)
        elif args.command=="folds": cmd_folds(cfg)
        elif args.command=="baselines": cmd_baselines(cfg)
        elif args.command=="cnn-lstm": print(json.dumps(run_cnn_lstm_temporal(_load_clean(cfg),cfg),indent=2))
        elif args.command=="cnn-lstm-spatial": print(json.dumps(run_cnn_lstm_spatial(_load_clean(cfg),cfg),indent=2))
        elif args.command=="cnn-lstm-seeds": print(json.dumps(run_cnn_lstm_seed_sensitivity(_load_clean(cfg),cfg),indent=2))
        elif args.command=="cnn-sensitivity": print(json.dumps(run_cnn_lstm_sensitivity(_load_clean(cfg),cfg),indent=2))
        elif args.command=="cnn-feature-ablation": print(json.dumps(run_feature_ablation_cnn(_load_clean(cfg),cfg),indent=2))
        elif args.command=="cnn-train-size": print(json.dumps(run_train_size_cnn(_load_clean(cfg),cfg),indent=2))
        elif args.command=="cnn-lstm-finalize-dev": print(json.dumps(train_development_model(_load_clean(cfg),cfg),indent=2))
        elif args.command=="ablate": print(json.dumps(run_feature_ablation(_load_clean(cfg),cfg),indent=2))
        elif args.command=="robustness": print(json.dumps(run_train_size_analysis(_load_clean(cfg),cfg),indent=2))
        elif args.command=="sensitivity": print(json.dumps(run_hyperparameter_sensitivity(_load_clean(cfg),cfg),indent=2))
        elif args.command=="preprocessing-ablation": print(json.dumps(run_preprocessing_ablation(_load_clean(cfg),cfg),indent=2))
        elif args.command=="local-explanations": print(json.dumps(run_local_explanations(_load_clean(cfg),cfg),indent=2))
        elif args.command=="ranking-stability": print(json.dumps(run_cross_scheme_ranking_stability(cfg),indent=2))
        elif args.command=="validation-scheme-ablation": print(json.dumps(run_validation_scheme_ablation(cfg),indent=2))
        elif args.command=="report-assets":
            out=Path(cfg["_config_dir"])/cfg["paths"]["results_root"]/cfg["project"]["group_id"]; print(json.dumps(generate_development_assets(out,seed=cfg["project"]["seed"]),indent=2))
        elif args.command=="evaluate-future": print(json.dumps(evaluate_future_once(_load_clean(cfg),cfg),indent=2))
        elif args.command=="run-all": cmd_audit(cfg); run_eda(_load_clean(cfg),cfg); cmd_folds(cfg); cmd_baselines(cfg); cmd_benchmark(cfg); print(json.dumps(run_feature_ablation(_load_clean(cfg),cfg),indent=2)); print(json.dumps(run_train_size_analysis(_load_clean(cfg),cfg),indent=2))
    except Exception as e:
        print(f"ERROR: {e}",file=sys.stderr); return 2
    return 0

if __name__=="__main__": raise SystemExit(main())
