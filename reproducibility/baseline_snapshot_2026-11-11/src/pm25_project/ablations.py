"""Development-fold feature-family ablations using identical held-out samples."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
from .features import build_origin_features
from .benchmark import supervised_frame
from .tuning import select_configuration
from .models.model_registry import model_registry
from .metrics import regression_metrics,paired_bootstrap_ci
from .schemas import ABLATION

def feature_family_variants(configured_features:list[str],family_by_feature:dict[str,str]):
    variants={"full":list(configured_features)}
    for family in sorted(set(family_by_feature.get(f,"Unclassified") for f in configured_features)):
        variants[f"without_{family}"]=[f for f in configured_features if family_by_feature.get(f,"Unclassified")!=family]
    return variants

def require_same_samples(full_keys,variant_keys):
    full=set(full_keys); variant=set(variant_keys)
    if full!=variant: raise ValueError("Ablation variants must be scored on identical held-out samples")

def run_feature_ablation(df,cfg,model_ids=None):
    """Drop one feature family at a time; refit imputation and tuning inside each training fold."""
    root=Path(cfg["_config_dir"]); out=root/cfg["paths"]["results_root"]/cfg["project"]["group_id"]; (out/"logs").mkdir(parents=True,exist_ok=True); folds_path=out/"folds_temporal.json"
    if not folds_path.exists(): raise FileNotFoundError("Run folds before ablation analysis")
    folds=json.loads(folds_path.read_text(encoding="utf-8"))["folds"]; samples,features=supervised_frame(df,cfg); family={}
    for name,spec in cfg["data"].get("predictors",{}).items(): family[name]=spec.get("family","Unclassified")
    for name in cfg["features"].get("calendar",[]): family[name]="Calendar"
    family["pm25_lag1"]="Lag"
    for name in features:
        if name.startswith("pm25_roll"): family[name]="Rolling"
    groups=sorted({family.get(n,"Unclassified") for n in features}); rows=[]; results={}; registry=model_registry(cfg["project"]["seed"])
    model_ids=model_ids or tuple(m for m in ("ElasticNet","SVR_RBF","XGBoost","Bagging","CatBoost") if m in registry)
    for fold in folds:
        tr=np.flatnonzero(samples.target_row.isin(set(fold["train_indices"])).to_numpy()); va=np.flatnonzero(samples.target_row.isin(set(fold["validation_indices"])).to_numpy())
        if not len(tr) or not len(va): continue
        dates=pd.to_datetime(samples.target_timestamp,utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[ns]"); y=samples.y.to_numpy(dtype=float)
        for model_id in model_ids:
            for variant in ["full"]+[f"without_{g}" for g in groups]:
                cols=features if variant=="full" else [col for col in features if family.get(col,"Unclassified")!=variant.removeprefix("without_")]
                if not cols: continue
                x=samples[cols].to_numpy(dtype=float); configs=select_configuration(model_id,x[tr],y[tr],dates[tr],cfg["project"]["seed"],cfg); est=registry[model_id]()
                if configs[0]["params"]: est.set_params(**configs[0]["params"])
                est.fit(x[tr],y[tr]); m=regression_metrics(y[va],est.predict(x[va]))
                for metric,value in m.items():
                    if metric=="n_samples": continue
                    results[(model_id,fold["fold"],variant,metric)]=value; rows.append({"model_id":model_id,"model_group":"comparators","ablation_axis":"feature_family","variant":variant,"cv_scheme":"temporal_panel_expanding","cv_fold":fold["fold"],"metric":metric,"value":value,"delta_vs_full":np.nan,"random_state":cfg["project"]["seed"],"notes":f"{len(cols)} features; tuning and preprocessing fit on outer training samples only."})
    table=pd.DataFrame(rows,columns=ABLATION)
    if table.empty: raise ValueError("No feature-ablation fold scores were generated; inspect window coverage and the temporal fold file")
    for i,row in table.iterrows():
        if row.variant!="full": table.loc[i,"delta_vs_full"]=row.value-results[(row.model_id,row.cv_fold,"full",row.metric)]
    # Supporting fold-paired mean-difference intervals retain the exact axes and sample keys.
    intervals=[]
    for (model,variant,metric),g in table[table.variant!="full"].groupby(["model_id","variant","metric"]):
        d=g.delta_vs_full.dropna().to_numpy(float); lo,hi=paired_bootstrap_ci(d,seed=cfg["project"]["seed"])
        intervals.append({"model_id":model,"variant":variant,"metric":metric,"mean_delta_vs_full":float(np.mean(d)) if len(d) else np.nan,"ci_lower":lo,"ci_upper":hi,"n_paired_folds":len(d)})
        mask=(table.model_id==model)&(table.variant==variant)&(table.metric==metric)
        table.loc[mask,"delta_ci_lower"]=lo
        table.loc[mask,"delta_ci_upper"]=hi
    ablation_path=out/"ablation.csv"
    if ablation_path.exists():
        previous=pd.read_csv(ablation_path)
        same_study=previous.ablation_axis.eq("feature_family")&previous.cv_scheme.eq("temporal_panel_expanding")
        previous=previous[~(same_study&previous.model_id.isin(model_ids))]
        table=pd.concat([previous,table],ignore_index=True)
    table.to_csv(ablation_path,index=False,encoding="utf-8")
    interval_path=out/"logs/ablation_intervals.csv"
    interval_table=pd.DataFrame(intervals)
    if interval_path.exists():
        previous_intervals=pd.read_csv(interval_path)
        previous_intervals=previous_intervals[~previous_intervals.model_id.isin(model_ids)]
        interval_table=pd.concat([previous_intervals,interval_table],ignore_index=True)
    interval_table.to_csv(interval_path,index=False,encoding="utf-8")
    return {"n_rows":len(table),"models":list(model_ids),"feature_families":groups}
