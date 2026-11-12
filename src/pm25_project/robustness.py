"""Development-only sensitivity analyses; callers must pass already reserved development partitions."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .metrics import regression_metrics
from pathlib import Path
import json,platform
from .benchmark import supervised_frame
from .models.model_registry import model_registry
from .tuning import select_configuration
from .schemas import RESULTS
from .summaries import add_fold_summaries

def chronological_training_size_curve(estimator_factory,x_dev,y_dev,date_dev,x_valid,y_valid,fractions=(.1,.25,.5,.75,1.0)):
    dates=pd.to_datetime(date_dev); order=np.argsort(dates.to_numpy()); unique=np.sort(dates.unique()); rows=[]
    for fraction in fractions:
        n_dates=max(1,int(np.ceil(len(unique)*fraction))); allowed=set(unique[:n_dates]); take=np.asarray([i for i in order if dates.iloc[i] in allowed],dtype=int)
        est=estimator_factory(); est.fit(np.asarray(x_dev)[take],np.asarray(y_dev)[take]); pred=est.predict(x_valid); m=regression_metrics(y_valid,pred)
        rows.append({"train_fraction":float(fraction),"train_samples":len(take),**m})
    for row,prev in zip(rows[1:],rows[:-1]): row["relative_rmse_improvement"]=float((prev["rmse"]-row["rmse"])/prev["rmse"]) if prev["rmse"] else np.nan
    rows[0]["relative_rmse_improvement"]=np.nan
    plateau=next((row["train_fraction"] for row in rows[1:] if row["relative_rmse_improvement"]<.05),None)
    for row in rows: row["plateau_reached_at_fraction"]=plateau
    return pd.DataFrame(rows)

def seed_sensitivity(estimator_factory,x_train,y_train,x_valid,y_valid,seeds=(0,1,2)):
    rows=[]
    for seed in seeds:
        est=estimator_factory(seed); est.fit(x_train,y_train); metrics=regression_metrics(y_valid,est.predict(x_valid)); rows.append({"seed":int(seed),**metrics})
    return pd.DataFrame(rows)

def run_train_size_analysis(df,cfg,top_n=3,model_ids=None,fractions=(.1,.25,.5,.75,1.0)):
    root=Path(cfg["_config_dir"]); out=root/cfg["paths"]["results_root"]/cfg["project"]["group_id"]; rpath=out/"results.csv"; foldpath=out/"folds_temporal.json"
    if not rpath.exists() or not foldpath.exists(): raise FileNotFoundError("Run development benchmark and fold creation before train-size analysis")
    prior=pd.read_csv(rpath); selection=prior[(prior.dataset_split=="validation")&(prior.cv_scheme=="temporal_panel_expanding")&(prior.metric=="rmse")&(prior.niveau=="fold")]
    if "cv_purpose" in selection:
        selection=selection[selection.cv_purpose.isin(["development_selection","development"])]
    if "robustness_variant" in selection:
        selection=selection[selection.robustness_variant.isna()|selection.robustness_variant.isin(["","reference"])]
    ranking=selection.groupby("model_id").value.mean().sort_values()
    registry=model_registry(cfg["project"]["seed"]); models=model_ids or [m for m in ranking.index if m in registry][:top_n]
    folds=json.loads(foldpath.read_text(encoding="utf-8"))["folds"]; samples,features=supervised_frame(df,cfg); X=samples[features].to_numpy(float); y=samples.y.to_numpy(float); rows=samples.target_row.to_numpy(int); dates=pd.to_datetime(samples.target_timestamp,utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[ns]"); new=[]
    for fold in folds:
        tr=np.flatnonzero(np.isin(rows,fold["train_indices"])); va=np.flatnonzero(np.isin(rows,fold["validation_indices"]))
        if not len(tr) or not len(va): continue
        unique=np.sort(np.unique(dates[tr]))
        for model_id in models:
            configs=select_configuration(model_id,X[tr],y[tr],dates[tr],cfg["project"]["seed"],cfg); params=configs[0]["params"]
            full_rmse=None
            for fraction in fractions:
                allowed=set(unique[:max(1,int(np.ceil(len(unique)*fraction))) ]); subset=tr[np.isin(dates[tr],list(allowed))]; est=registry[model_id]()
                if params: est.set_params(**params)
                est.fit(X[subset],y[subset]); metrics=regression_metrics(y[va],est.predict(X[va]));
                if fraction==1.0: full_rmse=metrics["rmse"]
                for metric,value in metrics.items():
                    if metric=="n_samples": continue
                    new.append({"model_id":model_id,"model_group":"train_size_sensitivity","model_family":model_id,"dataset_split":"validation","cv_scheme":"temporal_panel_expanding","cv_purpose":"training_size_robustness","cv_fold":fold["fold"],"niveau":"fold","metric":metric,"is_primary":metric=="rmse","value":value,"n_samples":metrics["n_samples"],"train_size":len(subset),"horizon_step":1,"forecast_origin":"end_of_day_t","robustness_variant":f"train_{fraction:.0%}","delta_vs_reference":np.nan,"random_state":cfg["project"]["seed"],"hardware":platform.platform(),"n_features":len(features),"fold_file":"folds_temporal.json","notes":"Smaller chronological prefixes train on fewer development dates; identical next-block validation; final test remains untouched."})
            if full_rmse is not None:
                for row in new:
                    if row["model_id"]==model_id and row["cv_fold"]==fold["fold"] and row["metric"]=="rmse": row["delta_vs_reference"]=row["value"]-full_rmse
    if not new: raise ValueError("No train-size robustness scores were generated")
    all_rows=add_fold_summaries(pd.DataFrame(new,columns=RESULTS),seed=cfg["project"]["seed"])
    replace=(prior.model_group.eq("train_size_sensitivity")&prior.cv_scheme.eq("temporal_panel_expanding")&prior.model_id.isin(models))
    keep=prior[~replace]; combined=pd.concat([keep,all_rows],ignore_index=True); combined.to_csv(rpath,index=False,encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot=combined[(combined.model_group=="train_size_sensitivity")&(combined.niveau=="fold")&(combined.metric=="rmse")].copy(); plot["fraction"]=plot.robustness_variant.str.extract(r"train_(\d+)%").astype(float)/100
    fig,ax=plt.subplots(figsize=(7,4))
    for model,g in plot.groupby("model_id"): ax.plot(g.groupby("fraction").value.mean().index,g.groupby("fraction").value.mean().values,marker="o",label=model)
    ax.set(xlabel="Chronological development data fraction",ylabel="Validation RMSE (µg/m³)",title="Training-size sensitivity"); ax.legend(); fig.tight_layout(); figdir=out/"figures"; figdir.mkdir(parents=True,exist_ok=True); fig.savefig(figdir/"train_size_sensitivity.png",dpi=160); plt.close(fig)
    return {"models":models,"n_rows":len(new),"fractions":list(fractions)}
