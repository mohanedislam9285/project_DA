"""Training-only chronological configuration screening with a fixed finite candidate list."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from sklearn.metrics import mean_squared_error
from .models.model_registry import model_registry


def candidate_configs(model_id:str,seed:int=42,config:dict|None=None):
    """Small predeclared robustness candidates; extend only in search_spaces.yaml."""
    base={"Lasso":[{"lasso__alpha":a} for a in (.01,.1,1,10)],"ElasticNet":[{"elasticnet__alpha":a,"elasticnet__l1_ratio":r} for a,r in ((.01,.5),(.1,.5),(1,.5),(1,.9))],"Ridge":[{"ridge__alpha":a} for a in (.1,1,10,100)],"Huber":[{"huberregressor__epsilon":e} for e in (1.1,1.35,1.5,2.)],"KNN":[{"kneighborsregressor__n_neighbors":k,"kneighborsregressor__weights":w} for k,w in ((3,"distance"),(5,"distance"),(10,"distance"),(15,"uniform"))],"SVR_RBF":[{"svr__C":c,"svr__epsilon":e} for c,e in ((.1,.1),(1,.1),(10,.1),(10,.5))],"ExtraTrees":[{"extratreesregressor__n_estimators":n,"extratreesregressor__max_depth":d} for n,d in ((100,None),(200,20),(500,30))],"GradientBoosting":[{"gradientboostingregressor__n_estimators":n,"gradientboostingregressor__learning_rate":lr,"gradientboostingregressor__max_depth":3} for n,lr in ((100,.1),(200,.05),(500,.01))],"MLP":[{"mlpregressor__hidden_layer_sizes":s,"mlpregressor__learning_rate_init":lr} for s,lr in (((64,32),.001),((128,64),.0005),((64,32),.0001))]}
    configured=(config or {}).get("_search_spaces",{}).get("implemented_candidates",{}).get(model_id)
    return deepcopy(configured if configured is not None else base.get(model_id,[{}]))

def chronological_inner_indices(sample_dates,validation_fraction=.2,gap_days=1):
    dates=np.asarray(sample_dates,dtype="datetime64[ns]"); unique=np.unique(dates); cut=max(1,int(np.floor(len(unique)*(1-validation_fraction))))
    if cut<=gap_days or cut>=len(unique): raise ValueError("Not enough unique dates for chronological inner validation")
    train_end=unique[cut-gap_days-1]; val_start=unique[cut]
    tr=np.flatnonzero(dates<=train_end); va=np.flatnonzero(dates>=val_start)
    if not len(tr) or not len(va): raise ValueError("Inner chronological split produced an empty partition")
    return tr,va

def select_configuration(model_id,x,y,dates,seed=42,config=None):
    from .metrics import regression_metrics
    registry=model_registry(seed); candidates=candidate_configs(model_id,seed,config); tr,va=chronological_inner_indices(dates)
    scores=[]
    for rank,params in enumerate(candidates,1):
        estimator=registry[model_id]();
        try:
            if params: estimator.set_params(**params)
            estimator.fit(x[tr],y[tr]); pred=estimator.predict(x[va]); rmse=regression_metrics(y[va],pred)["rmse"]
            scores.append({"rank":rank,"params":params,"rmse":rmse,"estimator":estimator})
        except (ValueError,TypeError) as exc:
            scores.append({"rank":rank,"params":params,"rmse":float("inf"),"error":str(exc),"estimator":None})
    scores.sort(key=lambda z:z["rmse"])
    if not scores or not np.isfinite(scores[0]["rmse"]): raise RuntimeError(f"All inner configurations failed for {model_id}")
    return scores
