from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def regression_metrics(y_true,y_pred):
    y=np.asarray(y_true,dtype=float); p=np.asarray(y_pred,dtype=float); ok=np.isfinite(y)&np.isfinite(p)
    if not ok.any(): raise ValueError("No finite paired targets and predictions")
    return {"rmse":float(np.sqrt(mean_squared_error(y[ok],p[ok]))),"mae":float(mean_absolute_error(y[ok],p[ok])),"r2":float(r2_score(y[ok],p[ok])) if ok.sum()>1 else float("nan"),"n_samples":int(ok.sum())}

def paired_bootstrap_ci(differences,seed=42,n_boot=10000,confidence=.95):
    x=np.asarray(differences,dtype=float); x=x[np.isfinite(x)]
    if not len(x): return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); means=np.mean(rng.choice(x,(n_boot,len(x)),replace=True),axis=1); a=(1-confidence)/2
    return float(np.quantile(means,a)),float(np.quantile(means,1-a))
