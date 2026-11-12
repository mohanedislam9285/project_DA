"""Fold-local validation permutation importance helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from .schemas import RANKING_STABILITY

def tabular_permutation_importance(estimator,x_valid,y_valid,feature_names,seed=42,n_repeats=20):
    result=permutation_importance(estimator,x_valid,y_valid,scoring="neg_root_mean_squared_error",n_repeats=n_repeats,random_state=seed,n_jobs=-1)
    out=pd.DataFrame({"feature":feature_names,"importance":result.importances_mean,"importance_std":result.importances_std})
    out["rank"]=out.importance.rank(method="min",ascending=False).astype(int)
    return out.sort_values("rank").reset_index(drop=True)

def sequence_channel_permutation(model,x_valid,y_valid,feature_names,seed=42,n_repeats=20):
    """Permute channels across examples while preserving each channel's full temporal shape."""
    rng=np.random.default_rng(seed); base=np.sqrt(np.mean((np.asarray(y_valid)-model.predict(x_valid,verbose=0).reshape(-1))**2)); rows=[]
    for j,name in enumerate(feature_names):
        deltas=[]
        for _ in range(n_repeats):
            perm=rng.permutation(len(x_valid)); xp=x_valid.copy(); xp[:,:,j]=x_valid[perm,:,j]
            yp=model.predict(xp,verbose=0).reshape(-1); deltas.append(np.sqrt(np.mean((y_valid-yp)**2))-base)
        rows.append({"feature":name,"importance":float(np.mean(deltas)),"importance_std":float(np.std(deltas,ddof=1)) if len(deltas)>1 else 0.0})
    out=pd.DataFrame(rows); out["rank"]=out.importance.rank(method="min",ascending=False).astype(int); return out.sort_values("rank").reset_index(drop=True)

def fold_ranking_stability(importance:pd.DataFrame,top_n=5)->pd.DataFrame:
    from scipy.stats import spearmanr
    rows=[]
    for model,group in importance.groupby("model_id",sort=True):
        by_fold={f:g.sort_values("rank").feature.astype(str).tolist() for f,g in group.groupby("cv_fold")}
        folds=sorted(by_fold)
        for i,a in enumerate(folds):
            for b in folds[i+1:]:
                features=sorted(set(by_fold[a])|set(by_fold[b])); ra=[by_fold[a].index(x)+1 if x in by_fold[a] else len(by_fold[a])+1 for x in features]; rb=[by_fold[b].index(x)+1 if x in by_fold[b] else len(by_fold[b])+1 for x in features]
                rho=float(spearmanr(ra,rb).statistic) if len(features)>1 else 1.0; sa=set(by_fold[a][:top_n]); sb=set(by_fold[b][:top_n]); jac=len(sa&sb)/len(sa|sb) if sa|sb else 1.0
                rows.append({"model_group":"comparators","model_a":model,"model_b":model,"comparison_axis":f"fold_{a}_vs_{b}","spearman_rho":rho,"jaccard_top5":jac,"rho_threshold_met":rho>.8,"jaccard_threshold_met":np.nan,"notes":"Within-scheme temporal fold stability; not a cross-scheme comparison."})
    return pd.DataFrame(rows,columns=RANKING_STABILITY)
