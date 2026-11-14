from __future__ import annotations
import numpy as np
import pandas as pd
from .schemas import RESULTS

def add_fold_summaries(results:pd.DataFrame,seed=42,n_boot=10000)->pd.DataFrame:
    """Add mean/std and fold bootstrap intervals without treating them as independent high-power evidence."""
    fold=results[(results.dataset_split=="validation")&(results.niveau=="fold")]
    summaries=[]; rng=np.random.default_rng(seed)
    for (model,scheme,metric),g in fold.groupby(["model_id","cv_scheme","metric"],sort=True):
        v=g.value.dropna().to_numpy(dtype=float)
        if not len(v): continue
        boot=np.mean(rng.choice(v,(n_boot,len(v)),replace=True),axis=1); row=g.iloc[0].to_dict(); row.update(niveau="global",value=float(v.mean()),std=float(v.std(ddof=1)) if len(v)>1 else np.nan,ci_lower=float(np.quantile(boot,.025)),ci_upper=float(np.quantile(boot,.975)),n_samples=int(g.n_samples.fillna(0).sum()),cv_fold=np.nan,notes=f"Mean and 95% bootstrap interval over {len(v)} outer folds; folds are dependent and inference has low power.")
        summaries.append(row)
    return pd.concat([results,pd.DataFrame(summaries,columns=RESULTS)],ignore_index=True) if summaries else results
