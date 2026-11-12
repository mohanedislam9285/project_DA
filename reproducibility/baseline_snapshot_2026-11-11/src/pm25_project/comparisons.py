from __future__ import annotations
import numpy as np
from .metrics import paired_bootstrap_ci
from .schemas import COMPARISONS

def paired_fold_comparison(model_a,model_b,seed=42,n_perm=10000):
    """Compare same-fold RMSE vectors on equal-size validation cohorts."""
    a=np.asarray(model_a,dtype=float); b=np.asarray(model_b,dtype=float)
    if a.shape!=b.shape: raise ValueError("Model score vectors must contain the same paired folds")
    ok=np.isfinite(a)&np.isfinite(b); d=a[ok]-b[ok]
    if not len(d): raise ValueError("No finite paired folds")
    ci=paired_bootstrap_ci(d,seed=seed); rng=np.random.default_rng(seed); signs=rng.choice([-1,1],size=(n_perm,len(d))); p=float((1+(np.abs((signs*d).mean(axis=1))>=abs(d.mean())).sum())/(n_perm+1))
    try:
        from scipy.stats import rankdata, wilcoxon
        wp=float(wilcoxon(d,zero_method="wilcox",alternative="two-sided",method="auto").pvalue) if np.any(d) else 1.0
    except ImportError: wp=float("nan")
    nonzero=d[d!=0]
    ranks=rankdata(np.abs(nonzero)) if len(nonzero) else np.asarray([])
    positive=float(ranks[nonzero>0].sum()) if len(nonzero) else 0.0
    negative=float(ranks[nonzero<0].sum()) if len(nonzero) else 0.0
    rank_biserial=(positive-negative)/(positive+negative) if positive+negative else 0.0
    return {"mean_delta_a_minus_b":float(d.mean()),"ci_lower":ci[0],"ci_upper":ci[1],"permutation_p":p,"wilcoxon_p":wp,"wilcoxon_rank_biserial_r":float(rank_biserial),"n_paired_folds":int(len(d)),"note":f"{len(d)} dependent folds; interpret inferential results cautiously."}

def comparison_table(results,seed=42,n_perm=10000):
    import itertools
    scores=results[(results.metric=="rmse")&(results.niveau=="fold")&(results.dataset_split=="validation")]
    if "cv_purpose" in scores:
        scores=scores[scores.cv_purpose.ne("training_size_robustness")]
    if "robustness_variant" in scores:
        scores=scores[scores.robustness_variant.isna()|scores.robustness_variant.isin(["","reference"])]
    if scores.empty: return __import__("pandas").DataFrame(columns=COMPARISONS)
    groups=scores.groupby("cv_scheme",sort=True); pairs=[]; pair_index=0
    for scheme,group in groups:
      piv=group.pivot_table(index="cv_fold",columns="model_id",values="value",aggfunc="mean")
      sample_counts=group.pivot_table(index="cv_fold",columns="model_id",values="n_samples",aggfunc="first") if "n_samples" in group else None
      for a,b in itertools.combinations(piv.columns,2):
        aligned=piv[[a,b]].dropna()
        if sample_counts is not None:
            counts=sample_counts[[a,b]].dropna()
            aligned=aligned.loc[counts.index[counts[a].eq(counts[b])].intersection(aligned.index)]
        if len(aligned)<2: continue
        pair_index+=1; stat=paired_fold_comparison(aligned[a].to_numpy(),aligned[b].to_numpy(),seed=seed+pair_index,n_perm=n_perm); delta=aligned[a].to_numpy()-aligned[b].to_numpy(); sd=float(np.std(delta,ddof=1))
        pairs.append({"model_id":f"{a}_vs_{b}","model_group":"paired_comparison","comparison_id":f"{scheme}_rmse_{a}_{b}","binome_id":f"{a}|{b}","model_a":a,"model_b":b,"metric":"rmse","is_primary_metric":True,"comparison_type":scheme,"horizon_step":1,"train_size":np.nan,"mean_a":float(aligned[a].mean()),"mean_b":float(aligned[b].mean()),"delta":stat["mean_delta_a_minus_b"],"delta_ci_lower":stat["ci_lower"],"delta_ci_upper":stat["ci_upper"],"effect_size_type":"Wilcoxon matched-pairs rank-biserial r","effect_size":stat["wilcoxon_rank_biserial_r"],"p_value":stat["wilcoxon_p"],"p_value_corrected":np.nan,"correction_method":"Holm-Bonferroni","test_used":"Wilcoxon signed-rank; paired sign permutation supplemental","n_permutations":n_perm,"bootstrap_n":10000,"n_paired_folds":stat["n_paired_folds"],"significant":False,"practically_relevant":False,"practically_relevant_threshold":np.nan,"is_baseline_comparison":a.startswith("persistence") or b.startswith("persistence") or a.startswith("seasonal") or b.startswith("seasonal"),"notes":stat["note"]})
    if pairs:
        order=sorted(range(len(pairs)),key=lambda i:pairs[i]["p_value"] if np.isfinite(pairs[i]["p_value"]) else 1.0); m=len(order); adjusted=np.ones(m)
        for rank,idx in enumerate(order): adjusted[rank]=min(1.0,(m-rank)*(pairs[idx]["p_value"] if np.isfinite(pairs[idx]["p_value"]) else 1.0))
        for rank in range(1,m): adjusted[rank]=max(adjusted[rank],adjusted[rank-1])
        for rank,idx in enumerate(order): pairs[idx]["p_value_corrected"]=adjusted[rank]; pairs[idx]["significant"]=bool(adjusted[rank]<.05)
    return __import__("pandas").DataFrame(pairs,columns=COMPARISONS)
