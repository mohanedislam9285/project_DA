"""Analyze lagged-environmental CNN-LSTM OOF predictions; no refitting or test data."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
ROOT=Path(__file__).resolve().parent; EXP=ROOT/'results'/'16'/'issue_time_lagged_exogenous_experiment'
df=pd.read_csv(EXP/'oof_predictions.csv'); variants=list(df.variant.drop_duplicates()); ref='target_lag_roll3'
# Keep only target rows present in every family, then enforce exactly-once OOF scoring.
groups={v:g.sort_values('sample_id').reset_index(drop=True) for v,g in df.groupby('variant')}
ids=set(groups[ref].sample_id)
if any(set(g.sample_id)!=ids for g in groups.values()): raise RuntimeError('OOF target sets differ')
for v,g in groups.items():
 if g.sample_id.duplicated().any(): raise RuntimeError(f'duplicate OOF row in {v}')
y=groups[ref].y_true.to_numpy();st=groups[ref].station_id.to_numpy();fold=groups[ref].cv_fold.to_numpy();persist=groups[ref].persistence.to_numpy()
if not all(np.allclose(groups[v].y_true,y) for v in variants):raise RuntimeError('Target arrays differ')

def metric(a,b):return {'rmse':float(np.sqrt(np.mean((a-b)**2))),'mae':float(np.mean(np.abs(a-b))),'r2':float(r2_score(a,b))}
pred={v:groups[v].y_pred.to_numpy() for v in variants};pred['persistence']=persist
# OOF selected procedure takes only the family chosen by inner validation for each outer fold.
sel=np.full(len(y),np.nan)
for v,g in groups.items():
 m=g.selected_by_inner_validation.astype(bool).to_numpy();sel[m]=g.loc[m,'y_pred'].to_numpy()
if np.isnan(sel).any():raise RuntimeError('Some or multiple selected OOF predictions missing')
pred['nested_inner_selected']=sel
metrics=[];folds=[]
for name,p in pred.items():
 m=metric(y,p);metrics.append({'model_id':name,'n':len(y),'n_stations':len(np.unique(st)),**m})
 for k in sorted(np.unique(fold)):
  ix=fold==k;folds.append({'model_id':name,'fold':int(k),'n':int(ix.sum()),**metric(y[ix],p[ix])})
# Station-cluster bootstrap conditioned on the observed folds, using cluster sufficient statistics.
unique=np.unique(st);loc={s:i for i,s in enumerate(unique)};si=np.array([loc[x] for x in st]);rng=np.random.default_rng(42016);B=20000
count=np.bincount(si,minlength=len(unique));
agg={}
for name,p in pred.items():
 agg[name]=(np.bincount(si,weights=(y-p)**2,minlength=len(unique)),np.bincount(si,weights=np.abs(y-p),minlength=len(unique)))
boots={}
for comparator in ['persistence',ref]:
 for name in pred:
  if name==comparator:continue
  sq_m,ab_m=agg[name];sq_c,ab_c=agg[comparator];dr=[];da=[]
  for _ in range(B):
   draw=rng.integers(0,len(unique),size=len(unique));n=count[draw].sum()
   dr.append(np.sqrt(sq_m[draw].sum()/n)-np.sqrt(sq_c[draw].sum()/n));da.append(ab_m[draw].sum()/n-ab_c[draw].sum()/n)
  boots[(name,comparator)]=(float(np.quantile(dr,.025)),float(np.quantile(dr,.975)),float(np.quantile(da,.025)),float(np.quantile(da,.975)))
comparisons=[]
for comparator in ['persistence',ref]:
 tested=[v for v in variants if v!=comparator]
 raw_rmse=[];raw_mae=[];temp=[]
 for name in tested:
  deltas=[]
  for k in sorted(np.unique(fold)):
   ix=fold==k;deltas.append(metric(y[ix],pred[name][ix])['rmse']-metric(y[ix],pred[comparator][ix])['rmse'])
  try: pr=float(wilcoxon(deltas,zero_method='wilcox',alternative='two-sided',method='exact').pvalue)
  except ValueError: pr=1.0
  mae_d=[]
  for k in sorted(np.unique(fold)):
   ix=fold==k;mae_d.append(metric(y[ix],pred[name][ix])['mae']-metric(y[ix],pred[comparator][ix])['mae'])
  try: pm=float(wilcoxon(mae_d,zero_method='wilcox',alternative='two-sided',method='exact').pvalue)
  except ValueError:pm=1.0
  raw_rmse.append(pr);raw_mae.append(pm);lo,hi,mlo,mhi=boots[(name,comparator)];mm=metric(y,pred[name]);cc=metric(y,pred[comparator])
  temp.append({'model_id':name,'comparator':comparator,'mean_fold_rmse_delta':float(np.mean(deltas)),'fold_rmse_deltas':json.dumps(deltas),'exact_wilcoxon_p_rmse_n5':pr,'mean_fold_mae_delta':float(np.mean(mae_d)),'exact_wilcoxon_p_mae_n5':pm,'pooled_rmse_delta':mm['rmse']-cc['rmse'],'cluster_ci_rmse_low':lo,'cluster_ci_rmse_high':hi,'pooled_mae_delta':mm['mae']-cc['mae'],'cluster_ci_mae_low':mlo,'cluster_ci_mae_high':mhi})
 adj_r=multipletests(raw_rmse,method='holm')[1];adj_m=multipletests(raw_mae,method='holm')[1]
 for i,row in enumerate(temp):row['holm_p_rmse_family']=float(adj_r[i]);row['holm_p_mae_family']=float(adj_m[i]);comparisons.append(row)
# Fixed final-development selected procedure against persistence (five outer-fold paired scores).
for c in ['persistence',ref]:
 for m in ['nested_inner_selected']:
  # record as separate diagnostic, no multiplicity correction since procedure is nested but five-fold inference remains low power
  d=[];a=[]
  for k in sorted(np.unique(fold)):
   ix=fold==k;d.append(metric(y[ix],pred[m][ix])['rmse']-metric(y[ix],pred[c][ix])['rmse']);a.append(metric(y[ix],pred[m][ix])['mae']-metric(y[ix],pred[c][ix])['mae'])
  try:pr=float(wilcoxon(d,alternative='two-sided',method='exact').pvalue)
  except ValueError:pr=1.0
  lo,hi,mlo,mhi=boots[(m,c)];comparisons.append({'model_id':m,'comparator':c,'mean_fold_rmse_delta':float(np.mean(d)),'fold_rmse_deltas':json.dumps(d),'exact_wilcoxon_p_rmse_n5':pr,'mean_fold_mae_delta':float(np.mean(a)),'pooled_rmse_delta':metric(y,pred[m])['rmse']-metric(y,pred[c])['rmse'],'cluster_ci_rmse_low':lo,'cluster_ci_rmse_high':hi,'pooled_mae_delta':metric(y,pred[m])['mae']-metric(y,pred[c])['mae'],'cluster_ci_mae_low':mlo,'cluster_ci_mae_high':mhi,'holm_p_rmse_family':'nested single procedure; no adjustment'})
pd.DataFrame(metrics).to_csv(EXP/'metric_summary.csv',index=False);pd.DataFrame(folds).to_csv(EXP/'metrics_by_fold.csv',index=False);pd.DataFrame(comparisons).to_csv(EXP/'paired_comparisons.csv',index=False)
method={'bootstrap_replicates':B,'seed':42016,'cluster_unit':'station; conditional on observed temporal folds; does not account for temporal dependence','fold_test':'exact two-sided Wilcoxon on five dependent outer folds','multiple_comparison':'Holm correction across fixed lagged-exogenous variants for each comparator and metric','future_test_labels_or_metrics_used':False,'note':'Intervals and five-fold tests answer different conditional questions; statistical significance does not verify operational availability.'}
(EXP/'uncertainty_method.json').write_text(json.dumps(method,indent=2),encoding='utf-8')
print(pd.DataFrame(metrics).to_string(index=False));print(pd.DataFrame(comparisons).to_string(index=False))
