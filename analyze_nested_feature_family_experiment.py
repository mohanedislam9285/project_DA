import itertools, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.multitest import multipletests
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results'/'16'; EXP=OUT/'feature_family_experiment'
raw=pd.read_csv(EXP/'feature_family_oof_predictions.csv')
sel=pd.read_csv(EXP/'selected_feature_family_by_fold.csv').set_index('cv_fold').selected_variant.to_dict()
nested=raw[raw.apply(lambda r:r.variant==sel[int(r.cv_fold)],axis=1)].copy(); nested['variant']='nested_inner_selected'; raw=pd.concat([raw,nested],ignore_index=True)
base=pd.read_csv(OUT/'reproducibility_audit'/'reloaded_temporal_oof_and_persistence.csv').set_index('sample_id').sort_index()
ids=set(base.index.astype(int))
for name,g in raw.groupby('variant'):
 z=g.set_index('sample_id').sort_index()
 if len(z)!=len(ids) or set(z.index.astype(int))!=ids or not np.allclose(z.y_true,base.loc[z.index,'y_true'],rtol=0,atol=1e-6): raise ValueError('Common OOF sample mismatch: '+name)
y=base.y_true.to_numpy(float); fold=base.cv_fold.to_numpy(int); folds=np.sort(np.unique(fold)); station=base.station_id.astype(str).to_numpy()
preds={'CNN_LSTM_OPTUNA_reference':base.cnn_lstm_prediction_reloaded.to_numpy(float),'persistence':base.persistence_prediction_origin_pm25.to_numpy(float)}
for name,g in raw.groupby('variant'): preds[name]=g.set_index('sample_id').sort_index().loc[base.index,'y_pred'].to_numpy(float)
B=20000; rng=np.random.default_rng(42016); stations=np.unique(station); code=pd.Categorical(station,categories=stations).codes; n_by=np.bincount(code,minlength=len(stations)).astype(float); draws=rng.multinomial(len(stations),np.full(len(stations),1/len(stations)),size=B); denom=draws@n_by
summary=[]; folds_out=[]
for name,p in preds.items():
 f_rm=[]; f_ma=[]
 for f in folds:
  ix=fold==f; rm=mean_squared_error(y[ix],p[ix])**.5; ma=mean_absolute_error(y[ix],p[ix]); f_rm.append(rm); f_ma.append(ma); folds_out.append({'model_id':name,'fold':int(f),'n':int(ix.sum()),'rmse':rm,'mae':ma,'r2':r2_score(y[ix],p[ix])})
 summary.append({'model_id':name,'n':len(y),'n_stations':len(stations),'pooled_rmse':mean_squared_error(y,p)**.5,'pooled_mae':mean_absolute_error(y,p),'pooled_r2':r2_score(y,p),'mean_fold_rmse':np.mean(f_rm),'sd_fold_rmse':np.std(f_rm,ddof=1),'mean_fold_mae':np.mean(f_ma),'sd_fold_mae':np.std(f_ma,ddof=1)})
pd.DataFrame(summary).to_csv(EXP/'feature_family_metric_summary.csv',index=False); pd.DataFrame(folds_out).to_csv(EXP/'feature_family_metrics_by_fold.csv',index=False)
def signp(d):
 obs=abs(d.mean()); vals=[abs(np.mean(d*np.asarray(s))) for s in itertools.product([-1.,1.],repeat=len(d))]; return np.mean(np.asarray(vals)>=obs-1e-12)
rows=[]; fam={'persistence':[],'CNN_LSTM_OPTUNA_reference':[]}
for name in [x for x in preds if x not in ('persistence','CNN_LSTM_OPTUNA_reference')]:
 p=preds[name]
 for comparator in fam:
  q=preds[comparator]; dr=[]; dm=[]
  for f in folds:
   ix=fold==f; dr.append(mean_squared_error(y[ix],p[ix])**.5-mean_squared_error(y[ix],q[ix])**.5); dm.append(mean_absolute_error(y[ix],p[ix])-mean_absolute_error(y[ix],q[ix]))
  dr=np.asarray(dr); dm=np.asarray(dm); boot_f=rng.choice(dr,size=(B,len(dr)),replace=True).mean(axis=1)
  ep=p-y; eq=q-y; sse_p=np.bincount(code,weights=ep**2,minlength=len(stations)); sse_q=np.bincount(code,weights=eq**2,minlength=len(stations)); sae_p=np.bincount(code,weights=np.abs(ep),minlength=len(stations)); sae_q=np.bincount(code,weights=np.abs(eq),minlength=len(stations))
  br=np.sqrt((draws@sse_p)/denom)-np.sqrt((draws@sse_q)/denom); bm=(draws@sae_p)/denom-(draws@sae_q)/denom
  row={'model_id':name,'comparator':comparator,'mean_fold_rmse_delta':dr.mean(),'fold_rmse_differences':json.dumps(dr.tolist()),'fold_bootstrap_ci_low_descriptive':np.quantile(boot_f,.025),'fold_bootstrap_ci_high_descriptive':np.quantile(boot_f,.975),'exact_wilcoxon_p_rmse_n5':wilcoxon(dr,method='exact').pvalue,'exact_sign_flip_p_rmse_n5':signp(dr),'mean_fold_mae_delta':dm.mean(),'exact_wilcoxon_p_mae_n5':wilcoxon(dm,method='exact').pvalue,'pooled_rmse_delta':mean_squared_error(y,p)**.5-mean_squared_error(y,q)**.5,'pooled_rmse_delta_cluster_ci_low':np.quantile(br,.025),'pooled_rmse_delta_cluster_ci_high':np.quantile(br,.975),'pooled_mae_delta':mean_absolute_error(y,p)-mean_absolute_error(y,q),'pooled_mae_delta_cluster_ci_low':np.quantile(bm,.025),'pooled_mae_delta_cluster_ci_high':np.quantile(bm,.975)}
  rows.append(row); fam[comparator].append(len(rows)-1)
comp=pd.DataFrame(rows); comp['holm_p_rmse']=np.nan; comp['holm_p_mae']=np.nan
for cmp,ix in fam.items():
 comp.loc[ix,'holm_p_rmse']=multipletests(comp.loc[ix,'exact_wilcoxon_p_rmse_n5'],method='holm')[1]; comp.loc[ix,'holm_p_mae']=multipletests(comp.loc[ix,'exact_wilcoxon_p_mae_n5'],method='holm')[1]
comp.to_csv(EXP/'feature_family_paired_comparisons.csv',index=False)
(EXP/'uncertainty_method.json').write_text(json.dumps({'bootstrap_replicates':B,'seed':42016,'cluster_unit':'station; conditional on observed folds','fold_tests':'exact two-sided Wilcoxon, five dependent folds','fold_bootstrap':'descriptive; resamples five dependent folds','multiplicity':'Holm across five feature procedures per comparator family, separate RMSE and MAE','future_test_labels_or_metrics_used':False},indent=2),encoding='utf-8')
print(pd.DataFrame(summary)[['model_id','pooled_rmse','pooled_mae','mean_fold_rmse','mean_fold_mae']].to_string(index=False)); print(comp[['model_id','comparator','mean_fold_rmse_delta','exact_wilcoxon_p_rmse_n5','holm_p_rmse','pooled_rmse_delta_cluster_ci_low','pooled_rmse_delta_cluster_ci_high']].to_string(index=False))
