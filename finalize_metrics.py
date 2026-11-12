from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pm25_project.features import build_origin_features
from pm25_project.sequences import make_sequences
from pm25_project.config import load_config

root=Path('.'); out=root/'results/16'
cfg,_=load_config('configs/project.yaml')
df=pd.read_csv('data/processed/cleaned_daily.csv',parse_dates=['timestamp']); df.timestamp=pd.to_datetime(df.timestamp,utc=True)
feature_table=build_origin_features(df,cfg)
features=list(cfg['features'].get('predictors',[]))+list(cfg['features'].get('calendar',[]))+['pm25_lag1']+[f"pm25_roll{w}_mean" for w in cfg['features'].get('rolling_windows_days',[3,7])]
X,y,keys=make_sequences(feature_table,features,lookback=7,horizon=1)
folds=json.loads((out/'folds_temporal.json').read_text())['folds']
pred=pd.read_csv(out/'predictions/cnn_lstm_temporal_oof.csv')
rows=[]
for f in folds:
 g=pred[pred.cv_fold.eq(f['fold'])]; yt=g.y_true.to_numpy(); yp=g.y_pred.to_numpy()
 tr=np.flatnonzero(keys.target_row.isin(f['train_indices']).to_numpy()); va=np.flatnonzero(keys.target_row.isin(f['validation_indices']).to_numpy())
 # Report directly observed trained sequence counts, not raw panel row counts.
 vals={'rmse':float(np.sqrt(mean_squared_error(yt,yp))),'mae':float(mean_absolute_error(yt,yp)),'r2':float(r2_score(yt,yp))}
 for metric,value in vals.items():
  rows.append({'model_id':'CNN_LSTM','model_group':'Group_16','model_family':'CNN-LSTM','dataset_split':'validation','cv_scheme':'temporal_panel_expanding','cv_purpose':'development_selection','cv_fold':int(f['fold']),'metric':metric,'value':value,'n_samples':len(g),'train_size':len(tr),'train_rows':f['train_n_samples'],'validation_rows':f['validation_n_samples'],'train_stations':f['train_n_stations'],'validation_stations':f['validation_n_stations'],'validation_start':f['validation_start'],'validation_end':f['validation_end'],'random_state':42+int(f['fold']),'fold_file':'folds_temporal.json','notes':'Metrics recomputed from saved development OOF predictions; model fit itself ran once per outer fold.'})
pd.DataFrame(rows).to_csv(out/'temporal_cv_results.csv',index=False)
# Append actual fold results without removing any pre-existing result rows.
old=pd.read_csv(out/'results.csv')
old=old[~(old.model_id.eq('CNN_LSTM') & old.cv_scheme.eq('temporal_panel_expanding') & old.cv_purpose.eq('development_selection'))]
old=old[~(old.model_id.eq('CNN_LSTM') & old.cv_scheme.eq('spatial_group_kfold') & old.cv_purpose.eq('development_generalization'))]
new=pd.DataFrame(rows)
new['niveau']='fold'; new['is_primary']=new.metric.eq('rmse'); new['horizon_step']=1; new['forecast_origin']='end_of_day_t'; new['n_features']=7
for c in old.columns:
 if c not in new: new[c]=np.nan
new=new[old.columns]
spatial=pd.read_csv(out/'spatial_cv_results.csv')
spatial['niveau']='fold'; spatial['is_primary']=spatial.metric.eq('rmse'); spatial['horizon_step']=1; spatial['forecast_origin']='end_of_day_t'; spatial['random_state']=42; spatial['fold_file']='folds_spatial.json'; spatial['cv_purpose']='development_generalization'; spatial['model_group']='Group_16'; spatial['model_family']='CNN-LSTM'
for c in old.columns:
 if c not in spatial: spatial[c]=np.nan
spatial=spatial[old.columns]
old=pd.concat([old,new,spatial],ignore_index=True); old.to_csv(out/'results.csv',index=False)
# Development fold uncertainty: resample five dependent temporal fold scores; interval is descriptive only.
def interval(values,seed):
 rng=np.random.default_rng(seed); a=np.asarray(values,float); b=np.mean(rng.choice(a,(50000,len(a)),replace=True),axis=1); return np.quantile(b,[.025,.975])
summ=[]
for scheme,file in [('temporal_panel_expanding','temporal_cv_results.csv'),('spatial_group_kfold','spatial_cv_results.csv')]:
 d=pd.read_csv(out/file); d=d[d.metric.eq('rmse')]
 for met in ['rmse','mae','r2']:
  q=pd.read_csv(out/file); q=q[q.metric.eq(met)]; v=q.value.to_numpy(); lo,hi=interval(v,42+(0 if scheme.startswith('temporal') else 100))
  predname='cnn_lstm_temporal_oof.csv' if scheme.startswith('temporal') else 'cnn_lstm_spatial_oof.csv'
  pp=pd.read_csv(out/'predictions'/predname)
  pooled={'rmse':float(np.sqrt(mean_squared_error(pp.y_true,pp.y_pred))),'mae':float(mean_absolute_error(pp.y_true,pp.y_pred)),'r2':float(r2_score(pp.y_true,pp.y_pred))}[met]
  summ.append({'cv_scheme':scheme,'metric':met,'fold_mean':float(v.mean()),'fold_sd':float(v.std(ddof=1)),'bootstrap_95_ci_lower':float(lo),'bootstrap_95_ci_upper':float(hi),'pooled_oof_metric':pooled,'n_folds':len(v),'total_oof_predictions':int(q.n_samples.sum())})
pd.DataFrame(summ).to_csv(out/'development_cv_summary.csv',index=False)
# Correct group metadata in development-only outputs.
for name in ['hyperparameters.csv','feature_importance.csv','feature_importance_spatial.csv','experiments.csv','seed_sensitivity.csv','seed_sensitivity_summary.csv','tables/cnn_lstm_seed_sensitivity_folds.csv','tables/cnn_lstm_seed_sensitivity_configs.csv']:
 p=out/name
 if p.is_file():
  d=pd.read_csv(p)
  if 'model_group' in d: d.loc[d.model_id.astype(str).str.upper().eq('CNN_LSTM'),'model_group']='Group_16'; d.to_csv(p,index=False)
for name in ['cnn_lstm_temporal_oof.csv','cnn_lstm_spatial_oof.csv','cnn_lstm_seed_oof.csv']:
 p=out/'predictions'/name
 if p.is_file():
  pass
print(pd.DataFrame(summ).to_string(index=False))
print('temporal folds',pd.DataFrame(rows).query("metric=='rmse'")[['cv_fold','value','n_samples','train_size','train_rows','train_stations','validation_stations']].to_string(index=False))
