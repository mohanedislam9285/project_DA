"""Nested CNN-LSTM test of cross-station PM2.5 at the forecast origin.

All neighbor predictors are measured on the sequence origin day t and predict
t+1; the self station is excluded. This uses the same availability assumption
as the persistence baseline and the existing station PM2.5 at t.
"""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np,pandas as pd,tensorflow as tf
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from pm25_project.config import load_config
from pm25_project.features import build_origin_features
from pm25_project.models.cnn_lstm import build_cnn_lstm
from pm25_project.preprocessing import SequencePreprocessor
from pm25_project.sequences import make_sequences
from pm25_project.tuning import chronological_inner_indices
ROOT=Path(__file__).resolve().parent;RES=ROOT/'results'/'16';EXP=RES/'spatial_predictor_experiment';(EXP/'models').mkdir(parents=True,exist_ok=True);(EXP/'histories').mkdir(parents=True,exist_ok=True)
lock=json.loads((RES/'future_test.json').read_text(encoding='utf-8'));devset=set(map(int,lock['development_indices']))
path=ROOT/'data'/'processed'/'cleaned_daily.csv'
df=pd.read_csv(path,skiprows=lambda line:line>0 and line-1 not in devset,parse_dates=['timestamp']);df['timestamp']=pd.to_datetime(df.timestamp,utc=True)
if set(df.clean_index.astype(int))!=devset:raise RuntimeError('Development row selection mismatch')
if df.timestamp.max()>pd.to_datetime(lock['development_end'],utc=True):raise RuntimeError('Dev rows crossed cutoff')
cfg,_=load_config(ROOT/'configs'/'project.yaml');ft=build_origin_features(df,cfg)
# Daily cross-section: neighbors are other stations observed at the origin date.
# Compute great-circle distances per date, with no target t+1 values involved.
for radius in [25,50,100]:ft[f'neighbor_pm25_mean_{radius}km']=np.nan;ft[f'neighbor_pm25_count_{radius}km']=0
coverage={r:{'rows_with_neighbor':0,'rows_total':0,'neighbor_observations':0} for r in [25,50,100]}
for _,g in ft.groupby('timestamp',sort=False):
    idx=g.index.to_numpy();n=len(g)
    for radius in [25,50,100]:coverage[radius]['rows_total']+=n
    if n<2:continue
    xy=np.deg2rad(g[['latitude','longitude']].to_numpy(float));lat=xy[:,0,None];lon=xy[:,1,None];lat2=xy[None,:,0];lon2=xy[None,:,1]
    a=np.sin((lat-lat2)/2)**2+np.cos(lat)*np.cos(lat2)*np.sin((lon-lon2)/2)**2
    dist=2*6371.0088*np.arcsin(np.sqrt(np.clip(a,0,1)));np.fill_diagonal(dist,np.inf)
    values=g.pm25.to_numpy(float)
    for radius in [25,50,100]:
        mask=dist<=radius;cnt=mask.sum(axis=1);means=np.divide((mask*values[None,:]).sum(axis=1),cnt,out=np.full(n,np.nan),where=cnt>0)
        ft.loc[idx,f'neighbor_pm25_mean_{radius}km']=means;ft.loc[idx,f'neighbor_pm25_count_{radius}km']=cnt
        coverage[radius]['rows_with_neighbor']+=int((cnt>0).sum());coverage[radius]['neighbor_observations']+=int(cnt.sum())
base=['pm25_lag1','pm25_roll3_mean'];variants={
 'target_lag_roll3':base,
 'target_plus_neighbor_25km':base+['neighbor_pm25_mean_25km','neighbor_pm25_count_25km'],
 'target_plus_neighbor_50km':base+['neighbor_pm25_mean_50km','neighbor_pm25_count_50km'],
 'target_plus_neighbor_100km':base+['neighbor_pm25_mean_100km','neighbor_pm25_count_100km'],
}
arrays={};common=None
for name,cols in variants.items():
 X,y,keys=make_sequences(ft,cols,lookback=7,horizon=1);ids=set(keys.target_row.astype(int));common=ids if common is None else common
 if ids!=common:raise RuntimeError('Feature groups do not share identical OOF target candidates')
 arrays[name]=(X,y,keys,cols)
folds=json.loads((RES/'folds_temporal.json').read_text(encoding='utf-8'))['folds'];inner_rows=[];outer_rows=[];pred_rows=[];selected=[]
for fold in folds:
 k=int(fold['fold']);tuned=json.loads((RES/'models'/f'cnn_lstm_optuna_temporal_fold_{k}_config.json').read_text(encoding='utf-8'));hp={key:tuned[key] for key in ('filters','kernel_size','lstm_units','dense_units','dropout','learning_rate','loss')};seed=int(tuned['seed']);batch=int(cfg['model']['cnn_lstm'].get('batch_size',64));idxs={};scores={};epochs={}
 for name,(X,y,keys,cols) in arrays.items():
  target=keys.target_row.to_numpy(int);tr=np.flatnonzero(np.isin(target,np.asarray(fold['train_indices'],int)));va=np.flatnonzero(np.isin(target,np.asarray(fold['validation_indices'],int)));idxs[name]=(tr,va)
  dates=pd.to_datetime(keys.iloc[tr].target_timestamp,utc=True).dt.tz_localize(None).to_numpy(dtype='datetime64[ns]');itr,iva=chronological_inner_indices(dates);prep=SequencePreprocessor().fit(X[tr[itr]]);xt=prep.transform(X[tr[itr]]);xv=prep.transform(X[tr[iva]])
  tf.keras.backend.clear_session();model=build_cnn_lstm(7,xt.shape[-1],{**hp,'seed':seed,'pool_size':2});callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=3,min_lr=1e-5,verbose=0),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True,verbose=0)];hist=model.fit(xt,y[tr[itr]],validation_data=(xv,y[tr[iva]]),epochs=100,batch_size=batch,callbacks=callbacks,verbose=0);pv=model.predict(xv,verbose=0).reshape(-1);scores[name]=float(mean_squared_error(y[tr[iva]],pv)**.5);epochs[name]=int(np.argmin(hist.history['val_loss'])+1);inner_rows.append({'cv_fold':k,'variant':name,'inner_rmse':scores[name],'inner_mae':float(mean_absolute_error(y[tr[iva]],pv)),'inner_train_n':len(itr),'inner_validation_n':len(iva),'best_epoch':epochs[name],'features':json.dumps(cols)});pd.DataFrame(hist.history).to_csv(EXP/'histories'/f'fold{k}_{name}_inner.csv',index=False)
 chosen=min(variants,key=lambda name:(scores[name],name));selected.append({'cv_fold':k,'selected_variant':chosen,'inner_rmse':scores[chosen],'best_epoch':epochs[chosen]})
 for name,(X,y,keys,cols) in arrays.items():
  tr,va=idxs[name];prep=SequencePreprocessor().fit(X[tr]);xt=prep.transform(X[tr]);xv=prep.transform(X[va]);tf.keras.backend.clear_session();model=build_cnn_lstm(7,xt.shape[-1],{**hp,'seed':seed,'pool_size':2});start=time.perf_counter();hist=model.fit(xt,y[tr],epochs=epochs[name],batch_size=batch,verbose=0);elapsed=time.perf_counter()-start;pv=model.predict(xv,verbose=0).reshape(-1);persist=X[va,-1,cols.index('pm25_lag1')]
  for j,seq in enumerate(va):
   key=keys.iloc[seq];pred_rows.append({'sample_id':int(key.target_row),'station_id':str(key.station_id),'cv_fold':k,'variant':name,'selected_by_inner_validation':name==chosen,'origin_timestamp':pd.Timestamp(key.origin_timestamp).isoformat(),'target_timestamp':pd.Timestamp(key.target_timestamp).isoformat(),'y_true':float(y[seq]),'y_pred':float(pv[j]),'persistence':float(persist[j])})
  model.save(EXP/'models'/f'fold{k}_{name}.keras');pd.DataFrame(hist.history).to_csv(EXP/'histories'/f'fold{k}_{name}_outer.csv',index=False);tf.keras.backend.clear_session()
pd.DataFrame(inner_rows).to_csv(EXP/'inner_scores.csv',index=False);pd.DataFrame(selected).to_csv(EXP/'selected_variant_by_fold.csv',index=False);pd.DataFrame(pred_rows).to_csv(EXP/'oof_predictions.csv',index=False)
meta={'experiment':'nested cross-station PM2.5 origin-feature sensitivity','scope':'development temporal folds only','common_oof_n':len(common),'origin':'end of day t','target':'station PM2.5 at t+1','neighbor_inputs':'other stations observed on same date t; self excluded; great-circle radius','availability_assumption':'neighbor PM2.5 on origin day is available under the same timing assumption as own-station persistence; source timestamps do not record delivery times','distance_coverage':{str(r):{**v,'fraction':v['rows_with_neighbor']/max(1,v['rows_total'])} for r,v in coverage.items()},'selection':'inner chronological RMSE only; same existing fold-specific Optuna parameters; fold-local preprocessing','future_test_labels_or_metrics_used':False,'outer_validation_used_for_selection':False}
(EXP/'experiment_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps({'selected':selected,'experiment':str(EXP),'coverage':meta['distance_coverage']},indent=2))
