"""Nested development-only test of one-day-lagged environmental inputs.

Environmental observations are shifted by one station-day before sequence
construction. This guarantees source dates no later than origin t-1 for an
origin t -> target t+1 forecast. It does not prove the source delivery latency.
"""
from __future__ import annotations
import json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd,tensorflow as tf
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from pm25_project.config import load_config
from pm25_project.features import build_origin_features
from pm25_project.models.cnn_lstm import build_cnn_lstm
from pm25_project.preprocessing import SequencePreprocessor
from pm25_project.sequences import make_sequences
from pm25_project.tuning import chronological_inner_indices

ROOT=Path(__file__).resolve().parent; RES=ROOT/'results'/'16'; EXP=RES/'issue_time_lagged_exogenous_experiment'
(EXP/'models').mkdir(parents=True,exist_ok=True);(EXP/'histories').mkdir(parents=True,exist_ok=True)
lock=json.loads((RES/'future_test.json').read_text(encoding='utf-8')); devset=set(map(int,lock['development_indices']))
# Skip all cleaned-daily rows outside the development partition during parsing.
path=ROOT/'data'/'processed'/'cleaned_daily.csv'
df=pd.read_csv(path,skiprows=lambda line: line>0 and line-1 not in devset,parse_dates=['timestamp'])
df['timestamp']=pd.to_datetime(df.timestamp,utc=True)
if not set(df.clean_index.astype(int))==devset: raise RuntimeError('Development row selection did not match locked indices')
if df.timestamp.max()>pd.to_datetime(lock['development_end'],utc=True): raise RuntimeError('Development rows cross cutoff')
cfg,_=load_config(ROOT/'configs'/'project.yaml'); features=build_origin_features(df,cfg)
base=['pm25_lag1','pm25_roll3_mean']
exo={'temperature_celsius':'temperature_celsius_lag1','pressure_mb':'pressure_mb_lag1','wind_u':'wind_u_lag1','wind_v':'wind_v_lag1','no2':'no2_lag1','co':'co_lag1','o3':'o3_lag1','aod':'aod_lag1'}
# Shift only within each station and mask across any non-daily gap.
groups=features.groupby('station_id',sort=False)
prev_date=groups.timestamp.shift(1)
for src,dst in exo.items():
    features[dst]=groups[src].shift(1)
    features.loc[(features.timestamp-prev_date).dt.days.ne(1),dst]=np.nan
variants={
 'target_lag_roll3':base,
 'target_plus_lagged_pollutants_aod':base+[exo[k] for k in ('no2','co','o3','aod')],
 'target_plus_lagged_meteorology':base+[exo[k] for k in ('temperature_celsius','pressure_mb','wind_u','wind_v')],
 'target_plus_all_lagged_environment':base+list(exo.values()),
}
lookback=7; arrays={}; expected=None
for name,cols in variants.items():
    x,y,keys=make_sequences(features,cols,lookback=lookback,horizon=1)
    ids=set(keys.target_row.astype(int)); expected=ids if expected is None else expected
    if ids!=expected: raise RuntimeError(f'Sample mismatch in {name}')
    arrays[name]=(x,y,keys,cols)
folds=json.loads((RES/'folds_temporal.json').read_text(encoding='utf-8'))['folds']; rows=[];preds=[];selected=[]
for fold in folds:
    k=int(fold['fold']); tuned=json.loads((RES/'models'/f'cnn_lstm_optuna_temporal_fold_{k}_config.json').read_text(encoding='utf-8'))
    hp={key:tuned[key] for key in ('filters','kernel_size','lstm_units','dense_units','dropout','learning_rate','loss')};seed=int(tuned['seed']); batch=int(cfg['model']['cnn_lstm'].get('batch_size',64))
    idxs={};scores={};epochs={}
    for name,(X,y,keys,cols) in arrays.items():
        target=keys.target_row.to_numpy(int);tr=np.flatnonzero(np.isin(target,np.asarray(fold['train_indices'],int)));va=np.flatnonzero(np.isin(target,np.asarray(fold['validation_indices'],int)))
        if not len(tr) or not len(va): raise RuntimeError(f'empty fold {k}/{name}')
        idxs[name]=(tr,va); dates=pd.to_datetime(keys.iloc[tr].target_timestamp,utc=True).dt.tz_localize(None).to_numpy(dtype='datetime64[ns]');itr,iva=chronological_inner_indices(dates)
        prep=SequencePreprocessor().fit(X[tr[itr]]);xt=prep.transform(X[tr[itr]]);xv=prep.transform(X[tr[iva]])
        tf.keras.backend.clear_session();model=build_cnn_lstm(lookback,xt.shape[-1],{**hp,'seed':seed,'pool_size':2})
        callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=3,min_lr=1e-5,verbose=0),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True,verbose=0)]
        hist=model.fit(xt,y[tr[itr]],validation_data=(xv,y[tr[iva]]),epochs=100,batch_size=batch,callbacks=callbacks,verbose=0)
        pv=model.predict(xv,verbose=0).reshape(-1);scores[name]=float(mean_squared_error(y[tr[iva]],pv)**.5);epochs[name]=int(np.argmin(hist.history['val_loss'])+1)
        rows.append({'cv_fold':k,'variant':name,'features':json.dumps(cols),'inner_train_n':len(itr),'inner_validation_n':len(iva),'inner_rmse':scores[name],'inner_mae':float(mean_absolute_error(y[tr[iva]],pv)),'selected_epoch':epochs[name]})
        pd.DataFrame(hist.history).to_csv(EXP/'histories'/f'fold{k}_{name}_inner.csv',index=False)
    chosen=min(variants,key=lambda name:(scores[name],name));selected.append({'cv_fold':k,'selected_variant':chosen,'inner_rmse':scores[chosen],'selected_epoch':epochs[chosen]})
    for name,(X,y,keys,cols) in arrays.items():
        tr,va=idxs[name];prep=SequencePreprocessor().fit(X[tr]);xt=prep.transform(X[tr]);xv=prep.transform(X[va]);tf.keras.backend.clear_session();model=build_cnn_lstm(lookback,xt.shape[-1],{**hp,'seed':seed,'pool_size':2})
        started=time.perf_counter();hist=model.fit(xt,y[tr],epochs=epochs[name],batch_size=batch,verbose=0);elapsed=time.perf_counter()-started
        pv=model.predict(xv,verbose=0).reshape(-1);lag=cols.index('pm25_lag1');persistence=X[va,-1,lag]
        for j,seq in enumerate(va):
            key=keys.iloc[seq];preds.append({'sample_id':int(key.target_row),'station_id':str(key.station_id),'cv_fold':k,'variant':name,'selected_by_inner_validation':name==chosen,'origin_timestamp':pd.Timestamp(key.origin_timestamp).isoformat(),'target_timestamp':pd.Timestamp(key.target_timestamp).isoformat(),'y_true':float(y[seq]),'y_pred':float(pv[j]),'persistence':float(persistence[j])})
        model.save(EXP/'models'/f'fold{k}_{name}.keras');joblib.dump(prep,EXP/'models'/f'fold{k}_{name}_preprocessor.joblib')
        (EXP/'models'/f'fold{k}_{name}_config.json').write_text(json.dumps({**hp,'seed':seed,'pool_size':2,'lookback_days':lookback,'features':cols,'selected_epoch':epochs[name],'feature_variant':name,'max_environmental_source_date_relative_to_origin_days':-1,'outer_validation_used_for_selection':False,'source_delivery_latency_verified':False},indent=2),encoding='utf-8')
        pd.DataFrame(hist.history).to_csv(EXP/'histories'/f'fold{k}_{name}_outer.csv',index=False);tf.keras.backend.clear_session()
pd.DataFrame(rows).to_csv(EXP/'inner_scores.csv',index=False);pd.DataFrame(selected).to_csv(EXP/'selected_variant_by_fold.csv',index=False);pd.DataFrame(preds).to_csv(EXP/'oof_predictions.csv',index=False)
meta={'experiment':'nested four-way CNN-LSTM lagged-environmental feature sensitivity','scope':'development temporal folds only','outer_folds':5,'common_oof_sample_count':len(expected),'forecast_origin':'end of day t','forecast_target':'PM2.5 at t+1','latest_environmental_source_date':'t-1','features_by_variant':variants,'inner_selection':'chronological inner validation RMSE; fold-specific TPE hyperparameters fixed','preprocessing':'median imputer/scaler fit on inner/outer training only with missingness indicators','outer_validation_used_for_selection':False,'future_test_labels_or_metrics_used':False,'source_delivery_latency_verified':False,'interpretation':'Conditional feature-timing sensitivity only. A one-day source-date lag is leakage-resistant but cannot prove operational availability without source delivery timestamps.'}
(EXP/'experiment_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(json.dumps({'n_common':len(expected),'selected':selected,'experiment':str(EXP)},indent=2))
