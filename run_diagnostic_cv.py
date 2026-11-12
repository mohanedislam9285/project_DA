"""Score a fixed CNN-LSTM and past-only baselines on development CV schemes."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from pm25_project.config import load_config
from pm25_project.baselines import predict_origin_baselines
from pm25_project.features import build_origin_features
from pm25_project.metrics import regression_metrics
from pm25_project.models.cnn_lstm import build_cnn_lstm
from pm25_project.preprocessing import SequencePreprocessor
from pm25_project.sequences import make_sequences
from pm25_project.tuning import chronological_inner_indices

ROOT=Path(__file__).resolve().parent
cfg,_=load_config(ROOT/'configs/project.yaml')
out=ROOT/cfg['paths']['results_root']/cfg['project']['group_id']
df=pd.read_csv(ROOT/'data/processed/cleaned_daily.csv',parse_dates=['timestamp'])
df['timestamp']=pd.to_datetime(df.timestamp,utc=True)
part=json.loads((out/'future_test.json').read_text(encoding='utf-8'))
dev=df.iloc[np.asarray(part['development_indices'],dtype=int)].copy()
standard_path=out/'folds_standard.json'
if not standard_path.is_file():
    from pm25_project.splits import random_diagnostic_folds
    dev_positions=np.asarray(part['development_indices'],dtype=int)
    standard=random_diagnostic_folds(dev,n_splits=5,seed=42)
    for fold in standard['folds']:
        fold['train_indices']=dev_positions[np.asarray(fold['train_indices'],dtype=int)].tolist()
        fold['validation_indices']=dev_positions[np.asarray(fold['validation_indices'],dtype=int)].tolist()
    standard['future_holdout_file']='future_test.json'
    standard_path.write_text(json.dumps(standard,indent=2),encoding='utf-8')
features=build_origin_features(dev,cfg)
columns=list(cfg['features'].get('calendar',[]))+['pm25_lag1']+[f"pm25_roll{w}_mean" for w in cfg['features'].get('rolling_windows_days',[3,7])]
columns=list(dict.fromkeys(c for c in columns if c in features))
lookback=int(cfg['project'].get('lookback_candidates_days',[7])[0])
X,y,keys=make_sequences(features,columns,lookback=lookback,horizon=1)
target_rows=keys.target_row.to_numpy(dtype=int)
dates=pd.to_datetime(keys.target_timestamp,utc=True).dt.tz_localize(None).to_numpy(dtype='datetime64[ns]')
base={k:v for k,v in cfg['model']['cnn_lstm'].items() if k in ['filters','kernel_size','dropout','lstm_units','dense_units','learning_rate','loss']}
metric_rows=[]; prediction_rows=[]
schemes=[('temporal_panel_expanding','folds_temporal.json'),('spatial_group_kfold','folds_spatial.json'),('random_diagnostic','folds_standard.json')]
for scheme,filename in schemes:
    definition=json.loads((out/filename).read_text(encoding='utf-8'))
    for fold in definition.get('folds',[]):
        fold_id=int(fold['fold'])
        train=np.flatnonzero(np.isin(target_rows,np.asarray(fold['train_indices'],dtype=int)))
        valid=np.flatnonzero(np.isin(target_rows,np.asarray(fold['validation_indices'],dtype=int)))
        if not len(train) or not len(valid): raise RuntimeError(f'{scheme} fold {fold_id} has no complete sequences')
        inner_tr,inner_va=chronological_inner_indices(dates[train])
        prep_inner=SequencePreprocessor().fit(X[train[inner_tr]])
        xi=prep_inner.transform(X[train[inner_tr]]); xv=prep_inner.transform(X[train[inner_va]])
        conf={**base,'seed':42+fold_id}
        callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=3,min_lr=1e-5,verbose=0),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True,verbose=0)]
        tf.keras.backend.clear_session(); model=build_cnn_lstm(lookback,xi.shape[-1],conf)
        hist=model.fit(xi,y[train[inner_tr]],validation_data=(xv,y[train[inner_va]]),epochs=100,batch_size=64,callbacks=callbacks,verbose=0)
        best_epoch=max(1,int(np.argmin(hist.history['val_loss'])+1))
        prep=SequencePreprocessor().fit(X[train]); xt=prep.transform(X[train]); xv=prep.transform(X[valid])
        tf.keras.backend.clear_session(); model=build_cnn_lstm(lookback,xt.shape[-1],conf)
        t=time.perf_counter(); model.fit(xt,y[train],epochs=best_epoch,batch_size=64,verbose=0); fit_s=time.perf_counter()-t
        t=time.perf_counter(); cnn=model.predict(xv,verbose=0).reshape(-1); inf_s=time.perf_counter()-t
        target=y[valid]
        for model_id,pred,group,family,note in [('CNN_LSTM_DIAGNOSTIC',cnn,'Group_16','CNN-LSTM fixed diagnostic config','Fixed predeclared config; no tuning on this validation scheme.')]:
            metrics=regression_metrics(target,pred)
            for name,value in metrics.items():
                if name!='n_samples': metric_rows.append({'model_id':model_id,'model_group':group,'model_family':family,'dataset_split':'validation','cv_scheme':scheme,'cv_purpose':'diagnostic_only','cv_fold':fold_id,'niveau':'fold','metric':name,'is_primary':name=='rmse','value':value,'n_samples':len(valid),'train_size':len(train),'random_state':42+fold_id,'fit_time_sec':fit_s,'inference_time_sec':inf_s,'n_parameters':int(model.count_params()),'n_features':len(columns),'fold_file':filename,'notes':note+' Random CV is an optimism diagnostic only.'})
            for j,idx in enumerate(valid):
                key=keys.iloc[idx]; prediction_rows.append({'sample_id':int(key.target_row),'station_id':key.station_id,'origin_timestamp':key.origin_timestamp,'target_timestamp':key.target_timestamp,'y_true':float(y[idx]),'y_pred':float(pred[j]),'residual':float(y[idx]-pred[j]),'dataset_split':'validation','cv_scheme':scheme,'cv_fold':fold_id,'model_id':model_id,'config_id':'predeclared_config','seed':42+fold_id})
        baselines=predict_origin_baselines(dev,keys.iloc[valid],int(cfg['project'].get('seasons_naive_period_days',7)))
        for model_id,pred in baselines.items():
            ok=np.isfinite(pred); metrics=regression_metrics(target[ok],pred[ok])
            for name,value in metrics.items():
                if name!='n_samples': metric_rows.append({'model_id':model_id,'model_group':'baseline','model_family':'past-only naive','dataset_split':'validation','cv_scheme':scheme,'cv_purpose':'diagnostic_only' if scheme=='random_diagnostic' else 'development','cv_fold':fold_id,'niveau':'fold','metric':name,'is_primary':name=='rmse','value':value,'n_samples':int(ok.sum()),'train_size':len(train),'random_state':42,'fold_file':filename,'notes':'Simple past-only forecast on the same validation sequence cohort.'})
            for j,idx in enumerate(valid):
                if not ok[j]: continue
                key=keys.iloc[idx]; prediction_rows.append({'sample_id':int(key.target_row),'station_id':key.station_id,'origin_timestamp':key.origin_timestamp,'target_timestamp':key.target_timestamp,'y_true':float(y[idx]),'y_pred':float(pred[j]),'residual':float(y[idx]-pred[j]),'dataset_split':'validation','cv_scheme':scheme,'cv_fold':fold_id,'model_id':model_id,'config_id':'past_only_naive','seed':42})
        tf.keras.backend.clear_session()

results=pd.DataFrame(metric_rows); preds=pd.DataFrame(prediction_rows)
results.to_csv(out/'diagnostic_cv_results.csv',index=False,encoding='utf-8')
preds.to_csv(out/'predictions'/'cnn_lstm_and_baselines_diagnostic_cv_oof.csv',index=False,encoding='utf-8')
rmse=results[(results.model_id=='CNN_LSTM_DIAGNOSTIC')&(results.metric=='rmse')]
means=rmse.groupby('cv_scheme').agg(mean_rmse=('value','mean'),sd_rmse=('value','std'),folds=('cv_fold','nunique')).to_dict('index')
gaps=[]
for left,right,name in [('random_diagnostic','temporal_panel_expanding','random_minus_temporal'),('random_diagnostic','spatial_group_kfold','random_minus_spatial')]:
    if left in means and right in means:
        gaps.append({'comparison':name,'model_id':'CNN_LSTM_DIAGNOSTIC','scheme_a':left,'scheme_b':right,'mean_rmse_a':means[left]['mean_rmse'],'mean_rmse_b':means[right]['mean_rmse'],'delta_rmse':means[left]['mean_rmse']-means[right]['mean_rmse'],'folds_a':means[left]['folds'],'folds_b':means[right]['folds'],'interpretation':'Unpaired diagnostic for the same predeclared architecture/configuration; random splits are not selection evidence.'})
pd.DataFrame(gaps).to_csv(out/'optimism_gaps.csv',index=False,encoding='utf-8')
print(json.dumps({'fold_rows':len(results),'prediction_rows':len(preds),'scheme_means':means,'optimism_gaps':gaps},indent=2))
