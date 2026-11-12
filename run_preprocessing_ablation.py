"""Development-only preprocessing/target-scale CNN-LSTM ablations."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from pm25_project.config import load_config
from pm25_project.features import build_origin_features
from pm25_project.metrics import regression_metrics
from pm25_project.models.cnn_lstm import build_cnn_lstm
from pm25_project.preprocessing import SequencePreprocessor
from pm25_project.sequences import make_sequences

ROOT=Path(__file__).resolve().parent
cfg,_=load_config(ROOT/'configs/project.yaml')
out=ROOT/cfg['paths']['results_root']/cfg['project']['group_id']
df=pd.read_csv(ROOT/'data/processed/cleaned_daily.csv',parse_dates=['timestamp'])
df.timestamp=pd.to_datetime(df.timestamp,utc=True)
part=json.loads((out/'future_test.json').read_text(encoding='utf-8'))
dev=df.iloc[np.asarray(part['development_indices'],dtype=int)].copy()
feature=build_origin_features(dev,cfg)
names=list(cfg['features'].get('calendar',[]))+['pm25_lag1']+[f"pm25_roll{w}_mean" for w in cfg['features'].get('rolling_windows_days',[3,7])]
names=list(dict.fromkeys(n for n in names if n in feature))
folds=json.loads((out/'folds_temporal.json').read_text(encoding='utf-8'))['folds']
rows=[]; prediction_rows=[]
for fold in folds:
    f=int(fold['fold'])
    config=json.loads((out/'models'/f'cnn_lstm_temporal_fold_{f}_config.json').read_text(encoding='utf-8'))
    X,y,keys=make_sequences(feature,names,lookback=int(config['lookback_days']),horizon=1)
    train=np.flatnonzero(keys.target_row.isin(fold['train_indices']).to_numpy())
    valid=np.flatnonzero(keys.target_row.isin(fold['validation_indices']).to_numpy())
    reference=pd.read_csv(out/'predictions/cnn_lstm_temporal_oof.csv')
    reference=reference[reference.cv_fold.eq(f)]
    common=set(pd.to_numeric(reference.sample_id).astype(int))
    train=train[keys.iloc[train].target_row.astype(int).isin(common|set(keys.iloc[train].target_row.astype(int))).to_numpy()]
    valid=valid[keys.iloc[valid].target_row.astype(int).isin(common).to_numpy()]
    # The ordinary target/median/indicator result is the original same-fold CNN-LSTM OOF run.
    y_true=y[valid]
    for variant,add_indicator,log_target in [('median_without_indicators',False,False),('median_with_indicators_log1p_target',True,True)]:
        prep=SequencePreprocessor(add_indicator=add_indicator).fit(X[train])
        xt=prep.transform(X[train]); xv=prep.transform(X[valid])
        model_config={k:config[k] for k in ('filters','kernel_size','dropout','lstm_units','dense_units','learning_rate','loss') if k in config}
        model_config['seed']=int(config.get('seed',42))
        train_y=np.log1p(y[train]) if log_target else y[train]
        tf.keras.backend.clear_session(); model=build_cnn_lstm(int(config['lookback_days']),xt.shape[-1],model_config)
        start=time.perf_counter(); model.fit(xt,train_y,epochs=int(config['best_epoch']),batch_size=int(config.get('batch_size',64)),verbose=0); fit_s=time.perf_counter()-start
        prediction=model.predict(xv,verbose=0).reshape(-1)
        if log_target: prediction=np.expm1(prediction)
        metrics=regression_metrics(y_true,prediction)
        for metric,value in metrics.items():
            if metric!='n_samples': rows.append({'variant':variant,'cv_scheme':'temporal_panel_expanding','cv_fold':f,'metric':metric,'value':value,'n_samples':metrics['n_samples'],'train_size':len(train),'fit_time_sec':fit_s,'random_state':model_config['seed'],'notes':'Same temporal folds and fixed fold-selected CNN-LSTM architecture/epoch as reference; no retuning. Log target is inverse-transformed before original-scale scoring.'})
        for j,idx in enumerate(valid):
            key=keys.iloc[idx]; prediction_rows.append({'variant':variant,'sample_id':int(key.target_row),'station_id':key.station_id,'target_timestamp':key.target_timestamp,'y_true':float(y[idx]),'y_pred':float(prediction[j]),'residual':float(y[idx]-prediction[j]),'cv_fold':f})
        tf.keras.backend.clear_session()
    full_metrics=pd.read_csv(out/'temporal_cv_results.csv')
    full_metrics=full_metrics[full_metrics.cv_fold.eq(f)]
    full_preds=reference
    for metric in ('rmse','mae','r2'):
        score=full_metrics[full_metrics.metric.eq(metric)].iloc[0]
        rows.append({'variant':'median_with_indicators_raw_target_reference','cv_scheme':'temporal_panel_expanding','cv_fold':f,'metric':metric,'value':float(score.value),'n_samples':int(len(full_preds)),'train_size':len(train),'fit_time_sec':np.nan,'random_state':int(full_preds.seed.iloc[0]),'notes':'Existing saved CNN-LSTM OOF result; not retrained.'})

table=pd.DataFrame(rows)
ref=table[(table.variant=='median_with_indicators_raw_target_reference')&(table.metric=='rmse')].set_index('cv_fold').value
table['delta_rmse_vs_reference']=np.nan
for idx,row in table[(table.metric=='rmse')&table.variant.ne('median_with_indicators_raw_target_reference')].iterrows():
    table.loc[idx,'delta_rmse_vs_reference']=row.value-ref.loc[row.cv_fold]
table.to_csv(out/'preprocessing_ablation.csv',index=False,encoding='utf-8')
pd.DataFrame(prediction_rows).to_csv(out/'predictions'/'cnn_lstm_preprocessing_ablation_oof.csv',index=False,encoding='utf-8')
summary=table[table.metric.eq('rmse')].groupby('variant').value.agg(['mean','std','count']).reset_index()
summary.to_csv(out/'preprocessing_ablation_summary.csv',index=False,encoding='utf-8')
print(summary.to_string(index=False))
