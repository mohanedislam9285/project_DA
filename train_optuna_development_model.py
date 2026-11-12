"""Fit a final all-development CNN-LSTM from nested TPE fold evidence."""
from __future__ import annotations
import json, time, platform
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from pm25_project.config import load_config
from pm25_project.features import build_origin_features
from pm25_project.models.cnn_lstm import build_cnn_lstm
from pm25_project.preprocessing import SequencePreprocessor
from pm25_project.sequences import make_sequences
from pm25_project.tuning import chronological_inner_indices

ROOT=Path(__file__).resolve().parent
cfg,_=load_config(ROOT/'configs/project.yaml')
out=ROOT/cfg['paths']['results_root']/cfg['project']['group_id']
df=pd.read_csv(ROOT/'data/processed/cleaned_daily.csv',parse_dates=['timestamp'])
df.timestamp=pd.to_datetime(df.timestamp,utc=True)
partition=json.loads((out/'future_test.json').read_text(encoding='utf-8'))
dev=df.iloc[np.asarray(partition['development_indices'],dtype=int)].copy()
fold_scores=pd.read_csv(out/'temporal_cv_optuna_results.csv')
fold_scores['config_key']=fold_scores.configuration.map(lambda x:json.dumps(json.loads(x),sort_keys=True))
choice=(fold_scores.groupby('config_key',as_index=False)
        .agg(folds=('cv_fold','nunique'),mean_inner_rmse=('inner_rmse','mean'),config=('configuration','first'))
        .sort_values(['folds','mean_inner_rmse'],ascending=[False,True]).iloc[0])
chosen=json.loads(choice.config)
features=build_origin_features(dev,cfg)
columns=list(cfg['features'].get('calendar',[]))+['pm25_lag1']+[f"pm25_roll{w}_mean" for w in cfg['features'].get('rolling_windows_days',[3,7])]
columns=list(dict.fromkeys(c for c in columns if c in features))
lookback=int(chosen['lookback_days'])
X,y,keys=make_sequences(features,columns,lookback=lookback,horizon=1)
dates=pd.to_datetime(keys.target_timestamp,utc=True).dt.tz_localize(None).to_numpy(dtype='datetime64[ns]')
inner_train,inner_val=chronological_inner_indices(dates)
model_config={k:chosen[k] for k in ['filters','kernel_size','lstm_units','dropout','learning_rate','dense_units','loss']}
model_config['seed']=42
prep_inner=SequencePreprocessor().fit(X[inner_train])
xi=prep_inner.transform(X[inner_train]); xv=prep_inner.transform(X[inner_val])
tf.keras.backend.clear_session(); probe=build_cnn_lstm(lookback,xi.shape[-1],model_config)
history=probe.fit(xi,y[inner_train],validation_data=(xv,y[inner_val]),epochs=100,batch_size=64,callbacks=[
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=3,min_lr=1e-5,verbose=0),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True,verbose=0),
],verbose=0)
best_epoch=int(np.argmin(history.history['val_loss'])+1)
hist=pd.DataFrame(history.history); hist.insert(0,'epoch',np.arange(1,len(hist)+1))
hist.to_csv(out/'histories'/'cnn_lstm_optuna_final_inner_history.csv',index=False)
del probe; tf.keras.backend.clear_session()
prep=SequencePreprocessor().fit(X); xt=prep.transform(X)
model=build_cnn_lstm(lookback,xt.shape[-1],model_config)
start=time.perf_counter(); final_history=model.fit(xt,y,epochs=best_epoch,batch_size=64,verbose=0); fit_seconds=time.perf_counter()-start
model_path=out/'models'/'cnn_lstm_development_optuna.keras'
model.save(model_path)
prep_path=out/'models'/'cnn_lstm_development_optuna_preprocessor.joblib'; joblib.dump(prep,prep_path)
config={**model_config,'lookback_days':lookback,'best_epoch':best_epoch,'feature_columns':columns,
    'feature_columns_after_preprocessing':prep.feature_names_out(columns),
    'selection_rule':'most frequent outer-fold TPE winner; tie-break by mean inner chronological-validation RMSE; config candidates originate only from outer-training inner validation.',
    'selection_frequency':int(choice.folds),'mean_inner_rmse_across_selected_folds':float(choice.mean_inner_rmse),
    'development_cutoff_exclusive':'2024-12-08T00:00:00Z','trained_on_development_sequences':int(len(y)),
    'training_epochs_selected_on_latest_chronological_inner_split':best_epoch}
(out/'models'/'cnn_lstm_development_optuna_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
pd.DataFrame(final_history.history).to_csv(out/'histories'/'cnn_lstm_development_optuna_training.csv',index=False)
loaded=tf.keras.models.load_model(model_path); loaded_prep=joblib.load(prep_path)
n=min(20,len(X)); start=time.perf_counter(); pred=loaded.predict(loaded_prep.transform(X[:n]),verbose=0).reshape(-1); inference=time.perf_counter()-start
pd.DataFrame({'sample_id':keys.iloc[:n].target_row,'station_id':keys.iloc[:n].station_id,'target_timestamp':keys.iloc[:n].target_timestamp,'y_true':y[:n],'y_pred_in_sample_smoke_test':pred}).to_csv(out/'predictions'/'development_optuna_checkpoint_smoke_test.csv',index=False)
meta={'status':'saved_and_loaded','model_path':str(model_path),'preprocessor_path':str(prep_path),'configuration_path':str(out/'models'/'cnn_lstm_development_optuna_config.json'),'architecture':'CNN-LSTM','model_group':'Group_16','development_sequences':int(len(y)),'parameters':int(loaded.count_params()),'model_size_mb':float(model_path.stat().st_size/1024**2),'fit_seconds':fit_seconds,'inference_seconds_20_in_sample_rows':inference,'smoke_test_rows':int(n),'smoke_test_is_in_sample':True,'seed':42,'python':platform.python_version(),'tensorflow':tf.__version__,'gpu_devices':[d.name for d in tf.config.list_physical_devices('GPU')]}
(out/'development_optuna_model_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(json.dumps(meta,indent=2))
