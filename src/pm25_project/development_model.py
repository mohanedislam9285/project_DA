"""Fit a final CNN-LSTM using development-only data and save a loadable bundle."""
from __future__ import annotations
import json,time,platform
from pathlib import Path
import numpy as np,pandas as pd
from .features import build_origin_features
from .sequences import make_sequences
from .preprocessing import SequencePreprocessor
from .models.cnn_lstm import build_cnn_lstm
from .tuning import chronological_inner_indices

def train_development_model(df,cfg):
    import tensorflow as tf, joblib
    root=Path(cfg['_config_dir']); out=root/cfg['paths']['results_root']/cfg['project']['group_id']; cutoff=pd.Timestamp('2024-12-08',tz='UTC')
    dev=df[pd.to_datetime(df.timestamp,utc=True)<cutoff].copy()
    feats=build_origin_features(dev,cfg); names=list(cfg['features'].get('predictors',[]))+list(cfg['features'].get('calendar',[]))+['pm25_lag1']+[f"pm25_roll{w}_mean" for w in cfg['features'].get('rolling_windows_days',[3,7])]
    names=list(dict.fromkeys(names)); lookbacks=cfg['project'].get('lookback_candidates_days',[7]); lookback=int(lookbacks[0])
    X,y,keys=make_sequences(feats,names,lookback=lookback,horizon=1)
    # Select a candidate using only pre-existing development-fold inner scores.
    top=pd.read_csv(out/'cnn_lstm_top3_configs.csv'); top=top[top.config_rank.eq(1)].copy()
    parsed=top.configuration.map(json.loads); top['config_key']=parsed.map(lambda x:json.dumps(x,sort_keys=True))
    rank=top.groupby('config_key').agg(selections=('cv_fold','nunique'),mean_inner_rmse=('inner_validation_rmse','mean')).sort_values(['selections','mean_inner_rmse'],ascending=[False,True])
    chosen=json.loads(rank.index[0]); rows=top[top.config_key.eq(rank.index[0])]
    epochs=max(1,int(np.median(rows.best_epoch)))
    target_dates=pd.to_datetime(keys.target_timestamp,utc=True)
    inner_train,inner_val=chronological_inner_indices(target_dates.to_numpy())
    # Hold out the latest chronological 20% only to estimate a stopping epoch.
    seed=int(cfg['project']['seed']); tuning={k:chosen[k] for k in ['filters','kernel_size','dropout','lstm_units','dense_units','learning_rate','loss'] if k in chosen}; tuning['seed']=seed
    prep_inner=SequencePreprocessor().fit(X[inner_train]); xi=prep_inner.transform(X[inner_train]); xv=prep_inner.transform(X[inner_val])
    tf.keras.backend.clear_session(); probe=build_cnn_lstm(lookback,xi.shape[-1],tuning)
    hist=probe.fit(xi,y[inner_train],validation_data=(xv,y[inner_val]),epochs=int(cfg['model']['cnn_lstm'].get('epochs',100)),batch_size=int(cfg['model']['cnn_lstm'].get('batch_size',64)),callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=int(cfg['model']['cnn_lstm'].get('patience',10)),restore_best_weights=True)],verbose=0)
    best_epoch=int(np.argmin(hist.history['val_loss'])+1); epochs=best_epoch
    del probe; tf.keras.backend.clear_session()
    prep=SequencePreprocessor().fit(X); Xt=prep.transform(X); conf={**tuning,'seed':seed}
    model=build_cnn_lstm(lookback,Xt.shape[-1],conf); started=time.perf_counter(); final_history=model.fit(Xt,y,epochs=epochs,batch_size=int(cfg['model']['cnn_lstm'].get('batch_size',64)),verbose=0); fit_seconds=time.perf_counter()-started
    modeldir=out/'models'; modeldir.mkdir(exist_ok=True); modelpath=modeldir/'cnn_lstm_development.keras'; model.save(modelpath); joblib.dump(prep,modeldir/'cnn_lstm_development_preprocessor.joblib')
    config={**conf,'lookback_days':lookback,'best_epoch':epochs,'feature_columns':names,'feature_columns_after_preprocessing':prep.feature_names_out(names),'selection_rule':'most frequently selected rank-1 configuration across five temporal development folds; tie-break by mean inner-validation RMSE','development_cutoff_exclusive':str(cutoff),'trained_on_development_samples':int(len(y))}
    (modeldir/'cnn_lstm_development_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (out/'histories').mkdir(parents=True,exist_ok=True)
    pd.DataFrame(final_history.history).to_csv(out/'histories'/'cnn_lstm_development_training.csv',index=False)
    tf.keras.backend.clear_session(); loaded=tf.keras.models.load_model(modelpath); loadedprep=joblib.load(modeldir/'cnn_lstm_development_preprocessor.joblib')
    sm_n=min(20,len(X)); start=time.perf_counter(); prediction=loaded.predict(loadedprep.transform(X[:sm_n]),verbose=0).reshape(-1); infer=time.perf_counter()-start
    pd.DataFrame({'sample_id':keys.iloc[:sm_n].target_row,'station_id':keys.iloc[:sm_n].station_id,'target_timestamp':keys.iloc[:sm_n].target_timestamp,'y_true':y[:sm_n],'y_pred_in_sample_smoke_test':prediction}).to_csv(out/'predictions'/'development_checkpoint_smoke_test.csv',index=False)
    meta={'status':'saved_and_loaded','model_path':str(modelpath),'preprocessor_path':str(modeldir/'cnn_lstm_development_preprocessor.joblib'),'configuration_path':str(modeldir/'cnn_lstm_development_config.json'),'model_group':'Group_16','architecture':'CNN-LSTM','train_rows':int(len(y)),'development_rows':int(len(dev)),'lookback_days':lookback,'n_features_before_imputation':len(names),'parameters':int(loaded.count_params()),'model_size_mb':round(modelpath.stat().st_size/1024**2,4),'fit_seconds':fit_seconds,'inference_seconds_for_smoke_test':infer,'smoke_test_predictions':int(len(prediction)),'smoke_test_is_in_sample':True,'python':platform.python_version(),'tensorflow':tf.__version__,'gpu_devices':[d.name for d in tf.config.list_physical_devices('GPU')]}
    (out/'development_model_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return meta
