"""Select and fit a refreshed CNN-LSTM using development-only inner validation.

The reserved future period is not used. The original development checkpoints
are preserved; this writes a separate reloadable model bundle.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import joblib, numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pm25_project.config import load_config
from pm25_project.features import build_origin_features
from pm25_project.models.cnn_lstm import build_cnn_lstm
from pm25_project.preprocessing import SequencePreprocessor
from pm25_project.sequences import make_sequences
from pm25_project.tuning import chronological_inner_indices

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results'/'16'; MODEL_DIR=OUT/'models'; HIST=OUT/'histories'
cfg,_=load_config(ROOT/'configs'/'project.yaml')
lock=json.loads((OUT/'future_test.json').read_text(encoding='utf-8'))
df=pd.read_csv(ROOT/'data'/'processed'/'cleaned_daily.csv',parse_dates=['timestamp']); df['timestamp']=pd.to_datetime(df.timestamp,utc=True)
dev=df.iloc[np.asarray(lock['development_indices'],dtype=int)].copy(); cutoff=pd.to_datetime(lock['development_end'],utc=True)
if dev.timestamp.max()>cutoff: raise ValueError('Development rows cross the locked future-test cutoff')
feature_table=build_origin_features(dev,cfg)
all_cols=list(cfg['features'].get('predictors',[]))+list(cfg['features'].get('calendar',[]))+['pm25_lag1']+[f"pm25_roll{w}_mean" for w in cfg['features'].get('rolling_windows_days',[3,7])]
all_cols=list(dict.fromkeys(c for c in all_cols if c in feature_table.columns)); calendar=set(cfg['features'].get('calendar',[]))
candidates={
 'full_calendar_lag_roll3_roll7':all_cols,
 'no_calendar_lag_roll3_roll7':[c for c in all_cols if c not in calendar],
 'no_calendar_lag_roll3':[c for c in all_cols if c not in calendar and c!='pm25_roll7_mean'],
 'no_calendar_lag_only':[c for c in all_cols if c not in calendar and c not in {'pm25_roll3_mean','pm25_roll7_mean'}],
}
lookback=int(cfg['project'].get('lookback_candidates_days',[7])[0]); arrays={}
for name,cols in candidates.items(): arrays[name]=make_sequences(feature_table,cols,lookback=lookback,horizon=1)
ids=[set(a[2].target_row.astype(int)) for a in arrays.values()]
if any(x!=ids[0] for x in ids[1:]): raise ValueError('Final feature candidates do not use identical sequences')
X0,y0,keys0=next(iter(arrays.values())); dates=pd.to_datetime(keys0.target_timestamp,utc=True).dt.tz_localize(None).to_numpy(dtype='datetime64[ns]')
inner_train,inner_valid=chronological_inner_indices(dates)
old=json.loads((MODEL_DIR/'cnn_lstm_development_optuna_config.json').read_text(encoding='utf-8'))
hp={k:old[k] for k in ('filters','kernel_size','lstm_units','dense_units','dropout','learning_rate','loss')}; seed=42016; batch=int(cfg['model']['cnn_lstm'].get('batch_size',64))
scores=[]; pred_rows=[]; inner_histories={}
for name,(X,y,keys) in arrays.items():
    prep=SequencePreprocessor().fit(X[inner_train]); xt=prep.transform(X[inner_train]); xv=prep.transform(X[inner_valid])
    tf.keras.backend.clear_session(); model=build_cnn_lstm(lookback,xt.shape[-1],{**hp,'seed':seed,'pool_size':2})
    callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=3,min_lr=1e-5,verbose=0),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True,verbose=0)]
    hist=model.fit(xt,y[inner_train],validation_data=(xv,y[inner_valid]),epochs=100,batch_size=batch,callbacks=callbacks,verbose=0)
    pred=model.predict(xv,verbose=0).reshape(-1); rmse=float(mean_squared_error(y[inner_valid],pred)**.5); mae=float(mean_absolute_error(y[inner_valid],pred)); epoch=int(np.argmin(hist.history['val_loss'])+1)
    scores.append({'variant':name,'feature_columns':json.dumps(candidates[name]),'inner_train_sequences':len(inner_train),'inner_validation_sequences':len(inner_valid),'inner_validation_rmse':rmse,'inner_validation_mae':mae,'selected_epoch':epoch,'min_train_loss':float(np.min(hist.history['loss'])),'min_validation_loss':float(np.min(hist.history['val_loss']))})
    inner_histories[name]=pd.DataFrame(hist.history)
    for j,seq in enumerate(inner_valid):
        key=keys.iloc[seq]; pred_rows.append({'variant':name,'sample_id':int(key.target_row),'station_id':str(key.station_id),'target_timestamp':pd.Timestamp(key.target_timestamp).isoformat(),'y_true':float(y[seq]),'y_pred':float(pred[j])})
    tf.keras.backend.clear_session()
scores_df=pd.DataFrame(scores).sort_values(['inner_validation_rmse','variant']).reset_index(drop=True)
chosen=scores_df.iloc[0]; variant=str(chosen.variant); cols=candidates[variant]; X,y,keys=arrays[variant]; epochs=int(chosen.selected_epoch)
prep=SequencePreprocessor().fit(X); xt=prep.transform(X); tf.keras.backend.clear_session(); model=build_cnn_lstm(lookback,xt.shape[-1],{**hp,'seed':seed,'pool_size':2})
start=time.perf_counter(); history=model.fit(xt,y,epochs=epochs,batch_size=batch,verbose=0); fit_time=time.perf_counter()-start
model_path=MODEL_DIR/'cnn_lstm_development_feature_selected.keras'; prep_path=MODEL_DIR/'cnn_lstm_development_feature_selected_preprocessor.joblib'; config_path=MODEL_DIR/'cnn_lstm_development_feature_selected_config.json'
model.save(model_path); joblib.dump(prep,prep_path)
config={**hp,'seed':seed,'pool_size':2,'lookback_days':lookback,'feature_variant':variant,'feature_columns':cols,'best_epoch':epochs,'inner_validation_rmse':float(chosen.inner_validation_rmse),'inner_validation_mae':float(chosen.inner_validation_mae),'selection_rule':'lowest RMSE on latest chronological 20% of development sequences; four predeclared feature families; no future-test labels used','development_cutoff_exclusive':cutoff.isoformat(),'trained_on_development_sequences':len(y),'selection_inner_train_n':len(inner_train),'selection_inner_validation_n':len(inner_valid),'outer_validation_used_for_selection':False}
config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
for name,h in inner_histories.items(): h.to_csv(HIST/f'feature_selected_final_inner_{name}.csv',index=False)
pd.DataFrame(history.history).to_csv(HIST/'cnn_lstm_development_feature_selected_training.csv',index=False)
scores_df.to_csv(OUT/'feature_family_experiment'/'final_development_inner_selection.csv',index=False)
pd.DataFrame(pred_rows).to_csv(OUT/'feature_family_experiment'/'final_development_inner_predictions.csv',index=False)
loaded=tf.keras.models.load_model(model_path); loaded_prep=joblib.load(prep_path); n=min(20,len(X)); smoke=loaded.predict(loaded_prep.transform(X[:n]),verbose=0).reshape(-1)
pd.DataFrame({'sample_id':keys.iloc[:n].target_row,'station_id':keys.iloc[:n].station_id,'target_timestamp':keys.iloc[:n].target_timestamp,'y_true':y[:n],'y_pred_in_sample_smoke_test':smoke}).to_csv(OUT/'predictions'/'development_feature_selected_smoke_test.csv',index=False)
metadata={'status':'saved_and_reloaded','model_path':str(model_path),'preprocessor_path':str(prep_path),'config_path':str(config_path),'architecture':'CNN-LSTM','parameters':int(loaded.count_params()),'model_size_mb':model_path.stat().st_size/1024**2,'fit_seconds':fit_time,'inner_selection_rows':len(inner_valid),'development_training_sequences':len(y),'smoke_test_rows':n,'smoke_test_is_in_sample':True,'python':__import__('platform').python_version(),'tensorflow':tf.__version__,'future_test_labels_or_metrics_used':False}
(OUT/'development_feature_selected_model_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(json.dumps({'selected_variant':variant,'inner_candidates':scores_df.to_dict(orient='records'),'metadata':metadata},indent=2))
