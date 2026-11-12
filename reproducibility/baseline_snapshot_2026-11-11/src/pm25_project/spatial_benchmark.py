"""Development-only CNN-LSTM grouped station validation."""
from __future__ import annotations
import json, time, platform
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from .features import build_origin_features
from .sequences import make_sequences
from .preprocessing import SequencePreprocessor
from .cnn_lstm_benchmark import cnn_lstm_candidates
from .models.cnn_lstm import build_cnn_lstm
from .metrics import regression_metrics
from .explainability import sequence_channel_permutation

def run_cnn_lstm_spatial(df, cfg):
    import tensorflow as tf
    root=Path(cfg['_config_dir']); out=root/cfg['paths']['results_root']/cfg['project']['group_id']
    folds=json.loads((out/'folds_spatial.json').read_text(encoding='utf-8'))['folds']
    table=build_origin_features(df,cfg)
    names=list(cfg['features'].get('predictors',[]))+list(cfg['features'].get('calendar',[]))
    if cfg['features'].get('include_target_lags',True): names.append('pm25_lag1')
    if cfg['features'].get('include_past_rolling',True): names += [f"pm25_roll{w}_mean" for w in cfg['features'].get('rolling_windows_days',[3,7])]
    names=list(dict.fromkeys(names)); candidates=cnn_lstm_candidates(cfg)
    score_rows=[]; pred_rows=[]; config_rows=[]; importance_rows=[]; history_dir=out/'histories'; history_dir.mkdir(parents=True,exist_ok=True)
    for fold in folds:
        cache={}; tunes=[]; search_start=time.perf_counter()
        for candidate in candidates:
            lb=int(candidate['lookback_days']); x,y,keys=make_sequences(table,names,lookback=lb,horizon=1)
            tr=np.flatnonzero(keys.target_row.isin(fold['train_indices']).to_numpy())
            va=np.flatnonzero(keys.target_row.isin(fold['validation_indices']).to_numpy())
            if not len(tr) or not len(va): raise ValueError(f"Spatial fold {fold['fold']} has no complete sequences")
            groups=keys.iloc[tr].station_id.astype(str).to_numpy()
            gss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=int(cfg['project']['seed'])+int(fold['fold']))
            itr,iva=next(gss.split(tr,groups=groups)); inner_train=tr[itr]; inner_val=tr[iva]
            if set(keys.iloc[inner_train].station_id)&set(keys.iloc[inner_val].station_id): raise AssertionError('Inner spatial station leakage')
            prep=SequencePreprocessor().fit(x[inner_train]); xt=prep.transform(x[inner_train]); xv=prep.transform(x[inner_val])
            conf={k:v for k,v in candidate.items() if k!='lookback_days'}; conf['seed']=int(cfg['project']['seed'])+int(fold['fold'])
            tf.keras.backend.clear_session(); model=build_cnn_lstm(lb,xt.shape[-1],conf)
            cb=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=int(conf.get('patience',10)),restore_best_weights=True)]
            hist=model.fit(xt,y[inner_train],validation_data=(xv,y[inner_val]),epochs=int(conf.get('epochs',100)),batch_size=int(conf.get('batch_size',64)),callbacks=cb,verbose=0)
            yp=model.predict(xv,verbose=0).reshape(-1); rmse=regression_metrics(y[inner_val],yp)['rmse']; epoch=int(np.argmin(hist.history['val_loss'])+1)
            pd.DataFrame(hist.history).to_csv(history_dir/f"cnn_lstm_spatial_fold_{fold['fold']}_candidate_{len(tunes)+1}.csv",index=False)
            tunes.append({'config':candidate,'rmse':rmse,'epoch':epoch,'inner_stations':int(pd.Series(keys.iloc[inner_val].station_id).nunique())})
            cache[json.dumps(candidate,sort_keys=True)]=(x,y,keys,tr,va)
        search_time=time.perf_counter()-search_start; tunes.sort(key=lambda r:r['rmse']); best=tunes[0]; c=best['config']
        x,y,keys,tr,va=cache[json.dumps(c,sort_keys=True)]
        prep=SequencePreprocessor().fit(x[tr]); xt=prep.transform(x[tr]); xv=prep.transform(x[va]); conf={k:v for k,v in c.items() if k!='lookback_days'}; conf['seed']=int(cfg['project']['seed'])+100+int(fold['fold'])
        tf.keras.backend.clear_session(); model=build_cnn_lstm(int(c['lookback_days']),xt.shape[-1],conf)
        start=time.perf_counter(); model.fit(xt,y[tr],epochs=best['epoch'],batch_size=int(conf.get('batch_size',64)),verbose=0); fit=time.perf_counter()-start
        start=time.perf_counter(); pred=model.predict(xv,verbose=0).reshape(-1); inference=time.perf_counter()-start
        m=regression_metrics(y[va],pred)
        importance=sequence_channel_permutation(model,xv,y[va],prep.feature_names_out(names),seed=conf['seed'],n_repeats=10)
        for r in importance.itertuples(): importance_rows.append({'model_id':'CNN_LSTM','model_group':'Group_16','cv_repetition':1,'cv_fold':int(fold['fold']),'method':'sequence_channel_permutation_rmse','is_mandatory':True,'feature':r.feature,'importance':r.importance,'rank':r.rank,'cv_scheme':'spatial_group_kfold'})
        for metric,value in m.items():
            if metric!='n_samples': score_rows.append({'model_id':'CNN_LSTM','model_group':'Group_16','model_family':'CNN-LSTM','dataset_split':'validation','cv_scheme':'spatial_group_kfold','cv_purpose':'development_generalization','cv_fold':int(fold['fold']),'metric':metric,'value':value,'n_samples':m['n_samples'],'train_size':len(tr),'train_stations':len(fold['train_groups']),'validation_stations':len(fold['validation_groups']),'lookback_days':c['lookback_days'],'configuration':json.dumps(c,sort_keys=True),'random_seed':conf['seed'],'fit_time_sec':fit,'search_time_sec':search_time,'inference_time_sec':inference,'hardware':platform.platform(),'notes':'Held-out station targets; past observations at each validation station supply the input history.'})
        for i,j in enumerate(va):
            k=keys.iloc[j]; pred_rows.append({'sample_id':k.target_row,'station_id':k.station_id,'origin_timestamp':k.origin_timestamp,'target_timestamp':k.target_timestamp,'y_true':y[j],'y_pred':pred[i],'residual':y[j]-pred[i],'dataset_split':'validation','cv_scheme':'spatial_group_kfold','cv_fold':int(fold['fold']),'model_id':'CNN_LSTM','config_id':f"spatial_fold{fold['fold']}",'seed':conf['seed']})
        config_rows.append({'cv_fold':int(fold['fold']),'selected_configuration':json.dumps(c,sort_keys=True),'inner_station_holdout_rmse':best['rmse'],'best_epoch':best['epoch'],'inner_validation_stations':best['inner_stations'],'candidate_results':json.dumps(tunes,sort_keys=True)})
        md=out/'models'; md.mkdir(exist_ok=True); model.save(md/f"cnn_lstm_spatial_fold_{fold['fold']}.keras")
        import joblib; joblib.dump(prep,md/f"cnn_lstm_spatial_fold_{fold['fold']}_preprocessor.joblib")
        (md/f"cnn_lstm_spatial_fold_{fold['fold']}_config.json").write_text(json.dumps({**c,**conf,'best_epoch':best['epoch'],'feature_columns':names},indent=2),encoding='utf-8')
    pd.DataFrame(score_rows).to_csv(out/'spatial_cv_results.csv',index=False)
    pd.DataFrame(importance_rows).to_csv(out/'feature_importance_spatial.csv',index=False)
    pd.DataFrame(pred_rows).to_csv(out/'predictions'/'cnn_lstm_spatial_oof.csv',index=False)
    pd.DataFrame(config_rows).to_csv(out/'cnn_lstm_spatial_configs.csv',index=False)
    return {'model_id':'CNN_LSTM','n_folds':len(folds),'metrics_rows':len(score_rows),'predictions':len(pred_rows)}
