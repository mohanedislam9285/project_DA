"""CNN-LSTM-only feature and chronological training-size sensitivity studies."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from .features import build_origin_features
from .sequences import make_sequences
from .preprocessing import SequencePreprocessor
from .models.cnn_lstm import build_cnn_lstm
from .metrics import regression_metrics,paired_bootstrap_ci

def _prepare(df,cfg):
    root=Path(cfg['_config_dir']); out=root/cfg['paths']['results_root']/cfg['project']['group_id']
    folds=json.loads((out/'folds_temporal.json').read_text())['folds']; feature=build_origin_features(df,cfg)
    cal=list(cfg['features'].get('calendar',[])); hist=['pm25_lag1']+[f"pm25_roll{w}_mean" for w in cfg['features'].get('rolling_windows_days',[3,7])]
    configs=[]
    for f in folds:
        c=json.loads((out/'models'/f"cnn_lstm_temporal_fold_{f['fold']}_config.json").read_text())
        configs.append((f,c))
    return out,feature,cal,hist,configs

def run_feature_ablation_cnn(df,cfg):
    import tensorflow as tf
    out,feature,cal,hist,folds=_prepare(df,cfg); variants={'full':cal+hist,'without_calendar':hist,'without_lag_and_rolling':cal,'without_rolling':cal+['pm25_lag1']}
    rows=[]; preds=[]
    for f,c in folds:
        ref=pd.read_csv(out/'predictions/cnn_lstm_temporal_oof.csv'); ref=ref[ref.cv_fold.eq(f['fold'])]; expected=set(pd.to_numeric(ref.sample_id).astype(int))
        for variant,names in variants.items():
            x,y,key=make_sequences(feature,names,lookback=int(c['lookback_days']),horizon=1)
            tr=np.flatnonzero(key.target_row.isin(f['train_indices']).to_numpy()); va=np.flatnonzero(key.target_row.isin(f['validation_indices']).to_numpy())
            tr=tr[key.iloc[tr].target_row.astype(int).isin(expected|set(key.iloc[tr].target_row.astype(int))).to_numpy()]
            keep=key.iloc[va].target_row.astype(int).isin(expected).to_numpy(); va=va[keep]
            prep=SequencePreprocessor().fit(x[tr]); xt=prep.transform(x[tr]); xv=prep.transform(x[va])
            conf={k:c[k] for k in ('filters','kernel_size','dropout','lstm_units','dense_units','learning_rate','loss') if k in c}; conf['seed']=int(c.get('seed',42))
            tf.keras.backend.clear_session(); m=build_cnn_lstm(int(c['lookback_days']),xt.shape[-1],conf)
            start=time.perf_counter(); m.fit(xt,y[tr],epochs=int(c['best_epoch']),batch_size=int(c.get('batch_size',64)),verbose=0); fit=time.perf_counter()-start
            yp=m.predict(xv,verbose=0).reshape(-1); met=regression_metrics(y[va],yp)
            for metric,value in met.items():
                if metric!='n_samples': rows.append({'model_id':'CNN_LSTM','model_group':'Group_16','ablation_axis':'feature_family','variant':variant,'cv_scheme':'temporal_panel_expanding','cv_fold':int(f['fold']),'metric':metric,'value':value,'delta_vs_full':np.nan,'random_state':conf['seed'],'n_samples':met['n_samples'],'train_size':len(tr),'fit_time_sec':fit,'features':json.dumps(names),'notes':'Fixed fold-selected architecture and epoch; same outer temporal folds.'})
            for i,j in enumerate(va):
                k=key.iloc[j]; preds.append({'sample_id':k.target_row,'station_id':k.station_id,'target_timestamp':k.target_timestamp,'y_true':y[j],'y_pred':yp[i],'residual':y[j]-yp[i],'variant':variant,'cv_fold':int(f['fold'])})
    d=pd.DataFrame(rows); full=d[(d.variant=='full')&(d.metric=='rmse')].set_index('cv_fold').value
    d['delta_ci_lower']=np.nan; d['delta_ci_upper']=np.nan
    rmse=d[d.metric.eq('rmse')]
    d.loc[rmse.index,'delta_vs_full']=[r.value-full.loc[r.cv_fold] if r.variant!='full' else 0 for r in rmse.itertuples()]
    rmse=d[d.metric.eq('rmse')]
    summary_rows=[]
    for variant,group in rmse.groupby('variant',sort=True):
        values=group.value.to_numpy(dtype=float)
        if variant=='full':
            delta_ci=(np.nan,np.nan); mean_delta=np.nan
        else:
            deltas=group.set_index('cv_fold').delta_vs_full.dropna().to_numpy(dtype=float)
            delta_ci=paired_bootstrap_ci(deltas,seed=cfg['project']['seed'])
            mean_delta=float(deltas.mean()) if len(deltas) else np.nan
            d.loc[group.index,'delta_ci_lower']=delta_ci[0]
            d.loc[group.index,'delta_ci_upper']=delta_ci[1]
        summary_rows.append({
            'variant':variant,'mean':float(values.mean()),
            'std':float(values.std(ddof=1)) if len(values)>1 else np.nan,
            'count':int(len(values)),'mean_delta_vs_full':mean_delta,
            'delta_ci_lower':delta_ci[0],'delta_ci_upper':delta_ci[1],
        })
    d.to_csv(out/'ablation.csv',index=False); pd.DataFrame(preds).to_csv(out/'predictions/cnn_lstm_feature_ablation_oof.csv',index=False)
    pd.DataFrame(summary_rows).to_csv(out/'ablation_summary.csv',index=False)
    return {'variants':list(variants),'folds':len(folds),'metric_rows':len(d)}

def run_train_size_cnn(df,cfg,fractions=(.10,.25,.50,.75,1.0)):
    import tensorflow as tf
    out,feature,cal,hist,folds=_prepare(df,cfg); names=cal+hist; rows=[]; preds=[]
    for f,c in folds:
        x,y,key=make_sequences(feature,names,lookback=int(c['lookback_days']),horizon=1)
        tr=np.flatnonzero(key.target_row.isin(f['train_indices']).to_numpy()); va=np.flatnonzero(key.target_row.isin(f['validation_indices']).to_numpy())
        order=np.argsort(pd.to_datetime(key.iloc[tr].target_timestamp,utc=True).to_numpy())
        for fraction in fractions:
            sub=tr[order[:max(1,int(np.ceil(len(order)*fraction)))] ]
            prep=SequencePreprocessor().fit(x[sub]); xt=prep.transform(x[sub]); xv=prep.transform(x[va])
            conf={k:c[k] for k in ('filters','kernel_size','dropout','lstm_units','dense_units','learning_rate','loss') if k in c}; conf['seed']=500+int(f['fold'])
            tf.keras.backend.clear_session(); m=build_cnn_lstm(int(c['lookback_days']),xt.shape[-1],conf)
            start=time.perf_counter(); m.fit(xt,y[sub],epochs=int(c['best_epoch']),batch_size=int(c.get('batch_size',64)),verbose=0); fit=time.perf_counter()-start
            yp=m.predict(xv,verbose=0).reshape(-1); met=regression_metrics(y[va],yp)
            for metric,value in met.items():
                if metric!='n_samples': rows.append({'model_id':'CNN_LSTM','cv_scheme':'temporal_panel_expanding','cv_fold':int(f['fold']),'fraction':fraction,'metric':metric,'value':value,'n_samples':met['n_samples'],'train_sequences':len(sub),'fixed_epochs':int(c['best_epoch']),'fit_time_sec':fit,'notes':'Earliest chronological proportion of each outer training fold; same validation fold.'})
            for i,j in enumerate(va):
                k=key.iloc[j]; preds.append({'sample_id':k.target_row,'station_id':k.station_id,'target_timestamp':k.target_timestamp,'y_true':y[j],'y_pred':yp[i],'residual':y[j]-yp[i],'fraction':fraction,'cv_fold':int(f['fold'])})
    d=pd.DataFrame(rows); d.to_csv(out/'training_size_sensitivity.csv',index=False); pd.DataFrame(preds).to_csv(out/'predictions/cnn_lstm_training_size_oof.csv',index=False)
    return {'fractions':list(fractions),'folds':len(folds),'metric_rows':len(d)}
