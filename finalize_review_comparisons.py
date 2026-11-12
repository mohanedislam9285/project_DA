"""Merge new development diagnostics and populate paired comparison outputs."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pm25_project.comparisons import comparison_table
from pm25_project.config import load_config
from pm25_project.development_diagnostics import run_validation_scheme_ablation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parent
cfg,_=load_config(ROOT/'configs/project.yaml')
out=ROOT/cfg['paths']['results_root']/cfg['project']['group_id']
diagnostic=pd.read_csv(out/'diagnostic_cv_results.csv')
optuna=pd.read_csv(out/'temporal_cv_optuna_results.csv')
optuna_pred=pd.read_csv(out/'predictions'/'cnn_lstm_optuna_temporal_oof.csv')
summary_rows=[]
for metric,fold_col in [('rmse','metric_rmse'),('mae','metric_mae'),('r2','metric_r2')]:
    values=optuna[fold_col].to_numpy(dtype=float)
    rng=np.random.default_rng(42+len(summary_rows))
    boot=np.mean(rng.choice(values,(50000,len(values)),replace=True),axis=1)
    pooled={'rmse':float(np.sqrt(mean_squared_error(optuna_pred.y_true,optuna_pred.y_pred))),
            'mae':float(mean_absolute_error(optuna_pred.y_true,optuna_pred.y_pred)),
            'r2':float(r2_score(optuna_pred.y_true,optuna_pred.y_pred))}[metric]
    summary_rows.append({'cv_scheme':'temporal_panel_expanding','model_id':'CNN_LSTM_OPTUNA','metric':metric,
        'fold_mean':float(values.mean()),'fold_sd':float(values.std(ddof=1)),
        'bootstrap_95_ci_lower':float(np.quantile(boot,.025)),'bootstrap_95_ci_upper':float(np.quantile(boot,.975)),
        'pooled_oof_metric':pooled,'n_folds':int(len(values)),'total_oof_predictions':int(optuna_pred.shape[0])})
pd.DataFrame(summary_rows).to_csv(out/'development_cv_optuna_summary.csv',index=False,encoding='utf-8')
optuna_rows=[]
for row in optuna.itertuples():
    for metric,col in [('rmse','metric_rmse'),('mae','metric_mae'),('r2','metric_r2')]:
        optuna_rows.append({'model_id':'CNN_LSTM_OPTUNA','model_group':'Group_16','model_family':'CNN-LSTM','dataset_split':'validation','cv_scheme':'temporal_panel_expanding','cv_purpose':'development_selection','cv_fold':int(row.cv_fold),'niveau':'fold','metric':metric,'is_primary':metric=='rmse','value':float(getattr(row,col)),'n_samples':int(row.n_samples),'train_size':int(row.training_sequences),'random_state':int(row.seed),'fit_time_sec':float(row.fit_time_sec),'search_time_sec':float(row.search_time_sec),'inference_time_sec':float(row.inference_time_sec),'model_size_mb':float(row.model_size_mb),'n_parameters':int(row.n_parameters),'n_features':7,'fold_file':'folds_temporal.json','notes':'Nested 50-trial seeded Optuna-TPE selection using chronological inner validation; outer validation used once for scoring.'})
new=pd.concat([diagnostic,pd.DataFrame(optuna_rows)],ignore_index=True,sort=False)
res_path=out/'results.csv'
old=pd.read_csv(res_path)
replace_ids={'CNN_LSTM_DIAGNOSTIC','CNN_LSTM_OPTUNA','persistence','seasonal_naive'}
old=old[~old.model_id.astype(str).isin(replace_ids)]
for c in old.columns:
    if c not in new: new[c]=np.nan
for c in new.columns:
    if c not in old: old[c]=np.nan
new=new[old.columns]
combined=pd.concat([old,new],ignore_index=True)
combined.to_csv(res_path,index=False,encoding='utf-8')
comparisons=comparison_table(combined,seed=42,n_perm=10000)
comparisons.to_csv(out/'comparisons.csv',index=False,encoding='utf-8')
gap=run_validation_scheme_ablation(cfg)
(out/'paired_comparison_metadata.json').write_text(json.dumps({
    'tests':'paired Wilcoxon signed-rank on fold-level RMSE; 10,000 paired sign permutations supplemental; 10,000 fold-pair bootstrap replicates for mean difference; Holm-Bonferroni across the generated pairwise tests.',
    'effect_size':'Wilcoxon matched-pairs rank-biserial r',
    'n_folds_per_comparison':5,
    'note':'Only five dependent folds provide low inferential power; practical deltas and intervals matter more than a thresholded p-value.',
    'random_cv_role':'diagnostic only; not used for model selection',
    'validation_scheme_gaps':gap,
},indent=2),encoding='utf-8')
print('result rows',len(combined),'paired comparisons',len(comparisons))
print(comparisons[['comparison_id','mean_a','mean_b','delta','delta_ci_lower','delta_ci_upper','effect_size','p_value','p_value_corrected','n_paired_folds']].to_string(index=False))
print('validation scheme',gap)
