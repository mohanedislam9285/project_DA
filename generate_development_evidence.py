from pathlib import Path
import json, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('results/16'); fig=out/'figures'; fig.mkdir(exist_ok=True)
pred=pd.read_csv(out/'predictions/cnn_lstm_temporal_oof.csv',parse_dates=['target_timestamp'])
pred['residual']=pred.y_true-pred.y_pred
pred['abs_error']=pred.residual.abs()
pred['pollution_band']=pd.cut(pred.y_true,[-np.inf,10,25,50,np.inf],labels=['<10','10-25','25-50','>50'])
pred['month']=pred.target_timestamp.dt.month
pred['season']=pred.month.map({12:'winter',1:'winter',2:'winter',3:'spring',4:'spring',5:'spring',6:'summer',7:'summer',8:'summer',9:'autumn',10:'autumn',11:'autumn'})
def metrics(g):
 return pd.Series({'n':len(g),'rmse':np.sqrt(np.mean(g.residual**2)),'mae':np.mean(g.abs_error),'mean_error_actual_minus_predicted':g.residual.mean(),'underprediction_rate':(g.residual>0).mean()})
pred.groupby('station_id',observed=True).apply(metrics,include_groups=False).reset_index().to_csv(out/'error_by_station.csv',index=False)
pred.groupby('season',observed=True).apply(metrics,include_groups=False).reset_index().to_csv(out/'error_by_season.csv',index=False)
pred.groupby('pollution_band',observed=True).apply(metrics,include_groups=False).reset_index().to_csv(out/'error_by_concentration_band.csv',index=False)
pred.groupby('month').apply(metrics,include_groups=False).reset_index().to_csv(out/'error_by_month.csv',index=False)
pred.to_csv(out/'predictions/cnn_lstm_temporal_oof.csv',index=False)
# Diagnostic figures
fig1,ax=plt.subplots(figsize=(7,5)); ax.scatter(pred.y_true,pred.y_pred,s=9,alpha=.25,color='#087f80'); lo=min(pred.y_true.min(),pred.y_pred.min()); hi=max(pred.y_true.max(),pred.y_pred.max()); ax.plot([lo,hi],[lo,hi],'--',color='#10243a'); ax.set(xlabel='Observed PM2.5 (µg/m³)',ylabel='Predicted PM2.5 (µg/m³)',title='Temporal out-of-fold predictions'); fig1.tight_layout(); fig1.savefig(fig/'temporal_oof_actual_vs_predicted.png',dpi=180); plt.close(fig1)
fig1,ax=plt.subplots(figsize=(7,4)); ax.hist(pred.residual,bins=40,color='#3977a8',alpha=.85); ax.axvline(0,color='#10243a',ls='--'); ax.set(xlabel='Residual: observed - predicted (µg/m³)',ylabel='Count',title='Temporal out-of-fold residual distribution'); fig1.tight_layout(); fig1.savefig(fig/'temporal_oof_residuals.png',dpi=180); plt.close(fig1)
fig1,ax=plt.subplots(figsize=(8,4)); (pred.groupby('target_timestamp').residual.mean()).plot(ax=ax,color='#087f80'); ax.axhline(0,color='#10243a',ls='--'); ax.set(xlabel='Target date',ylabel='Mean residual (µg/m³)',title='Mean temporal out-of-fold residual by date'); fig1.tight_layout(); fig1.savefig(fig/'temporal_oof_residuals_over_time.png',dpi=180); plt.close(fig1)
importance=pd.read_csv(out/'feature_importance.csv'); imp=importance.groupby('feature').agg(mean_importance=('importance','mean'),sd_importance=('importance','std'),mean_rank=('rank','mean'),folds=('cv_fold','nunique')).sort_values('mean_importance',ascending=False).reset_index(); imp.to_csv(out/'feature_importance_summary.csv',index=False)
top=imp.head(12).sort_values('mean_importance'); fig1,ax=plt.subplots(figsize=(8,5)); ax.barh(top.feature,top.mean_importance,xerr=top.sd_importance.fillna(0),color='#087f80'); ax.set(xlabel='Increase in validation RMSE after channel permutation (µg/m³)',title='CNN-LSTM temporal permutation importance'); fig1.tight_layout(); fig1.savefig(fig/'cnn_lstm_feature_importance.png',dpi=180); plt.close(fig1)
folds=pd.read_csv(out/'temporal_cv_results.csv'); folds=folds[folds.metric.eq('rmse')]; fig1,ax=plt.subplots(figsize=(7,4)); ax.bar(folds.cv_fold,folds.value,color='#3977a8'); ax.set(xlabel='Temporal fold',ylabel='RMSE (µg/m³)',title='CNN-LSTM development temporal fold RMSE'); fig1.tight_layout(); fig1.savefig(fig/'temporal_cv_rmse.png',dpi=180); plt.close(fig1)
sp=pd.read_csv(out/'spatial_cv_results.csv'); sp=sp[sp.metric.eq('rmse')]; fig1,ax=plt.subplots(figsize=(7,4)); ax.bar(sp.cv_fold,sp.value,color='#087f80'); ax.set(xlabel='Station-group fold',ylabel='RMSE (µg/m³)',title='CNN-LSTM unseen-station fold RMSE'); fig1.tight_layout(); fig1.savefig(fig/'spatial_cv_rmse.png',dpi=180); plt.close(fig1)
spimp=pd.read_csv(out/'feature_importance_spatial.csv')
spavg=spimp.groupby('feature').agg(spatial_mean_importance=('importance','mean'),spatial_mean_rank=('rank','mean'),spatial_folds=('cv_fold','nunique')).reset_index()
combined=imp.merge(spavg,on='feature',how='outer'); combined.to_csv(out/'feature_importance_cross_scheme.csv',index=False)
from scipy.stats import spearmanr
paired=combined.dropna(subset=['mean_rank','spatial_mean_rank']); rho=float(spearmanr(paired.mean_rank,paired.spatial_mean_rank).statistic)
ta=set(imp.nsmallest(5,'mean_rank').feature); sa=set(spavg.nsmallest(5,'spatial_mean_rank').feature); jac=len(ta&sa)/len(ta|sa)
pd.DataFrame([{'comparison':'temporal_vs_spatial','spearman_rho_mean_rank':rho,'jaccard_top5':jac,'n_features':len(paired),'temporal_top5':';'.join(sorted(ta)),'spatial_top5':';'.join(sorted(sa)),'notes':'Ranks average across five folds per scheme; importance is conditional on each model and validation cohort.'}]).to_csv(out/'feature_importance_ranking_stability.csv',index=False)
for src,dst in [('ablation.csv','ablation'),('training_size_sensitivity.csv','training_size_sensitivity')]:
 d=pd.read_csv(out/src); d=d[d.metric.eq('rmse')]
 if dst=='ablation': d.groupby('variant').value.agg(['mean','std','count']).reset_index().to_csv(out/'ablation_summary.csv',index=False); x='variant'
 else: d.groupby('fraction').value.agg(['mean','std','count']).reset_index().to_csv(out/'training_size_sensitivity_summary.csv',index=False); x='fraction'
 groups=list(d.groupby(x,sort=True)); fig1,ax=plt.subplots(figsize=(8,4)); labels=[str(k) for k,_ in groups]; means=[g.value.mean() for _,g in groups]; sd=[g.value.std(ddof=1) for _,g in groups]; ax.bar(labels,means,yerr=sd,color='#3977a8',capsize=3); ax.set(ylabel='RMSE (µg/m³; mean ± SD across folds)',title=dst.replace('_',' ').title()); ax.tick_params(axis='x',rotation=20); fig1.tight_layout(); fig1.savefig(fig/f'{dst}.png',dpi=180); plt.close(fig1)
print('temporal_predictions',len(pred),'importance_features',len(imp),'spatial_folds',len(sp))
