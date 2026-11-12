from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

P=Path('presentation'); P.mkdir(exist_ok=True)
RESULTS=Path('results/16'); FIGURES=RESULTS/'figures'; FIGURES.mkdir(parents=True,exist_ok=True)
diagnostic=pd.read_csv(RESULTS/'diagnostic_cv_results.csv')
optuna_folds=pd.read_csv(RESULTS/'temporal_cv_optuna_results.csv')
ablation=pd.read_csv(RESULTS/'ablation_summary.csv')
tail=pd.read_csv(RESULTS/'extreme_event_weighting_summary.csv')

baseline_labels=['CNN-LSTM Optuna','Persistence','Seasonal naive']
baseline_frames=[
    optuna_folds.metric_rmse,
    diagnostic[(diagnostic.model_id=='persistence')&(diagnostic.cv_scheme=='temporal_panel_expanding')&(diagnostic.metric=='rmse')].value,
    diagnostic[(diagnostic.model_id=='seasonal_naive')&(diagnostic.cv_scheme=='temporal_panel_expanding')&(diagnostic.metric=='rmse')].value,
]
baseline_values=[float(values.mean()) for values in baseline_frames]
baseline_errors=[float(values.std(ddof=1)) for values in baseline_frames]
fig,ax=plt.subplots(figsize=(7,4.2))
ax.bar(baseline_labels,baseline_values,yerr=baseline_errors,capsize=5,color=['#087F80','#E09F3E','#546575'])
ax.set_ylabel('Mean fold RMSE (µg/m³)'); ax.set_title('Temporal development CV (5 folds)')
ax.tick_params(axis='x',rotation=12); ax.grid(axis='y',alpha=.25)
fig.tight_layout(); fig.savefig(FIGURES/'temporal_baseline_comparison.png',dpi=180); plt.close(fig)

fig,ax=plt.subplots(figsize=(7,4.2))
ax.bar(ablation.variant,ablation['mean'],yerr=ablation['std'],capsize=5,color='#087F80')
ax.set_ylabel('Mean fold RMSE (µg/m³)'); ax.set_title('Available-feature ablation (5 folds)')
ax.tick_params(axis='x',rotation=18); ax.grid(axis='y',alpha=.25)
fig.tight_layout(); fig.savefig(FIGURES/'available_feature_ablation.png',dpi=180); plt.close(fig)

tail_rmse=tail[(tail.metric=='rmse')&(tail.model_id.isin(['CNN_LSTM_OPTUNA_REFERENCE','CNN_LSTM_WEIGHTED_MSE']))]
tail_order=['low_le25','high_25_50','extreme_gt50']
tail_means=tail_rmse.pivot(index='concentration_band',columns='model_id',values='fold_mean').reindex(tail_order)
fig,ax=plt.subplots(figsize=(7,4.2))
tail_means.plot(kind='bar',ax=ax,color=['#546575','#087F80'])
ax.set_ylabel('Mean fold RMSE (µg/m³; log scale)'); ax.set_yscale('log'); ax.set_title('Tail weighting trade-off')
ax.set_xticklabels(['≤25','25–50','>50'],rotation=0); ax.grid(axis='y',alpha=.25)
ax.legend(['Unweighted MSE','Weighted MSE'])
fig.tight_layout(); fig.savefig(FIGURES/'extreme_event_weighting.png',dpi=180); plt.close(fig)

prs=Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5)
NAVY=RGBColor(16,36,58); TEAL=RGBColor(8,127,128); INK=RGBColor(28,43,55); MUTED=RGBColor(84,101,117); WHITE=RGBColor(255,255,255)
def txt(s,x,y,w,h,text,size=21,color=INK,bold=False):
 b=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); f=b.text_frame; f.word_wrap=True
 for i,line in enumerate(text.split('\n')):
  p=f.paragraphs[0] if i==0 else f.add_paragraph(); p.text=line;p.font.name='Aptos';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=color;p.space_after=Pt(9)
 return b
def note(s,body):
 try:s.notes_slide.notes_text_frame.text=body
 except Exception:pass
def base(title):
 s=prs.slides.add_slide(prs.slide_layouts[6]); s.background.fill.solid();s.background.fill.fore_color.rgb=WHITE
 txt(s,.65,.35,12,.6,title,28,NAVY,True); bar=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(.65),Inches(1.08),Inches(1),Inches(.06));bar.fill.solid();bar.fill.fore_color.rgb=TEAL;bar.line.fill.background()
 txt(s,.65,7.13,12,.2,f'GROUP 16  |  CNN-LSTM  |  {len(prs.slides):02d}',9,MUTED)
 return s
def content(title,body,speaker,image=None):
 s=base(title)
 if image:
  txt(s,.8,1.5,5.8,5.2,body,20)
  s.shapes.add_picture(str(image),Inches(7),Inches(1.5),width=Inches(5.55),height=Inches(4.9))
 else: txt(s,.9,1.55,11.4,5.2,body,22)
 note(s,speaker); return s

s=prs.slides.add_slide(prs.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=NAVY
txt(s,.9,1.1,11.4,1.5,'One-Day-Ahead\nPM2.5 Forecasting',36,WHITE,True)
txt(s,.95,3.05,11,.6,'CNN-LSTM  |  Group 16',24,RGBColor(98,198,194),True)
txt(s,.95,4.25,10.5,1.2,'Historical future result is frozen.\nDevelopment CV is scored; material limitations remain.',22,WHITE)
note(s,'Introduce the task and explain that development CV is now scored, while licensing and some rubric evidence remain open.')
content('Why forecast PM2.5?','Daily forecasts can support air-quality monitoring.\n\nA single split does not show whether performance holds across time or stations.\n\nInputs must be available when the forecast is issued.','Explain the motivation and the importance of a clear forecast origin.')
content('Dataset and coverage','40,983 cleaned observations\n95 stations with finite coordinates\n2016-11-25 to 2026-04-06\nTarget: PM2.5 (µg/m³)\n\nCompiled-data redistribution terms remain unresolved.','Name the source and be direct about the rights caveat. Coordinates show station coverage, not pollution concentration everywhere.','results/16/eda/station_map.png')
content('Data quality and EDA','735 target rows removed by stated bounds\nNo duplicate station-day keys after cleaning\nNO2/CO: about 5.9% missing; O3: about 6.0%\nMedian: 7.85 µg/m³; 99th percentile: 33.56\n352 station-relative high-value flags retained for review\nNo invalid targets/coordinates or duplicate station-days in dev.','Explain the right-skew and missingness. Do not treat correlation as causation.','results/16/figures/pm25_distribution.png')
content('Forecast contract','End of day t → predict station PM2.5 at t+1\nSeven consecutive daily history steps\nSequences stay within station and reject date gaps\nCalendar terms + past target history\nFuture exogenous values excluded without availability evidence','Describe exactly what the model receives and predicts.')
content('CNN-LSTM architecture','Input 7 days × 7 features → causal Conv1D (16, kernel 3) → LayerNorm → ReLU → Dropout 0.347 → MaxPool → LSTM (64) → Dense (32, ReLU) → scalar PM2.5 output\n\nAdam; MSE; learning rate 0.000775; 45 epochs on development data.','This is the saved Optuna development model, not the unavailable historical model. Pooling can be disabled for future-step-preservation experiments.')
content('Development validation and baselines','Optuna temporal CV: 8.520 ± 6.034 µg/m³\nPersistence: 8.771 ± 8.984\nWeekly seasonal naive: 10.798 ± 10.039\nPaired exact Wilcoxon p=.625\n\nFive folds; no significant gain over persistence.','Persistence uses PM2.5 at forecast origin t. The five-fold test is low power; report the result as inconclusive.',FIGURES/'temporal_baseline_comparison.png')
content('Frozen historical estimate','RMSE  6.0272 µg/m³\nMAE   4.0240 µg/m³\nR²       0.2394\nN         1,716\n\nReconstruction reproduces all saved predictions (max |Δ| 7.1e-15).','This is output equivalence from a development-only refit and a prediction-array check. No holdout labels were read. Original binary weights cannot be verified byte-for-byte.')
content('Available-feature ablation','Without calendar: ΔRMSE −0.467 (95% fold CI −0.957 to −0.088)\nWithout lag/rolling: ΔRMSE +0.617 (0.249 to 1.022)\nWithout rolling: ΔRMSE −0.177 (−0.337 to −0.043)\n\nFive temporal folds; no retuning.','These compare the approved lag/rolling set without retuning; unverified exogenous channels are tested separately in nested folds and did not improve results.',FIGURES/'available_feature_ablation.png')
content('Extreme-event weighting','Weighted MSE helps the >25 bands slightly, but worsens overall RMSE 8.520 → 8.600 and ≤25 RMSE 4.423 → 4.622.\n\nOnly 7 OOF cases exceeded 50 µg/m³.','The exploratory training weights were 1x for ≤25, 2x for >25–50, and 4x for >50. The small tail sample makes the apparent benefit unstable.',FIGURES/'extreme_event_weighting.png')
content('Sensitivity and feature-family experiments','Nested feature-family selection: mean-fold RMSE 8.378; MAE 3.784.\nFixed no-calendar + lag-1 + rolling-3: 7.894 / 3.396 vs reference 8.520 / 3.978.\nPooled ΔRMSE −0.250 (cluster CI −0.547 to −0.125); fold p=.063, Holm p=.313.\nVs persistence: p=.813, Holm p=1.00.\n\n3-day vs 7-day RMSE 9.579 vs 9.571. Tail n>50 = 7; MAE 143.4 vs persistence 127.2.','Feature selection varied across folds; this remains exploratory with only five dependent folds. Lagged environmental and same-day spatial neighbor predictors failed to improve target-only OOF scores. Architecture results were mixed; tail estimates are unstable.')
content('Spatial validation and predictor checks','Held-out station groups (conditional on historical sequences):\nRMSE 8.700 | MAE 3.287 | R² .154 | n=3,349\n\nAdding 25/50/100-km same-day neighbor PM raised OOF RMSE to 9.361/9.439/9.408 vs target history 9.322.\nLagged environmental groups also failed to improve.\n\nTemporal/spatial MAPE 50.14% / 42.13%; unstable near zero.','Temporal and spatial OOF results are separate evaluation schemes and not interchangeable. The spatial split does not test a station with no historical input. MAPE includes all positive targets but remains sensitive to small denominators.')
content('Limitations and readiness','Compiled-data license and omitted-row rule unresolved\nExogenous AOD/gas/meteorology blocked pending timing evidence\nLead-lag associations exploratory, not predictive validation\nOnly six stations support the longer formal and lead-lag diagnostics','The audit verified all bundled rows against source files, but cannot reconstruct the publisher’s exact ETL. ACF/PACF use 63 stations; formal/lead-lag evidence is restricted to six stations.')
content('Conclusion and readiness','Development CV, temporal and spatial baselines, nested feature-family comparisons, provenance audits and prediction-equivalent historical reconstruction are saved.\n\nPredictive gain over persistence remains nonsignificant; high-concentration errors are poor.\n\nLicensing, inclusion logic and issue-time feature availability remain open.','Close with what is supported: correct one-day alignment, exact prediction reproduction and mixed, nonsignificant model results.')
prs.save(P/'final_presentation.pptx')
notes='''# Speaker notes

1. Introduce Group 16 and state that development validation is measured, with open data and rubric limitations.
2. Explain the motivation and why time/station validation matters.
3. Describe dataset size, dates, station coverage and unresolved license terms.
4. Explain cleaning, missingness and skew. The station-relative IQR screen flagged observations for review but did not automatically delete them.
5. Define forecast origin, horizon, lookback and issue-time inputs.
6. Explain the saved Optuna CNN-LSTM architecture and training configuration.
7. Persistence uses PM2.5 at the forecast origin. Five-fold paired evidence does not show significant improvement.
8. Historical metrics remain frozen; a development-only reconstruction matches outputs but cannot prove original binary identity.
9. Available-feature ablation results use the same folds; unverified exogenous variables remain excluded.
10. Explain the small weighting trade-off and that only seven OOF targets exceeded 50 µg/m³.
11. Feature-family selection varied by fold. The fixed no-calendar lag-plus-roll3 family has the best observed mean-fold score, but corrected fold tests remain inconclusive and no variant is established as superior.
12. Temporal and spatial diagnostics are distinct. Spatial folds evaluate held-out station groups conditional on historical sequences. MAPE is supplementary and unstable near zero.
13. Six stations support the longer formal/lead-lag diagnostics; these associations are exploratory rather than validated predictive effects.
14. Close with the remaining license, inclusion-rule, issue-time availability and generalization limits.
'''
(P/'speaker_notes.md').write_text(notes,encoding='utf-8')
print('slides',len(prs.slides))
