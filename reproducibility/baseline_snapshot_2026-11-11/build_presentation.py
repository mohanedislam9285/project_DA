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
content('Data quality and EDA','735 target rows removed by stated bounds\nNo duplicate station-day keys after cleaning\nNO2/CO: about 5.9% missing; O3: about 6.0%\nMedian: 7.85 µg/m³; 99th percentile: 33.56\nRare high values can influence RMSE.','Explain the right-skew and missingness. Do not treat correlation as causation.','results/16/figures/pm25_distribution.png')
content('Forecast contract','End of day t → predict station PM2.5 at t+1\nSeven consecutive daily history steps\nSequences stay within station and reject date gaps\nCalendar terms + past target history\nFuture exogenous values excluded without availability evidence','Describe exactly what the model receives and predicts.')
content('CNN-LSTM architecture','Input 7 days × 7 features → causal Conv1D (16, kernel 3) → LayerNorm → ReLU → Dropout 0.347 → MaxPool → LSTM (64) → Dense (32, ReLU) → scalar PM2.5 output\n\nAdam; MSE; learning rate 0.000775; 45 epochs on development data.','This is the saved Optuna development model, not the unavailable historical model. Pooling can be disabled for future-step-preservation experiments.')
content('Development validation and baselines','Optuna temporal CV: 8.520 ± 6.034 µg/m³\nPersistence: 8.771 ± 8.984\nWeekly seasonal naive: 10.798 ± 10.039\nCNN-LSTM vs persistence: Δ −0.251; Holm p=1.000\n\nFive dependent folds; interval is wide.','Persistence now uses the last observation at forecast origin t. The CNN-LSTM/persistence difference is not statistically significant after Holm correction.',FIGURES/'temporal_baseline_comparison.png')
content('Frozen historical estimate','RMSE  6.0272 µg/m³\nMAE   4.0240 µg/m³\nR²       0.2394\nN         1,716\n\nSaved predictions begin 2024-12-12; reserved block begins 2024-12-08. Historical checkpoint unavailable.','Label these as historical saved metrics. No future evaluation was rerun and no saved checkpoint was found.')
content('Available-feature ablation','Without calendar: ΔRMSE −0.467 (95% fold CI −0.957 to −0.088)\nWithout lag/rolling: ΔRMSE +0.617 (0.249 to 1.022)\nWithout rolling: ΔRMSE −0.177 (−0.337 to −0.043)\n\nFive temporal folds; no retuning.','These are ablations of the currently approved feature set only; unverified exogenous variables are excluded.',FIGURES/'available_feature_ablation.png')
content('Extreme-event weighting','Weighted MSE helps the >25 bands slightly, but worsens overall RMSE 8.520 → 8.600 and ≤25 RMSE 4.423 → 4.622.\n\nOnly 7 OOF cases exceeded 50 µg/m³.','The exploratory training weights were 1x for ≤25, 2x for >25–50, and 4x for >50. The small tail sample makes the apparent benefit unstable.',FIGURES/'extreme_event_weighting.png')
content('Limitations and next steps','Compiled-data license and upstream join remain unresolved\nExogenous AOD/gas/meteorology ablations blocked pending timing evidence\nLead-lag associations are descriptive and limited by pollutant coverage\nHistorical future checkpoint/configuration is unavailable','ACF/PACF were computed on uninterrupted development segments for 63 stations. Formal stationarity tests and adequate pollutant lead-lag pairs were available for only six stations. Lead-lag correlations do not establish causation or forecast-time availability.')
content('Conclusion and readiness','Development CV, nested tuning, diagnostics and a reloadable development checkpoint are saved. Predictive skill is modest; no significant improvement over persistence was established.\n\nHistorical future evidence remains frozen and is not reproducible from the available checkpoint.\n\nMaterial limitations: compiled-data rights, upstream join provenance, forecast-time availability of exogenous features, and sparse lead-lag coverage.','Close by distinguishing completed development evidence from unresolved provenance and legacy-evaluation limitations.')
prs.save(P/'final_presentation.pptx')
notes='''# Speaker notes\n\n1. Introduce Group 16 and state that development validation is now measured, with open data and rubric limitations.\n2. Explain the public-health motivation and need for time/station validation.\n3. Describe dataset size, dates, station locations and unresolved license terms.\n4. Explain cleaning, missingness and skew; do not overinterpret correlation.\n5. Define forecast origin, horizon, lookback and available inputs.\n6. Explain the actual CNN-LSTM architecture and configured training settings.\n7. Persistence uses PM2.5 at the forecast origin, not the next-day target. The five-fold CNN-LSTM/persistence difference is not statistically significant after correction.\n8. Metrics are frozen historical values; the checkpoint was not recovered and the evaluation was not repeated.\n9. Report the available-feature ablation and its five-fold confidence intervals; exogenous inputs were excluded.\n10. Explain the tail-weighting trade-off and the fact that only seven validation predictions exceed 50 µg/m³.\n11. Explain why AOD/gas/meteorology ablations cannot be claimed yet, and why licensing and join provenance remain open.
12. Close by distinguishing completed development evidence from unresolved provenance and legacy-evaluation limitations.\n'''
(P/'speaker_notes.md').write_text(notes,encoding='utf-8')
print('slides',len(prs.slides))
