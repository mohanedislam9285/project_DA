from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pandas as pd

OUT=Path('reports'); (OUT/'source').mkdir(parents=True,exist_ok=True)
styles=getSampleStyleSheet(); navy=colors.HexColor('#10243A'); teal=colors.HexColor('#087F80')
if Path('C:/Windows/Fonts/arial.ttf').exists():
 pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
 pdfmetrics.registerFont(TTFont('Arial-Bold', 'C:/Windows/Fonts/arialbd.ttf'))
 pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial',boldItalic='Arial-Bold')
 FONT='Arial'
else: FONT='Helvetica'
styles.add(ParagraphStyle(name='T',parent=styles['Title'],fontName=FONT,fontSize=24,leading=30,textColor=navy,alignment=1))
styles.add(ParagraphStyle(name='H',parent=styles['Heading1'],fontName=FONT,fontSize=16,leading=20,textColor=navy,spaceAfter=8))
styles.add(ParagraphStyle(name='B',parent=styles['BodyText'],fontName=FONT,fontSize=9.5,leading=14,spaceAfter=7))
styles.add(ParagraphStyle(name='S',parent=styles['BodyText'],fontName=FONT,fontSize=8,leading=11,textColor=colors.HexColor('#546575')))
p=lambda s,st='B': Paragraph(s,styles[st])
def tab(rows,widths):
 t=Table([[p(str(x),'S') for x in r] for r in rows],colWidths=widths,repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#CBD6DC')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#EAF1F4')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)])); return t
pages=[
('One-Day-Ahead PM2.5 Forecasting','CNN-LSTM | Group 16<br/><br/><b>Final technical report and evidence audit</b><br/><br/><b>Development evidence complete; material data limitations disclosed</b><br/><br/>Five-fold temporal and five-fold station-grouped CNN-LSTM development evaluations are scored. A development checkpoint was trained and loaded successfully. The package discloses unresolved dataset redistribution rights, an unverified upstream joining procedure, modest predictive skill, and the irreproducibility of the legacy future result. Group 16 scope is CNN-LSTM only, following the owner clarification and group-specific guide.'),
('1. Abstract and study scope','This project forecasts daily station-level PM2.5 using a CNN-LSTM. At the end of day t, the model predicts the station measurement at t+1. The cleaned source has 40,983 observations across 95 stations from 2016-11-25 through 2026-04-06. Development data before 2024-12-08 were evaluated with five temporal folds (2,867 out-of-fold predictions) and five station-group folds (3,349 predictions). Mean fold RMSE was 8.455 µg/m³ temporally and 7.962 µg/m³ spatially; fold-to-fold variability was large. Seed, feature ablation, training-size sensitivity, permutation importance, and residual analyses were also run. A 3,349-sequence development checkpoint was saved and loaded.<br/><br/>The historical future result remains frozen: 1,716 saved predictions, RMSE 6.0272 µg/m³, MAE 4.0240 µg/m³ and R² 0.2394. Its checkpoint is unavailable, and the historical result is not independently reproducible. No future-evaluation code was run during this work. Keywords: PM2.5; forecasting; CNN-LSTM; temporal validation; spatial generalization.'),
('2. Motivation and research questions','Daily PM2.5 forecasts may support air-quality monitoring. A result from one split does not establish reliability across time or locations. The assignment calls for data-quality analysis, leakage control, temporal and spatial validation, robustness, explainability and reporting of limitations [1].<br/><br/>Questions: (1) How should one-day-ahead station forecasting be defined using fields demonstrably available at issue time? (2) How does CNN-LSTM perform across later development periods and unseen stations? (3) How do sparse station records, target skew and missing predictor values constrain interpretation?<br/><br/>The CNN-LSTM guide identifies this architecture for the group [2]. The project-owner clarification states that Group 16 develops CNN-LSTM only. Generic codes such as G16 or G21 in the broad brief refer to model-family labels and do not prove a student-group mapping.'),
('3. Dataset and provenance','The active input is `combined_gee_pm25.csv`, a prepared table combining PM2.5 with environmental attributes. The manifest identifies Kaggle’s “Air Quality - Mediterranean Basin (2016-2026)” dataset by Sarah Ameziane, version 1 [3]. It has 41,718 input rows and 15 source columns. The cleaned table has 40,983 rows, 19 columns, 95 station IDs, and UTC dates from 2016-11-25 to 2026-04-06. All 95 station IDs have finite coordinates.<br/><br/>The available package does not include an auditable transformation log that proves the upstream join key or nearest-coordinate/date operation. This report calls the source a prepared combined table and does not invent a joining procedure.<br/><br/>Kaggle reports the license as “Other (specified in description).” A review on 2026-11-11 of the Kaggle page and official OpenAQ, Copernicus and NASA terms found that source-specific licenses/terms apply; general platform statements do not clear this compiled bundle. OpenAQ specifically directs users to the original data-provider terms. The package does not claim unrestricted redistribution rights [4-7].'),
('4. Cleaning and exploratory analysis','The teacher specifies removal of PM2.5 values ≤0 or ≥500 before analysis [1]. The audit removed 727 values ≤0 and 8 values ≥500 (735 total). No timestamp/target parsing failures or duplicate station-day keys remained. Four same-name/day records at different coordinates remain separate stations.<br/><br/>Median target is 7.85 µg/m³; the 95th percentile is 21.65, 99th is 33.56, and the largest retained value is 484.98. About 2.98% of observations exceed 25 µg/m³ and 0.285% exceed 50 µg/m³. The right tail can contribute disproportionately to RMSE.<br/><br/>',),
('4.1 EDA: distribution and coverage','NO2 and CO each have 2,407 missing values (about 5.9%); O3 has 2,448 (about 6.0%). The development station-month missingness table contains 6,537 station-month-pollutant rows. On each station’s longest uninterrupted development segment, 63 stations had at least 14 daily observations for ACF/PACF; six had at least 30 observations for ADF, KPSS and Ljung-Box tests. Descriptive pollutant/PM2.5 correlations at offsets from -7 to +7 days are saved in `lead_lag_summary_development.csv`; sparse pollutant coverage yielded sufficient valid pairs at only six stations. These associations do not establish causation or predictor availability at issue time.'),
('5. Forecast definition and preprocessing','For station s and day t, the forecast uses data through end of day t to predict PM2.5 on t+1. Seven consecutive daily rows stay within station and windows with gaps are rejected. Only calendar cycles, PM2.5 lag 1 and rolling 3/7-day means are used. Weather, trace gases and AOD remain excluded: their observation dates exist, but availability by forecast issue time is unverified. The sequence preprocessor fits median imputation and standardization on each training fold only, with missingness indicators. The reserved historical future block remains frozen; no future-test evaluation was run.'),
('6. CNN-LSTM implementation','The model is causal Conv1D → LayerNormalization → ReLU → Dropout → optional MaxPooling1D → LSTM → Dense(ReLU) → scalar output. Pooling can be disabled to preserve every time step; the existing default remains pool size 2. The saved Optuna development checkpoint uses 16 filters, kernel 3, dropout 0.347, 64 LSTM units, 32 dense units, Adam learning rate 0.000775 and MSE. It trained 3,349 development sequences for 45 epochs on seven daily steps and seven features (with two missingness indicators after preprocessing). It was saved, reloaded and smoke-tested in-sample; it is not the unavailable historical model.'),
('7. Validation design and historical evaluation','Development cutoff is exclusive at 2024-12-08. Five expanding temporal folds use whole dates and a one-day embargo; five station-group folds test held-out stations conditional on historical inputs, not cold-start prediction. Optuna used 50 inner-validation trials per temporal fold. The selected CNN-LSTM mean fold RMSE is 8.520 µg/m³ (SD 6.034; 95% fold-bootstrap interval 4.853-14.086; 2,867 OOF predictions). Corrected persistence is 8.771 (SD 8.984); seasonal naive is 10.798 (SD 10.039). CNN-LSTM minus persistence is -0.251 µg/m³ (paired 95% interval -2.960 to 1.530; Holm-adjusted p=1.000, five folds): no statistically significant difference was established. Fold-bootstrap uncertainty is descriptive because folds are few and dependent.<br/><br/>The frozen historical future score remains RMSE 6.027207 µg/m³, MAE 4.024034 µg/m³, R² 0.239368 on 1,716 saved predictions. It was not recalculated, is not directly comparable to development CV, and has no recoverable checkpoint.'),
('8. Development results and diagnostics','Across the three seeds, mean fold RMSE ranged from 8.338 to 8.415; training-size changes were modest relative to fold variability. The available-feature ablation results (fixed fold-selected architecture/epochs, no retuning) are shown below. These features exclude exogenous fields: the full required AOD/gas/meteorology axes are blocked until timing is verified.<br/><br/>A 2x weight for PM2.5 >25 and 4x for >50 reduced mean fold RMSE in the 25-50 band from 19.524 to 18.888, and in the >50 band from 166.521 to 165.693 (only seven >50 cases in two folds). Overall mean fold RMSE rose from 8.520 to 8.600, and RMSE at ≤25 rose from 4.423 to 4.622. This exploratory trade-off does not justify replacing the unweighted model.'),
('9. Interpretation, reproducibility and conclusion','The Optuna OOF prediction SD was lower than the actual-target SD; stratified errors show underprediction in the sparse high-concentration tail. Seven >50 cases are too few for a stable conclusion. Temporal permutation importance is conditional on the fitted model; correlated channels can substitute for one another. MAPE is omitted because near-zero values make percentage errors unstable.<br/><br/>The safe suite passed 49 tests; `tests/test_final_evaluation.py` remained excluded to avoid invoking its single-use future evaluator. Validation used workspace-local source via PYTHONPATH without changing the active package installation. Python 3.12.10 and TensorFlow 2.19.1 ran on CPU. The 3,349-sequence Optuna development checkpoint and preprocessor reloaded successfully; its 20-row smoke test was in-sample, not an evaluation.<br/><br/><b>Conclusion:</b> corrected baselines, nested Optuna evidence, available-feature ablation, weighted-tail analysis, station-month missingness, uninterrupted-segment time-series tests and descriptive lead-lag summaries are saved. Pollutant coverage limits lead-lag evidence to six stations. Predictive skill remains modest; no significant improvement over persistence was established. Compiled-data rights, upstream join provenance and exogenous forecast-time availability remain unresolved. The historical future checkpoint cannot be recovered, so its frozen score is not reproducible.<br/><br/>References: [1] Supplied “Updated version PM25 Forecasting Mini Project,” pp. 1-11. [2] Supplied “PM25 CNN-LSTM Layers and Libraries,” pp. 1-2. [3] Ameziane, “Air Quality - Mediterranean Basin (2016-2026),” Kaggle, v1, https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026. [4] OpenAQ, Terms of Use, https://docs.openaq.org/about/terms. [5] OpenAQ, Licenses resource, https://docs.openaq.org/resources/licenses. [6] Copernicus Climate Data Store, Terms of Use, https://cds.climate.copernicus.eu/licences/terms-of-use-cds. [7] NASA Science, Information Policy, https://science.nasa.gov/researchers/science-information-policy.'),
('Appendix A. Evidence and remaining work','Evidence: corrected diagnostic OOF baselines and comparisons; nested Optuna folds and checkpoint; available-feature ablation and CI summary; extreme-event weighted and stratified errors; station-month missingness, continuous-segment and descriptive lead-lag diagnostics are saved under `results/16/`.<br/><br/>Unresolved: confirm compiled-data rights; document the upstream join; verify exogenous data delivery timing before expanding features; recover the historical checkpoint/configuration if possible. Frozen future artifacts were not changed.'),
]
story=[]
cvsummary=pd.read_csv('results/16/development_cv_summary.csv')
temporal_folds=pd.read_csv('results/16/temporal_cv_results.csv')
spatial_folds=pd.read_csv('results/16/spatial_cv_results.csv')
ablation_summary=pd.read_csv('results/16/ablation_summary.csv')
size_summary=pd.read_csv('results/16/training_size_sensitivity_summary.csv')
diagnostic_results=pd.read_csv('results/16/diagnostic_cv_results.csv')
optuna_folds=pd.read_csv('results/16/temporal_cv_optuna_results.csv')
comparisons=pd.read_csv('results/16/comparisons.csv')
tail_summary=pd.read_csv('results/16/extreme_event_weighting_summary.csv')
for i,(title,text) in enumerate(pages):
 if i==0: story += [Spacer(1,42*mm),p(title,'T'),Spacer(1,8*mm),p(text),PageBreak()]; continue
 story += [p(title,'H'),p(text)]
 if title.startswith('7. Validation'):
  rows=[['Scheme','Metric','Fold mean','Fold SD','95% bootstrap CI','OOF N']]
  for _,r in cvsummary[cvsummary.metric.eq('rmse')].iterrows(): rows.append([r.cv_scheme,r.metric,f"{r.fold_mean:.3f}",f"{r.fold_sd:.3f}",f"{r.bootstrap_95_ci_lower:.3f}-{r.bootstrap_95_ci_upper:.3f}",str(int(r.total_oof_predictions))])
  story += [tab(rows,[43*mm,18*mm,24*mm,22*mm,36*mm,22*mm]),Spacer(1,3*mm)]
  foldrows=[['Temporal fold','RMSE','MAE','R²','N']]
  t=temporal_folds[temporal_folds.metric.isin(['rmse','mae','r2'])].pivot(index='cv_fold',columns='metric',values='value')
  n=temporal_folds[temporal_folds.metric.eq('rmse')].set_index('cv_fold').n_samples
  for k,r in t.iterrows(): foldrows.append([str(int(k)),f"{r.rmse:.3f}",f"{r.mae:.3f}",f"{r.r2:.3f}",str(int(n.loc[k]))])
  story += [tab(foldrows,[30*mm,27*mm,27*mm,27*mm,27*mm]),Spacer(1,3*mm)]
  baseline_rows=[['Temporal model','Mean fold RMSE','Fold SD','Folds']]
  for model_id,label,source in [
   ('CNN_LSTM_OPTUNA','CNN-LSTM Optuna',optuna_folds),
   ('persistence','Persistence',diagnostic_results),
   ('seasonal_naive','Seasonal naive',diagnostic_results),
  ]:
   if model_id=='CNN_LSTM_OPTUNA':
    values=source.metric_rmse.to_numpy(dtype=float)
   else:
    values=source[(source.model_id.eq(model_id))&(source.cv_scheme.eq('temporal_panel_expanding'))&(source.metric.eq('rmse'))].value.to_numpy(dtype=float)
   baseline_rows.append([label,f'{values.mean():.3f}',f'{values.std(ddof=1):.3f}',str(len(values))])
  story += [tab(baseline_rows,[61*mm,34*mm,30*mm,20*mm])]
  paired=comparisons[(comparisons.comparison_type.eq('temporal_panel_expanding'))&(comparisons.model_a.eq('CNN_LSTM_OPTUNA'))&(comparisons.model_b.eq('persistence'))]
  if not paired.empty:
   r=paired.iloc[0]
   story += [p(f"CNN-LSTM minus persistence: {r.delta:.3f} µg/m³ (95% interval {r.delta_ci_lower:.3f} to {r.delta_ci_upper:.3f}; Holm-adjusted p={r.p_value_corrected:.3f}; {int(r.n_paired_folds)} paired folds). No significant difference was established.",'S')]
 if title.startswith('8. Development'):
  abr=[['Available feature set','Mean RMSE','SD','Δ vs full','95% Δ interval']]
  for _,r in ablation_summary.iterrows():
   ci='—' if pd.isna(r.delta_ci_lower) else f'{r.delta_ci_lower:.3f} to {r.delta_ci_upper:.3f}'
   delta='—' if pd.isna(r.mean_delta_vs_full) else f'{r.mean_delta_vs_full:.3f}'
   abr.append([r.variant,f"{r['mean']:.3f}",f"{r['std']:.3f}",delta,ci])
  story += [tab(abr,[48*mm,24*mm,20*mm,22*mm,42*mm]),Spacer(1,3*mm)]
  sz=[['Chronological training fraction','Mean RMSE','SD']]
  for _,r in size_summary.iterrows(): sz.append([f"{r.fraction:.0%}",f"{r['mean']:.3f}",f"{r['std']:.3f}"])
  story += [tab(sz,[58*mm,30*mm,28*mm]),Spacer(1,3*mm)]
  tail=tail_summary[(tail_summary.model_id.eq('CNN_LSTM_WEIGHTED_MSE'))&(tail_summary.metric.eq('rmse'))]
  tail_rows=[['Target band','N','Unweighted','Weighted','Weighted − unweighted']]
  reference_tail=tail_summary[(tail_summary.model_id.eq('CNN_LSTM_OPTUNA_REFERENCE'))&(tail_summary.metric.eq('rmse'))].set_index('concentration_band')
  for _,r in tail.iterrows():
   ref=reference_tail.loc[r.concentration_band]
   ci='—' if pd.isna(r.delta_ci_lower) else f'{r.mean_delta_vs_reference:.3f} ({r.delta_ci_lower:.3f}, {r.delta_ci_upper:.3f})'
   tail_rows.append([r.concentration_band,str(int(r.total_oof_samples)),f'{ref.fold_mean:.3f}',f'{r.fold_mean:.3f}',ci])
  story += [tab(tail_rows,[38*mm,16*mm,27*mm,27*mm,48*mm]),Spacer(1,3*mm)]
 if title.startswith('4. Cleaning'):
  story += [Image('results/16/eda/station_map.png',width=145*mm,height=82*mm),p('Figure 1. Station location and station median PM2.5; marker size reflects record count. This is not an interpolated concentration surface.','S')]
 if title.startswith('4.1 EDA'):
  story += [Image('results/16/figures/pm25_distribution.png',width=145*mm,height=70*mm),p('Figure 2. Cleaned PM2.5 distribution (µg/m³).','S')]
 story += [PageBreak()]
pdf=OUT/'final_report.pdf'
def footer(c,d):
 c.saveState(); c.setFont(FONT,8); c.setFillColor(colors.HexColor('#546575')); c.drawString(18*mm,9*mm,'Group 16 | CNN-LSTM | DEVELOPMENT EVIDENCE'); c.drawRightString(192*mm,9*mm,str(d.page)); c.restoreState()
SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=16*mm,bottomMargin=18*mm,title='PM2.5 CNN-LSTM Group 16 Report').build(story,onFirstPage=footer,onLaterPages=footer)
source=['# One-Day-Ahead PM2.5 Forecasting - CNN-LSTM, Group 16','','**Development evidence complete; material data limitations disclosed.** This editable source accompanies `../final_report.pdf`. Historical future metrics are frozen and reported separately from development experiments.','']
for title,text in pages:
 source += ['## '+title,'',text.replace('<br/><br/>','\n\n').replace('<b>','**').replace('</b>','**'),'']
source += ['## Results tables','','| Scheme | Metric | Fold mean | Fold SD | Descriptive 95% interval | OOF N |','|---|---:|---:|---:|---:|---:|']
for _,r in cvsummary[cvsummary.metric.eq('rmse')].iterrows(): source.append(f"| {r.cv_scheme} | RMSE | {r.fold_mean:.3f} | {r.fold_sd:.3f} | {r.bootstrap_95_ci_lower:.3f}-{r.bootstrap_95_ci_upper:.3f} | {int(r.total_oof_predictions)} |")
source += ['', '## Corrected temporal baseline comparisons', '', '| Model | Mean fold RMSE | Fold SD | Folds |', '|---|---:|---:|---:|']
for model_id,label,frame in [
 ('CNN_LSTM_OPTUNA','CNN-LSTM Optuna',optuna_folds),
 ('persistence','Persistence',diagnostic_results),
 ('seasonal_naive','Seasonal naive',diagnostic_results),
]:
 if model_id=='CNN_LSTM_OPTUNA': values=frame.metric_rmse.to_numpy(dtype=float)
 else: values=frame[(frame.model_id.eq(model_id))&(frame.cv_scheme.eq('temporal_panel_expanding'))&(frame.metric.eq('rmse'))].value.to_numpy(dtype=float)
 source.append(f"| {label} | {values.mean():.3f} | {values.std(ddof=1):.3f} | {len(values)} |")
paired=comparisons[(comparisons.comparison_type.eq('temporal_panel_expanding'))&(comparisons.model_a.eq('CNN_LSTM_OPTUNA'))&(comparisons.model_b.eq('persistence'))]
if not paired.empty:
 r=paired.iloc[0]
 source += ['', f"CNN-LSTM minus persistence: {r.delta:.3f} µg/m³ (95% interval {r.delta_ci_lower:.3f} to {r.delta_ci_upper:.3f}; Holm-adjusted p={r.p_value_corrected:.3f}; {int(r.n_paired_folds)} folds)."]
source += ['', '## Available-feature ablation', '', '| Variant | Mean RMSE | Fold SD | Δ vs full | 95% Δ interval |', '|---|---:|---:|---:|---:|']
for _,r in ablation_summary.iterrows():
 delta='—' if pd.isna(r.mean_delta_vs_full) else f'{r.mean_delta_vs_full:.3f}'
 ci='—' if pd.isna(r.delta_ci_lower) else f'{r.delta_ci_lower:.3f} to {r.delta_ci_upper:.3f}'
 source.append(f"| {r.variant} | {r['mean']:.3f} | {r['std']:.3f} | {delta} | {ci} |")
source += ['', '## Extreme-event weighting', '', '| Band | N | Unweighted RMSE | Weighted RMSE | Weighted − unweighted (95% fold interval) |', '|---|---:|---:|---:|---:|']
reference_tail=tail_summary[(tail_summary.model_id.eq('CNN_LSTM_OPTUNA_REFERENCE'))&(tail_summary.metric.eq('rmse'))].set_index('concentration_band')
for _,r in tail_summary[(tail_summary.model_id.eq('CNN_LSTM_WEIGHTED_MSE'))&(tail_summary.metric.eq('rmse'))].iterrows():
 delta='—' if pd.isna(r.delta_ci_lower) else f"{r.mean_delta_vs_reference:.3f} ({r.delta_ci_lower:.3f}, {r.delta_ci_upper:.3f})"
 source.append(f"| {r.concentration_band} | {int(r.total_oof_samples)} | {reference_tail.loc[r.concentration_band].fold_mean:.3f} | {r.fold_mean:.3f} | {delta} |")
source += ['', '## References','', '[1] Supplied general PM2.5 forecasting mini-project assignment PDF.','[2] Supplied PM2.5 CNN-LSTM layers and libraries guide.','[3] Ameziane, Air Quality - Mediterranean Basin (2016-2026), Kaggle, version 1. https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026.','[4] OpenAQ Terms of Use, https://docs.openaq.org/about/terms.','[5] OpenAQ Licenses resource, https://docs.openaq.org/resources/licenses.','[6] Copernicus CDS Terms of Use, https://cds.climate.copernicus.eu/licences/terms-of-use-cds.','[7] NASA Science Information Policy, https://science.nasa.gov/researchers/science-information-policy.','']
(OUT/'source'/'final_report_source.md').write_text('\n'.join(source),encoding='utf-8')
(OUT/'source'/'build_report.py').write_text(Path('build_report.py').read_text(encoding='utf-8'),encoding='utf-8')
print(pdf)
