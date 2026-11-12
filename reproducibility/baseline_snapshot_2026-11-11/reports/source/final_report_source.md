# One-Day-Ahead PM2.5 Forecasting - CNN-LSTM, Group 16

**Development evidence complete; material data limitations disclosed.** This editable source accompanies `../final_report.pdf`. Historical future metrics are frozen and reported separately from development experiments.

## One-Day-Ahead PM2.5 Forecasting

CNN-LSTM | Group 16

**Final technical report and evidence audit**

**Development evidence complete; material data limitations disclosed**

Five-fold temporal and five-fold station-grouped CNN-LSTM development evaluations are scored. A development checkpoint was trained and loaded successfully. The package discloses unresolved dataset redistribution rights, an unverified upstream joining procedure, modest predictive skill, and the irreproducibility of the legacy future result. Group 16 scope is CNN-LSTM only, following the owner clarification and group-specific guide.

## 1. Abstract and study scope

This project forecasts daily station-level PM2.5 using a CNN-LSTM. At the end of day t, the model predicts the station measurement at t+1. The cleaned source has 40,983 observations across 95 stations from 2016-11-25 through 2026-04-06. Development data before 2024-12-08 were evaluated with five temporal folds (2,867 out-of-fold predictions) and five station-group folds (3,349 predictions). Mean fold RMSE was 8.455 µg/m³ temporally and 7.962 µg/m³ spatially; fold-to-fold variability was large. Seed, feature ablation, training-size sensitivity, permutation importance, and residual analyses were also run. A 3,349-sequence development checkpoint was saved and loaded.

The historical future result remains frozen: 1,716 saved predictions, RMSE 6.0272 µg/m³, MAE 4.0240 µg/m³ and R² 0.2394. Its checkpoint is unavailable, and the historical result is not independently reproducible. No future-evaluation code was run during this work. Keywords: PM2.5; forecasting; CNN-LSTM; temporal validation; spatial generalization.

## 2. Motivation and research questions

Daily PM2.5 forecasts may support air-quality monitoring. A result from one split does not establish reliability across time or locations. The assignment calls for data-quality analysis, leakage control, temporal and spatial validation, robustness, explainability and reporting of limitations [1].

Questions: (1) How should one-day-ahead station forecasting be defined using fields demonstrably available at issue time? (2) How does CNN-LSTM perform across later development periods and unseen stations? (3) How do sparse station records, target skew and missing predictor values constrain interpretation?

The CNN-LSTM guide identifies this architecture for the group [2]. The project-owner clarification states that Group 16 develops CNN-LSTM only. Generic codes such as G16 or G21 in the broad brief refer to model-family labels and do not prove a student-group mapping.

## 3. Dataset and provenance

The active input is `combined_gee_pm25.csv`, a prepared table combining PM2.5 with environmental attributes. The manifest identifies Kaggle’s “Air Quality - Mediterranean Basin (2016-2026)” dataset by Sarah Ameziane, version 1 [3]. It has 41,718 input rows and 15 source columns. The cleaned table has 40,983 rows, 19 columns, 95 station IDs, and UTC dates from 2016-11-25 to 2026-04-06. All 95 station IDs have finite coordinates.

The available package does not include an auditable transformation log that proves the upstream join key or nearest-coordinate/date operation. This report calls the source a prepared combined table and does not invent a joining procedure.

Kaggle reports the license as “Other (specified in description).” A review on 2026-11-11 of the Kaggle page and official OpenAQ, Copernicus and NASA terms found that source-specific licenses/terms apply; general platform statements do not clear this compiled bundle. OpenAQ specifically directs users to the original data-provider terms. The package does not claim unrestricted redistribution rights [4-7].

## 4. Cleaning and exploratory analysis

The teacher specifies removal of PM2.5 values ≤0 or ≥500 before analysis [1]. The audit removed 727 values ≤0 and 8 values ≥500 (735 total). No timestamp/target parsing failures or duplicate station-day keys remained. Four same-name/day records at different coordinates remain separate stations.

Median target is 7.85 µg/m³; the 95th percentile is 21.65, 99th is 33.56, and the largest retained value is 484.98. About 2.98% of observations exceed 25 µg/m³ and 0.285% exceed 50 µg/m³. The right tail can contribute disproportionately to RMSE.



## 4.1 EDA: distribution and coverage

NO2 and CO each have 2,407 missing values (about 5.9%); O3 has 2,448 (about 6.0%). The development station-month missingness table contains 6,537 station-month-pollutant rows. On each station’s longest uninterrupted development segment, 63 stations had at least 14 daily observations for ACF/PACF; six had at least 30 observations for ADF, KPSS and Ljung-Box tests. Descriptive pollutant/PM2.5 correlations at offsets from -7 to +7 days are saved in `lead_lag_summary_development.csv`; sparse pollutant coverage yielded sufficient valid pairs at only six stations. These associations do not establish causation or predictor availability at issue time.

## 5. Forecast definition and preprocessing

For station s and day t, the forecast uses data through end of day t to predict PM2.5 on t+1. Seven consecutive daily rows stay within station and windows with gaps are rejected. Only calendar cycles, PM2.5 lag 1 and rolling 3/7-day means are used. Weather, trace gases and AOD remain excluded: their observation dates exist, but availability by forecast issue time is unverified. The sequence preprocessor fits median imputation and standardization on each training fold only, with missingness indicators. The reserved historical future block remains frozen; no future-test evaluation was run.

## 6. CNN-LSTM implementation

The model is causal Conv1D → LayerNormalization → ReLU → Dropout → optional MaxPooling1D → LSTM → Dense(ReLU) → scalar output. Pooling can be disabled to preserve every time step; the existing default remains pool size 2. The saved Optuna development checkpoint uses 16 filters, kernel 3, dropout 0.347, 64 LSTM units, 32 dense units, Adam learning rate 0.000775 and MSE. It trained 3,349 development sequences for 45 epochs on seven daily steps and seven features (with two missingness indicators after preprocessing). It was saved, reloaded and smoke-tested in-sample; it is not the unavailable historical model.

## 7. Validation design and historical evaluation

Development cutoff is exclusive at 2024-12-08. Five expanding temporal folds use whole dates and a one-day embargo; five station-group folds test held-out stations conditional on historical inputs, not cold-start prediction. Optuna used 50 inner-validation trials per temporal fold. The selected CNN-LSTM mean fold RMSE is 8.520 µg/m³ (SD 6.034; 95% fold-bootstrap interval 4.853-14.086; 2,867 OOF predictions). Corrected persistence is 8.771 (SD 8.984); seasonal naive is 10.798 (SD 10.039). CNN-LSTM minus persistence is -0.251 µg/m³ (paired 95% interval -2.960 to 1.530; Holm-adjusted p=1.000, five folds): no statistically significant difference was established. Fold-bootstrap uncertainty is descriptive because folds are few and dependent.

The frozen historical future score remains RMSE 6.027207 µg/m³, MAE 4.024034 µg/m³, R² 0.239368 on 1,716 saved predictions. It was not recalculated, is not directly comparable to development CV, and has no recoverable checkpoint.

## 8. Development results and diagnostics

Across the three seeds, mean fold RMSE ranged from 8.338 to 8.415; training-size changes were modest relative to fold variability. The available-feature ablation results (fixed fold-selected architecture/epochs, no retuning) are shown below. These features exclude exogenous fields: the full required AOD/gas/meteorology axes are blocked until timing is verified.

A 2x weight for PM2.5 >25 and 4x for >50 reduced mean fold RMSE in the 25-50 band from 19.524 to 18.888, and in the >50 band from 166.521 to 165.693 (only seven >50 cases in two folds). Overall mean fold RMSE rose from 8.520 to 8.600, and RMSE at ≤25 rose from 4.423 to 4.622. This exploratory trade-off does not justify replacing the unweighted model.

## 9. Interpretation, reproducibility and conclusion

The Optuna OOF prediction SD was lower than the actual-target SD; stratified errors show underprediction in the sparse high-concentration tail. Seven >50 cases are too few for a stable conclusion. Temporal permutation importance is conditional on the fitted model; correlated channels can substitute for one another. MAPE is omitted because near-zero values make percentage errors unstable.

The safe suite passed 49 tests; `tests/test_final_evaluation.py` remained excluded to avoid invoking its single-use future evaluator. Validation used workspace-local source via PYTHONPATH without changing the active package installation. Python 3.12.10 and TensorFlow 2.19.1 ran on CPU. The 3,349-sequence Optuna development checkpoint and preprocessor reloaded successfully; its 20-row smoke test was in-sample, not an evaluation.

**Conclusion:** corrected baselines, nested Optuna evidence, available-feature ablation, weighted-tail analysis, station-month missingness, uninterrupted-segment time-series tests and descriptive lead-lag summaries are saved. Pollutant coverage limits lead-lag evidence to six stations. Predictive skill remains modest; no significant improvement over persistence was established. Compiled-data rights, upstream join provenance and exogenous forecast-time availability remain unresolved. The historical future checkpoint cannot be recovered, so its frozen score is not reproducible.

References: [1] Supplied “Updated version PM25 Forecasting Mini Project,” pp. 1-11. [2] Supplied “PM25 CNN-LSTM Layers and Libraries,” pp. 1-2. [3] Ameziane, “Air Quality - Mediterranean Basin (2016-2026),” Kaggle, v1, https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026. [4] OpenAQ, Terms of Use, https://docs.openaq.org/about/terms. [5] OpenAQ, Licenses resource, https://docs.openaq.org/resources/licenses. [6] Copernicus Climate Data Store, Terms of Use, https://cds.climate.copernicus.eu/licences/terms-of-use-cds. [7] NASA Science, Information Policy, https://science.nasa.gov/researchers/science-information-policy.

## Appendix A. Evidence and remaining work

Evidence: corrected diagnostic OOF baselines and comparisons; nested Optuna folds and checkpoint; available-feature ablation and CI summary; extreme-event weighted and stratified errors; station-month missingness, continuous-segment and descriptive lead-lag diagnostics are saved under `results/16/`.

Unresolved: confirm compiled-data rights; document the upstream join; verify exogenous data delivery timing before expanding features; recover the historical checkpoint/configuration if possible. Frozen future artifacts were not changed.

## Results tables

| Scheme | Metric | Fold mean | Fold SD | Descriptive 95% interval | OOF N |
|---|---:|---:|---:|---:|---:|
| temporal_panel_expanding | RMSE | 8.455 | 6.161 | 4.670-13.885 | 2867 |
| spatial_group_kfold | RMSE | 7.962 | 4.143 | 4.972-11.527 | 3349 |

## Corrected temporal baseline comparisons

| Model | Mean fold RMSE | Fold SD | Folds |
|---|---:|---:|---:|
| CNN-LSTM Optuna | 8.520 | 6.034 | 5 |
| Persistence | 8.771 | 8.984 | 5 |
| Seasonal naive | 10.798 | 10.039 | 5 |

CNN-LSTM minus persistence: -0.251 µg/m³ (95% interval -2.960 to 1.530; Holm-adjusted p=1.000; 5 folds).

## Available-feature ablation

| Variant | Mean RMSE | Fold SD | Δ vs full | 95% Δ interval |
|---|---:|---:|---:|---:|
| full | 8.455 | 6.161 | — | — |
| without_calendar | 7.988 | 6.486 | -0.467 | -0.957 to -0.088 |
| without_lag_and_rolling | 9.072 | 5.972 | 0.617 | 0.249 to 1.022 |
| without_rolling | 8.278 | 6.234 | -0.177 | -0.337 to -0.043 |

## Extreme-event weighting

| Band | N | Unweighted RMSE | Weighted RMSE | Weighted − unweighted (95% fold interval) |
|---|---:|---:|---:|---:|
| all | 2867 | 8.520 | 8.600 | 0.080 (-0.094, 0.308) |
| extreme_gt50 | 7 | 166.521 | 165.693 | -0.828 (-1.590, -0.065) |
| high_25_50 | 56 | 19.524 | 18.888 | -0.636 (-0.825, -0.447) |
| low_le25 | 2804 | 4.423 | 4.622 | 0.200 (0.004, 0.443) |

## References

[1] Supplied general PM2.5 forecasting mini-project assignment PDF.
[2] Supplied PM2.5 CNN-LSTM layers and libraries guide.
[3] Ameziane, Air Quality - Mediterranean Basin (2016-2026), Kaggle, version 1. https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026.
[4] OpenAQ Terms of Use, https://docs.openaq.org/about/terms.
[5] OpenAQ Licenses resource, https://docs.openaq.org/resources/licenses.
[6] Copernicus CDS Terms of Use, https://cds.climate.copernicus.eu/licences/terms-of-use-cds.
[7] NASA Science Information Policy, https://science.nasa.gov/researchers/science-information-policy.
