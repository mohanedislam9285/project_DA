# One-Day-Ahead PM2.5 Forecasting - CNN-LSTM, Group 16

**Development evidence complete; material data limitations disclosed.** This editable source accompanies `../final_report.pdf`. Historical future metrics are frozen and reported separately from development experiments.

## One-Day-Ahead PM2.5 Forecasting

CNN-LSTM | Group 16

**Technical report and evidence audit**

**Development evidence with material data limitations disclosed**

Five-fold temporal and station-grouped evaluations, matched-sample model experiments, and row-level provenance audits are saved. A development-only reconstruction reproduces all historical saved predictions, without reading holdout labels; byte-identical original weights cannot be proven. Compiled-data redistribution terms and part of the upstream inclusion logic remain unresolved. Group 16 scope is CNN-LSTM only.

## 1. Abstract and study scope

This project forecasts daily station-level PM2.5 using a CNN-LSTM. At the end of day t, the model predicts station measurement at t+1. The cleaned source has 40,983 observations across 95 stations. The main temporal comparison uses 2,867 paired out-of-fold targets. Mean-fold RMSE was 8.520 for CNN-LSTM and 8.771 for persistence; mean-fold MAE was 3.978 and 3.158. Paired fold tests do not establish a significant gain. Pooling, residual output, and 3-vs-7-day history experiments are mixed and do not yield a reliable winner.

The frozen historical metrics are RMSE 6.0272 µg/m³, MAE 4.0240, R² 0.2394 on 1,716 predictions. A development-only refit reproduces all recorded outputs to a maximum absolute difference of 7.1e-15; the original binary file cannot be proven identical. No holdout labels or new future metrics were used. Keywords: PM2.5; forecasting; CNN-LSTM; temporal validation; provenance.

## 2. Motivation and research questions

Daily PM2.5 forecasts may support air-quality monitoring. A result from one split does not establish reliability across time or locations. The assignment calls for data-quality analysis, leakage control, temporal and spatial validation, robustness, explainability and reporting of limitations [1].

Questions: (1) How should one-day-ahead station forecasting be defined using fields demonstrably available at issue time? (2) How does CNN-LSTM perform across later development periods and unseen stations? (3) How do sparse station records, target skew and missing predictor values constrain interpretation?

The CNN-LSTM guide identifies this architecture for the group [2]. The project-owner clarification states that Group 16 develops CNN-LSTM only. Generic codes such as G16 or G21 in the broad brief refer to model-family labels and do not prove a student-group mapping.

## 2.1 Related work and methodological position

CNN-LSTM studies have reported daily PM2.5 forecasting with richer inputs, including ground observations and meteorological forecasts (Kim et al., 2022), while Bai et al. (2024) studied multiple stations in Qingdao. Their data sources, sites, forecast setups and validation designs differ from this six-station continuous-diagnostic subset within the larger 95-station panel; published scores are therefore context, not a direct benchmark. This project excludes meteorology, satellite and pollutant fields until their issue-time availability is verified. Forecast-evaluation reviews recommend preserving temporal order and using meaningful benchmarks because nonstationarity, short series and weak baselines can make apparent model gains unreliable (Hewamalage et al., 2023). This motivates the expanding temporal folds and persistence comparator. References [9-11].

## 3. Dataset and provenance

The active input is `combined_gee_pm25.csv`, a prepared PM2.5/environmental table from Kaggle’s “Air Quality - Mediterranean Basin (2016-2026),” version 1 [3]. It has 41,718 rows; cleaning retains 40,983 rows across 95 stations (2016-11-25 to 2026-04-06 UTC). All stations have finite coordinates.

The publisher describes nearest-coordinate/date matching within about 0.25 degrees. Our row-level audit finds all 41,718 combined rows correspond to exact PM station/date/coordinates and exact GEE id/date; paired coordinates coincide (0 km), PM values agree within 3.6e-15 µg/m³, and each selected GEE row is independently nearest on that date. This verifies the resulting correspondences, not the upstream ETL implementation or the selection rule for omitted records. Of 88,041 source PM rows, 46,323 are absent from the combined bundle; 12,074 additional PM rows meet the stated approximate distance condition but are omitted. Tie and multi-candidate counts are documented in `docs/DATA_PROVENANCE.md` and machine-readable audit outputs.

The captured Kaggle metadata reports “Other (specified in description),” not a standard license. Component sources have their own use/attribution terms, but the publisher provides no compiled-bundle license or component-level attribution mapping. Redistribution rights therefore remain unresolved [4-8].

## 4. Cleaning and exploratory analysis

The teacher specifies removal of PM2.5 values ≤0 or ≥500 before analysis [1]. The audit removed 727 values ≤0 and 8 values ≥500 (735 total). No timestamp/target parsing failures or duplicate station-day keys remained. Four same-name/day records at different coordinates remain separate stations.

Median target is 7.85 µg/m³; the 95th percentile is 21.65, 99th is 33.56, and the largest retained value is 484.98. About 2.98% of observations exceed 25 µg/m³ and 0.285% exceed 50 µg/m³. The right tail can contribute disproportionately to RMSE.

A development-only QA audit (30,651 rows, 73 stations through 2024-12-07) found no duplicate station-day keys, invalid coordinates, non-finite PM2.5 or values outside 0 &lt; PM2.5 &lt; 500. A station-specific Q3 + 3×IQR screen flagged 352 values across 71 stations with at least 30 readings. These remain in the data: a statistical flag is not proof of sensor failure. The development partition includes 90 values &gt;50, 36 &gt;100 and 11 &gt;250 µg/m³. Flag rows and per-variable missingness are saved in `results/16/cleaning_quality_audit/`.



## 4.1 EDA: distribution and coverage

NO2 and CO each have 2,407 missing values (about 5.9%); O3 has 2,448 (about 6.0%). The development station-month missingness table contains 6,537 station-month-pollutant rows. On each station’s longest uninterrupted development segment, 63 stations had at least 14 daily observations for ACF/PACF; six had at least 30 observations for ADF, KPSS and Ljung-Box tests. Descriptive pollutant/PM2.5 correlations at offsets from -7 to +7 days are saved in `lead_lag_summary_development.csv`; sparse pollutant coverage yielded sufficient valid pairs at only six stations. These associations do not establish causation or predictor availability at issue time.

## 5. Forecast definition and preprocessing

For station s and day t, the forecast uses data through end of day t to predict PM2.5 on t+1. Seven consecutive daily rows stay within station and windows with gaps are rejected. Only calendar cycles, PM2.5 lag 1 and rolling 3/7-day means are used. Weather, trace gases and AOD remain excluded: their observation dates exist, but availability by forecast issue time is unverified. The sequence preprocessor fits median imputation and standardization on each training fold only, with missingness indicators. The reserved historical future block remains frozen; no future-test evaluation was run.

## 6. CNN-LSTM implementation

The model is causal Conv1D → LayerNormalization → ReLU → Dropout → optional MaxPooling1D → LSTM → Dense(ReLU) → scalar output. Pooling can be disabled to preserve every time step; the existing default remains pool size 2. The saved Optuna development checkpoint uses 16 filters, kernel 3, dropout 0.347, 64 LSTM units, 32 dense units, Adam learning rate 0.000775 and MSE. It trained 3,349 development sequences for 45 epochs on seven daily steps and seven features (with two missingness indicators after preprocessing). It was saved, reloaded and smoke-tested in-sample; it is distinct from the development-only reconstruction of the historical prediction model.

## 7. Validation design and historical evaluation

Development cutoff is exclusive at 2024-12-08. Five expanding temporal folds use whole dates and a one-day raw-date embargo; all 2,867 temporal OOF targets are paired across model and persistence. Each sample’s seven-step input, lagged PM2.5 at origin t and label at t+1 were verified. Fold-level RMSE, MAE and R² were recalculated from actual prediction arrays. The original model gives mean-fold RMSE 8.520 µg/m³ (SD 6.034; descriptive fold-bootstrap 95% interval 4.853-13.891) and MAE 3.978 (SD 0.892; interval 3.354-4.731). Persistence gives 8.771 RMSE and 3.158 MAE. Pooled OOF RMSE is 9.571 (station-cluster bootstrap 95% CI 5.365-13.777) for CNN-LSTM and 10.703 (5.109-15.842) for persistence; pooled MAE is 3.927 (3.525-4.393) and 3.050 (2.586-3.660). For paired predictions, CNN-LSTM minus persistence is -1.131 pooled RMSE (72-station cluster 95% CI -2.824 to 0.797) and +0.877 pooled MAE (0.583 to 1.150). Fold RMSE differences are mixed; exact Wilcoxon p=.625 (Holm-adjusted p=1.00); the descriptive fold-bootstrap interval also includes zero. No significant overall improvement was established.

The historical checkpoint file was missing, but a development-only refit using the recorded candidate and selected 22 epochs reproduces all 1,716 stored future predictions (maximum absolute difference 7.1e-15). This validates prediction reproducibility, but cannot establish identity of the original weight bytes. The stored historical score remains frozen; no future labels were read or scored.

## 8. Development results and diagnostics

Across the three seeds, mean fold RMSE ranged from 8.338 to 8.415; training-size changes were modest relative to fold variability. The available-feature ablation results (fixed fold-selected architecture/epochs, no retuning) are shown below. These features exclude exogenous fields: the full required AOD/gas/meteorology axes are blocked until timing is verified.

A 2x weight for PM2.5 >25 and 4x for >50 reduced mean fold RMSE in the 25-50 band from 19.524 to 18.888, and in the >50 band from 166.521 to 165.693 (only seven >50 cases in two folds). Overall mean fold RMSE rose from 8.520 to 8.600, and RMSE at ≤25 rose from 4.423 to 4.622. This exploratory trade-off does not justify replacing the unweighted model.

## 8. Feature-family experiment and finalized development model

To test whether engineered calendar and rolling inputs were adding noise, four feature families were compared in a nested development-only procedure. Each fold used the existing fold hyperparameters; family selection used inner chronological validation and outer scores used identical 2,867 targets. The inner-selected procedure averaged RMSE 8.378 and MAE 3.784 by fold (pooled RMSE 9.551; MAE 3.769), slightly below the original reference by fold average, but it was not better than persistence by paired evidence (exact five-fold RMSE p=.813; Holm-adjusted p=1.00; pooled RMSE-difference station-cluster 95% CI −2.815 to +0.676). Selection varied across folds: full family twice, lag-only twice, and lag plus rolling-3 once.

Across fixed family variants, no-calendar + lag-1 + rolling-3 had the best mean-fold RMSE (7.894) and MAE (3.396), versus reference 8.520/3.978. Its pooled RMSE was 9.322 and MAE 3.462. Paired pooled deltas versus the reference were −0.250 RMSE (station-cluster 95% CI −0.547 to −0.125) and −0.465 MAE (−0.596 to −0.342), conditional on the observed folds. The exact fold-level RMSE test was p=.063 and Holm-adjusted p=.313 across five procedures. The cluster intervals do not account for family search. Against persistence, the exact fold-level RMSE test was p=.813 (Holm p=1.00) and the pooled RMSE-difference cluster interval was −3.065 to +0.383. This is exploratory evidence with model-family multiplicity and only five dependent temporal folds; it does not establish a reliable winner or significant superiority over persistence. Full paired results and the bootstrap method are saved under `results/16/feature_family_experiment/`.

An additional nested feature-timing experiment compared target history with one-day-lagged pollutant/AOD, meteorology and combined environmental channels. No environmental family improved pooled RMSE/MAE over target history (9.322/3.462); the inner-selected mixture was 9.356/3.645. The exact fold test versus persistence was p=.813, with station-cluster RMSE-difference 95% CI −3.094 to +0.533. These channels remain exploratory because source delivery times are undocumented.

A second nested test added same-origin PM2.5 means from other stations within 25, 50 or 100 km. Development coverage was 80.8%, 88.6% and 93.7%; all three fixed-radius models scored worse than target-history-only on the same 2,867 OOF targets (pooled RMSE 9.361, 9.439 and 9.408 versus 9.322). The inner-selected spatial mixture scored 9.386 RMSE / 3.659 MAE. Neighbor readings share the persistence origin-time availability assumption; source delivery timestamps are not available. Neither tested input expansion improved the primary predictor set.

For a deployable development artifact, a final chronological inner split (2,275 train / 1,059 validation sequences) selected no-calendar + lag-1 only. The saved model was then refit on all 3,349 development sequences for 27 epochs. Its inner-split RMSE was 9.224 and MAE 3.600. This is selection evidence, not an independent score; no outer test was used for final selection and no future holdout metric was computed. The new model bundle, preprocessing state, config, seed and fit metadata are saved in `results/16/models/cnn_lstm_development_feature_selected.keras` and adjacent files.

## 8. Model sensitivity experiments

Architecture variants were compared on the same five temporal folds and the same 2,867 samples. Direct output without pooling had mean-fold RMSE 7.901 and MAE 3.382; the reference pooled model had 8.520 and 3.978. Its RMSE paired test is nonsignificant (exact Wilcoxon p=1.00; Holm p=1.00 across variants), and the station-cluster interval for pooled RMSE difference includes zero. Residual output without pooling had RMSE 8.660 and MAE 3.141; exact paired p=.125 (Holm=.75). Nested inner selection among architectures scored worse overall (mean-fold RMSE 9.171). The variants do not support claiming a robust winner.

In the matched 3-versus-7-day experiment (same 2,867 samples), pooled RMSE was 9.579 for three days and 9.571 for seven days; pooled MAE was 3.899 and 3.927. Inner validation selected seven days in four folds. The difference is negligible, so the established seven-day history is retained. A 14-day history supplied only 505 development sequences overall, with validation-fold counts 45, 2, 8, 145 and 255, too sparse for a balanced five-fold comparison.

Original training RMSE versus outer validation RMSE shows substantial gap in folds 1 and 3 (1.728 vs 5.792; 4.043 vs 18.877), consistent with overfitting there. Fold 4 reverses (8.828 vs 5.106), reflecting a harder training distribution and temporal variability rather than a stable generalization pattern. Detailed scores, fitted fold models, histories, prediction arrays and uncertainty methods are saved under `results/16/architecture_experiment/` and `results/16/lookback_experiment/`.

## 9. Interpretation, reproducibility and conclusion

The Optuna OOF prediction SD was lower than the actual-target SD; banded errors show overprediction at 0-10 µg/m³ and underprediction at 10-25, 25-50 and ≥50. For only seven >50 cases, CNN-LSTM MAE is 143.4 µg/m³ and persistence MAE is 127.2; these high-tail estimates are unstable but show poor extreme-event handling. The dataset/model evaluation covers 95 stations, but only six stations had adequate uninterrupted data and non-missing pairs for formal stationarity and lead-lag diagnostics. Those six cannot support basin-wide pollutant association claims. Lead-lag results are exploratory correlations, not validated predictive evidence. For completeness, zero-excluded pooled temporal and spatial diagnostic MAPE are 50.14% (n=2,867) and 42.13% (n=3,349), respectively; all cleaned targets are positive, but values near zero still make MAPE unstable, so RMSE and MAE remain primary. Temporal permutation importance ranks PM2.5 lag-1 first in all five folds (mean ΔRMSE 0.651 ± 0.443 µg/m³); rolling-3 is a weaker second-tier input (mean 0.153 ± 0.167). Importance is conditional and not causal. Spatial grouped results are separately reported (pooled RMSE 8.700, MAE 3.287, R² 0.154, n=3,349); this tests held-out station groups with their historical input sequences, not a completely unseen future/cold-start deployment.

The complete suite passed 55 tests in an isolated Python 3.12 environment (11 Keras deprecation warnings). The development checkpoint reloads and reproduces saved fold predictions to numerical precision. The historical configuration has been reconstructed to reproduce its saved outputs; original binary weights cannot be proven byte-identical. All five locked future artifacts retain their original SHA256 hashes, and no holdout labels were used for selection or recomputed scoring.

**Conclusion:** validation, baseline comparisons, architecture/lookback sensitivity, checkpoint reproducibility and row-level source matching are now auditable. Predictive gain over persistence remains nonsignificant, architecture results are mixed, and extreme events are poorly predicted. Compiled-data licensing, omitted-row inclusion criteria and exogenous forecast-time availability remain unresolved. These limitations constrain the defensible grade.

References: [1] Supplied “Updated version PM25 Forecasting Mini Project,” pp. 1-11. [2] Supplied “PM25 CNN-LSTM Layers and Libraries,” pp. 1-2. [3] Ameziane, “Air Quality - Mediterranean Basin (2016-2026),” Kaggle, v1, https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026. [4] OpenAQ, Terms of Use, https://docs.openaq.org/about/terms. [5] OpenAQ, Licenses resource, https://docs.openaq.org/resources/licenses. [6] Copernicus Climate Data Store, Terms of Use, https://cds.climate.copernicus.eu/licences/terms-of-use-cds. [7] NASA Science, Information Policy, https://science.nasa.gov/researchers/science-information-policy. [8] Copernicus Sentinel Data Legal Notice, https://cds.climate.copernicus.eu/licences/ec-sentinel. [9] H. S. Kim et al. (2022), Development of a CNN+LSTM Hybrid Neural Network for Daily PM2.5 Prediction, Atmosphere 13(12), 2124, https://doi.org/10.3390/atmos13122124. [10] X. Bai et al. (2024), Prediction of PM2.5 concentration based on a CNN-LSTM neural network algorithm, PeerJ 12:e17811, https://doi.org/10.7717/peerj.17811. [11] H. Hewamalage, K. Ackermann & C. Bergmeir (2023), Forecast evaluation for data scientists: common pitfalls and best practices, Data Mining and Knowledge Discovery, https://doi.org/10.1007/s10618-022-00894-5.

## Appendix A. Evidence and remaining work

Evidence: saved fold OOF arrays and metrics; nested Optuna, feature-family, architecture and lookback experiments; reloadable fold checkpoints; prediction-equivalent historical reconstruction; source row-level match tables; uncertainty methods; concentration-stratified errors; and full test output are recorded under `results/16/`, `data/provenance/`, and `reproducibility/`.

Remaining: obtain definitive compiled-data redistribution terms; clarify why eligible source rows were omitted; verify exogenous data delivery timing before expanding features. Original weight-file identity remains unproven. Frozen future artifacts were not changed.

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

Mean-fold RMSE difference (CNN-LSTM minus persistence): -0.251 µg/m³ (descriptive fold-bootstrap 95% interval -2.980 to 1.492); exact Wilcoxon p=0.625, Holm-adjusted p=1.000, five folds.

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
[8] Copernicus Sentinel Data Legal Notice, https://cds.climate.copernicus.eu/licences/ec-sentinel.
[9] Kim et al. (2022), Atmosphere 13(12), 2124, https://doi.org/10.3390/atmos13122124.
[10] Bai et al. (2024), PeerJ 12:e17811, https://doi.org/10.7717/peerj.17811.
[11] Hewamalage, Ackermann & Bergmeir (2023), Data Mining and Knowledge Discovery, https://doi.org/10.1007/s10618-022-00894-5.
