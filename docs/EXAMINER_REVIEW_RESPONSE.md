# Response to the pasted examination review

This response separates verified criticisms from claims that conflict with the supplied project artifacts or cannot be checked without touching the frozen future holdout. The review was supplied by the project owner; its claim that a grade was entered in a university system is not independently verifiable from this workspace.

## Verified gaps and work underway

| Review point | Verification against project/teacher materials | Response |
|---|---|---|
| Empty `comparisons.csv` | The file contains only its schema; teacher sections 7.3–7.4 require paired fold comparisons and Holm correction. | Added the missing comparison workflow; it will compare the CNN-LSTM with simple past-only baselines on identical development folds and report the five-fold power limitation. |
| Missing random-CV diagnostic | Teacher section 5.4 calls shuffled five-fold CV diagnostic-only; sections 6.4 and 8.4 request optimism gaps. Saved standard folds existed but had not been scored. | Scored five random folds using a fixed CNN-LSTM configuration and past-only baselines. Mean RMSE: random 8.091, temporal 8.301, spatial 7.939. Gaps are −0.209 random-minus-temporal and +0.152 random-minus-spatial. These gaps do not support the assumption that random CV must be optimistic in this run. |
| Insufficient search | The current search used three predeclared candidates. The teacher optimization table recommends seeded Optuna-TPE for its deep/hybrid family with a fixed budget, e.g. 50 trials. | A reproducible, nested Optuna-TPE search with 50 trials per outer temporal fold is running. Inner chronological validation chooses the trial; outer validation scores it once. |
| Missing local cases | Teacher section 6.7.3 requests representative pollution bands and the largest error. | Five representative cases were selected from saved temporal development OOF predictions, with input values and non-causal interpretation notes. Only the assigned CNN-LSTM is treated as a learned model; no other architecture was added to satisfy the “top three models” wording. |
| Missing station-month pollutant missingness | Teacher section 5.3.1 requests it for NO₂/CO/O₃. | Added 6,537 development-only station-month-variable rows in `results/16/pollutant_missingness_station_month_development.csv`. |
| ACF/PACF and stationarity coverage | Teacher sections 5.2 and 6.2 request temporal diagnostics. Irregular gaps make station-level tests on the entire series questionable. | ACF/PACF were computed on each station’s longest uninterrupted daily development segment of at least 14 days (63 stations). ADF/KPSS/Ljung–Box are available for only six stations with at least 30 consecutive days. This limited coverage will be reported; no gap-spanning series was treated as regular. |
| Ten-percent training-size condition | Teacher section 6.4.2 lists 10%, 25%, 50%, 75%, and 100%. Existing evidence omitted 10%. | Completed a development-only five-fold rerun. Mean fold RMSE was 10.151, 8.656, 8.480, 8.383, and 8.382 at 10%, 25%, 50%, 75%, and 100%. The 75%-to-100% change is under 5%, meeting the rubric’s descriptive plateau rule. |
| Exogenous predictor exclusion | The configured predictors are empty because source availability at forecast origin and the prepared table’s exact join are not auditable. | They remain excluded from the selected model. Adding them without proving source/date availability or join integrity could create leakage or false precision. |

## Claims not accepted as verified facts

- The review says the dataset has 51 monitoring sites. The audited cleaned file has 95 station IDs; use the audited manifest and EDA evidence.
- It identifies `report/TECHNICAL_REPORT_FINAL.pdf` as the 4-page report. That file was superseded. The current `reports/final_report.pdf` is 12 pages total: cover, 10 main-content pages, and appendix.
- It labels its grade as official and entered into the university system, but no official grading-system evidence is included in the pasted report. The points table sums to 15.0; its stated half-point deduction arithmetically yields 14.5, but that does not authenticate the grade.
- The pasted review contains detailed future-holdout subgroup metrics and MAPE. Those claims are not being independently recomputed or used for model selection because the future artifact is locked. Only the previously recorded aggregate historical metrics remain in project materials.
- Its comments about the student-group mapping do not override the owner’s explicit CNN-LSTM-only scope. No LSTM-only, GRU, or other competing learned architecture will be trained or added.

The diagnostic fold table is in `results/16/diagnostic_cv_results.csv`; scheme gaps are in `results/16/optimism_gaps.csv`; training-size results are in `results/16/training_size_sensitivity.csv`.

## Reproducibility and statistical limits

The TPE trials and supplemental diagnostics use only saved development indices ending before 2024-12-08. The paired tests have only five fold-level pairs and low power; corrected p-values will be descriptive evidence, not proof of superiority. Preprocessing ablations involving NO₂/CO/O₃ KNN imputation are not applicable while those variables are excluded. A log1p-target and missingness-indicator ablation will be run only on matching temporal development folds and scored after inverse transformation on the original PM2.5 scale.

No grade can be guaranteed by these changes. Dataset redistribution rights, upstream join provenance, incomplete contiguous-series diagnostics, and the unavailable historical future checkpoint remain material limitations.
