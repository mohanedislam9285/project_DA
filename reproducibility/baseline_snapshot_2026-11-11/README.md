# Group 16 PM2.5 Forecasting: CNN-LSTM

This project forecasts daily station-level PM2.5 one day ahead. PM2.5 is fine particulate matter; forecasting its concentration may support air-quality monitoring. Group 16's assigned learned architecture is CNN-LSTM only. The historical future-test result is frozen. New model experiments use development data only.

## Readiness

Current status: **Complete development evidence with disclosed material data limitations**. Corrected origin-time baselines, nested Optuna development evidence, a reloadable development checkpoint, available-feature ablation, weighted-tail diagnostics, station-month missingness, uninterrupted-segment time-series tests and descriptive lead-lag correlations are saved. Compiled-data rights and the exact upstream feature-joining procedure remain unresolved; exogenous input timing is unverified. Predictive skill is modest and does not establish a significant improvement over persistence. See `docs/FINAL_QUALITY_AUDIT.md` for the evidence status and remaining limitations.

The result CSVs and figures in `results/16/` separate development folds from frozen historical evidence. Five temporal and five spatial fold scores are saved. The legacy future score is not reproducible from the available checkpoint/configuration and must be described as frozen historical evidence.

## What the model does

At the end of day *t*, the CNN-LSTM receives seven days of station history and predicts PM2.5 for day *t+1*. Inputs include calendar cycles and past PM2.5 lag/rolling values. Environmental variables are present in the source table but excluded because their availability at forecast time is not documented. Sequences do not cross stations or date gaps.

The network is causal Conv1D, LayerNormalization, ReLU, Dropout, optional MaxPooling1D, LSTM, Dense(ReLU), and a scalar output. Pooling can be disabled to preserve all time steps; the established model uses pool size 2. It uses Adam and MSE. Full settings are in `configs/project.yaml` and the saved configuration beside each model.

## Dataset and licensing

The active source is `data/raw/extracted/combined_gee_pm25.csv`, from Kaggle's [Air Quality - Mediterranean Basin dataset](https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026). Cleaning yields 40,983 rows from 95 stations (2016-11-25 to 2026-04-06 UTC). Source file checksums and the license caveat are in `data/manifest.yaml` and `docs/DATA_PROVENANCE.md`.

Kaggle labels the compiled dataset license “Other (specified in description)”. Its description points to OpenAQ, Copernicus and NASA sources, but does not resolve redistribution rights for this compiled bundle. Raw and derived files are retained locally; the package does not claim permission for unrestricted redistribution. The source-provided combined CSV also lacks a transformation log proving its upstream join method.

## Project map

- `src/pm25_project/` - data, CNN-LSTM, validation and training code.
- `configs/` - project settings, feature families and predeclared search candidates.
- `data/` - local source/processed data and provenance manifest.
- `results/16/` - EDA, development fold results, predictions, models, figures and historical frozen files.
- `reports/final_report.pdf` and `reports/source/final_report_source.md` - report and editable source.
- `presentation/final_presentation.pptx` and `presentation/speaker_notes.md` - slides and notes.
- `docs/` - teacher matrix, audit, provenance, methodology and reproducibility notes.

## Windows PowerShell setup

Use Python 3.12 (64-bit). From the project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Safe validation and development commands

```powershell
python -m pm25_project.cli validate-config
python -m pytest -q --ignore=tests/test_final_evaluation.py
python -m pm25_project.cli audit
python -m pm25_project.cli eda
python -m pm25_project.cli folds
python -m pm25_project.cli cnn-lstm --config configs/project.yaml
python -m pm25_project.cli cnn-lstm-spatial --config configs/project.yaml
python -m pm25_project.cli cnn-lstm-seeds --config configs/project.yaml
python -m pm25_project.cli cnn-sensitivity --config configs/project.yaml
python -m pm25_project.cli cnn-feature-ablation --config configs/project.yaml
python -m pm25_project.cli cnn-train-size --config configs/project.yaml
python -m pm25_project.cli cnn-lstm-finalize-dev --config configs/project.yaml
python run_optuna_development.py
python run_diagnostic_cv.py
python run_preprocessing_ablation.py
python generate_review_evidence.py
python run_extreme_event_weighting.py
python run_lead_lag_diagnostics.py
python train_optuna_development_model.py
python finalize_review_comparisons.py
```

The test exclusion is deliberate: `tests/test_final_evaluation.py` contains a call to the one-time evaluator. The `folds` command reuses the existing locked future partition and does not rewrite `future_test.json`. The CNN-LSTM development commands use fold JSONs and development rows only. Do not invoke the removed `evaluate-future` command, `final_evaluation.py`, or `run-all`.

The standalone review scripts are also development-only. Optuna runs 50 seeded TPE trials inside each of the five temporal outer folds; its objective uses only an inner chronological split of that fold's training data. Random K-fold results are labelled diagnostic-only and are never used for model selection. The finalizer only merges development fold results and computes paired comparisons.

The already-generated checkpoint is `results/16/models/cnn_lstm_development.keras`; its preprocessor and config share that folder. The smoke-test predictions in `results/16/predictions/development_checkpoint_smoke_test.csv` are in-sample and are only a serialization/inference check, not an evaluation score.

## Results

- `temporal_cv_results.csv`: five expanding development folds, one-day embargo.
- `spatial_cv_results.csv`: five station-grouped development folds; these measure unseen station groups conditional on available station history, not future dates.
- `development_cv_summary.csv`: fold means, standard deviations, descriptive fold-bootstrap intervals, and pooled OOF scores. Pooled scores weight every prediction equally and need not equal the unweighted fold mean. Five dependent folds provide limited uncertainty information.
- `predictions/cnn_lstm_temporal_oof.csv` and `predictions/cnn_lstm_spatial_oof.csv`: development-only out-of-fold predictions.
- `ablation.csv`, `training_size_sensitivity.csv`, `tables/cnn_lstm_seed_sensitivity_summary.csv`, `feature_importance.csv`, and `error_by_*.csv`: diagnostics based on development folds.
- `optuna_trials.csv`, `temporal_cv_optuna_results.csv`, and `optuna_best_hyperparameters.csv`: nested 50-trial-per-fold TPE search and outer-fold development scores.
- `diagnostic_cv_results.csv`, `optimism_gaps.csv`, and `comparisons.csv`: fixed-config diagnostic/random CV, corrected persistence/seasonal-naive scores, and paired development comparisons.
- `ablation.csv` and `ablation_summary.csv`: available-feature ablation with fold uncertainty; `feature_group_ablation_status.csv` marks exogenous axes blocked until timing is verified.
- `extreme_event_weighting_folds.csv`, `extreme_event_weighting_summary.csv`, and `predictions/cnn_lstm_extreme_weighting_oof.csv`: 1x/2x/4x sample-weighted MSE experiment and concentration-stratified errors.
- `preprocessing_ablation.csv`, `pollutant_missingness_station_month_development.csv`, `stationarity_acf_development.csv`, and `local_development_cases.csv`: additional evidence derived from development data/OOF predictions.
- `lead_lag_correlations_development.csv` and `lead_lag_summary_development.csv`: descriptive station-level pollutant/PM2.5 correlations at offsets -7 to +7 days on uninterrupted development segments; sparse pollutant coverage means only stations with at least 30 valid pairs contribute. These do not establish causality or forecast-time availability.
- `future_test_results.csv` and `predictions/future_test.csv`: historical frozen evidence only. They must not be used for tuning or overwritten.

MAPE is not reported because PM2.5 observations close to zero make percentage errors unstable. RMSE, MAE and R² are reported on the original µg/m³ scale.

## Tests and environment

The safe suite passed 49 tests in the current Python 3.12 environment; the five tests in `tests/test_final_evaluation.py` were excluded. This implementation pass used workspace-local source via `PYTHONPATH` and did not install the package editable into the active environment. TensorFlow 2.19.1 ran on CPU; no GPU was detected. Details and limitations are in `docs/REPRODUCIBILITY.md`.

## Historical frozen metrics

Saved future evaluation (not rerun): RMSE 6.0272 µg/m³, MAE 4.0240 µg/m³, R² 0.2394, N=1,716. The stored predictions begin 2024-12-12 while the reserved block begins 2024-12-08; the exact omitted station-days cannot be reconstructed from metadata alone. The corresponding historical checkpoint is absent.

## References and reproducibility

See `docs/TEACHER_REQUIREMENTS_MATRIX.md`, `docs/PROJECT_AUDIT.md`, `docs/METHODOLOGY.md`, `docs/REPRODUCIBILITY.md`, and `docs/FINAL_QUALITY_AUDIT.md`. See `reports/final_report.pdf` and the PowerPoint for the summarized results and remaining limitations.
