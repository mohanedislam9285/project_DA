# Project Audit

## Inventory and protection
- Original repository and external backup paths were confirmed to exist before changes. The original repository was not modified. The initial workspace inventory is `inventory_initial.csv`.
- Frozen historical files under `results/16/` were hashed before work; the list is `results/16/future_artifacts_lock_initial.csv`. They were not used for model fitting, rewritten, or used to select development settings.
- Superseded report/PDF drafts, unused legacy assets, and review-only renders were moved to `garbeg/` for manual deletion. Current reviewed artifacts are in `reports/` and `presentation/`.

## Dataset and pipeline
- Raw combined CSV contains 41,718 rows and 15 source columns. Cleaned daily table has 40,983 rows, 19 columns, 95 station IDs, and dates 2016-11-25 through 2026-04-06 UTC.
- Cleaning removed 735 rows: 727 PM2.5 values <= 0 and 8 values >= 500; no missing timestamp/target rows or duplicate station-days remained.
- Station identity combines sensor name and coordinates rounded to six decimals. Four same-name/date observations at differing coordinates remain separate.
- The publisher describes a date/nearest-coordinate join (about 0.25°). Row-level audit matches all 41,718 bundle rows to exact source PM/GEE records with 0 km coordinate distance; however, 12,074 additional PM source rows meet the approximate distance condition but are omitted. Results reproduce the bundled correspondences, not the actual ETL or full inclusion rule.
- NO2, CO, O3 missingness is about 5.9-6.0%; other cleaned predictors are complete.
- Exogenous satellite/reanalysis predictors are marked unavailable at forecast origin and configured predictor list is empty. Active feature path uses calendar features plus target lag/rolling history.

## CNN-LSTM
Source: causal Conv1D -> LayerNormalization -> ReLU -> Dropout -> MaxPooling1D -> LSTM -> Dense(ReLU) -> Dense(1). Config: filters=32, kernel=3, dropout=.2, LSTM=64, dense=32, Adam learning rate=.001, MSE, batch=64, max 100 epochs, patience=10. Historical metadata says lookback=7, 22 epochs, seed=42; no historical checkpoint/config was found under `results/16/models/`. Do not attribute historical scores to any new model.

## Holdout date gap
The reserved block begins 2024-12-08; saved scored predictions begin 2024-12-12 (1,716 predictions). Sequence generation requires a station-contained contiguous lookback and one-day target; early station-days without admissible windows are not scored. Artifacts do not establish exactly which dates/stations cause the initial gap, so no more precise cause is claimed.

## Validation and results
Five scored temporal folds (2,867 OOF predictions) and five station-grouped folds (3,349 OOF predictions) are saved. The current Optuna temporal folds have mean RMSE=8.520 (SD=6.034) and mean MAE=3.978. Persistence on identical targets gives mean RMSE=8.771 and MAE=3.158; the difference is nonsignificant (exact Wilcoxon p=.625). Architecture and 3-vs-7-day sensitivity use matched target samples; no robust winner was found. Fold-level actual prediction metrics, architecture scores and confidence procedures are saved. The historical checkpoint reconstruction reproduces all saved predictions without using holdout labels; original binary weights remain unverified.

The reserved period still starts 2024-12-08 and saved predictions begin 2024-12-12. Sequence-building requirements filter windows without complete station-specific contiguous input history. The exact omitted station-days are not established from recorded metadata; do not claim a more precise cause.

## Reproducibility and tests
A clean isolated Python 3.12.10 virtual environment installed the project editable. TensorFlow 2.19.1 ran on CPU; no GPU was detected. The full test suite passed: 55 passed, 0 failed, with 11 Keras deprecation warnings.
