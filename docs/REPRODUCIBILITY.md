# Reproducibility Record

- OS: Windows 11 x64; Python 3.12.10.
- Clean environment: created outside the submission folder; installed pinned `requirements.txt`, then installed the project editable. Installation completed successfully.
- TensorFlow 2.19.1; CPU execution, no detected GPU.
- Complete test run in isolated venv: 55 passed, 0 failed; 11 Keras deprecation warnings. Tests use temporary fixtures/mocks and did not modify locked future artifacts.
- In the current implementation pass, the safe suite was run from workspace-local source using `PYTHONPATH=src`; the active environment was not changed with an editable install.
- The isolated environment was created at `C:\Users\dell\AppData\Local\Temp\pm25_group16_validation_20261111`, with project installed editable and required packages inherited from the pinned Python 3.12 base. Full suite command: `python -m pytest -q`.
- Dataset checksums: `data/manifest.yaml`. Compiled dataset redistribution terms remain unresolved.
- Configured project seed 42. Temporal outer folds used fold seeds 43-47; seed sensitivity used 42, 43, 44.
- CNN-LSTM temporal command: `python -m pm25_project.cli cnn-lstm --config configs/project.yaml`. Spatial: `cnn-lstm-spatial`. Both operate only on the development folds.
- Robustness: `cnn-lstm-seeds`, `cnn-sensitivity`, `cnn-feature-ablation`, `cnn-train-size`.
- Development checkpoint: `results/16/models/cnn_lstm_development.keras`; preprocessor and config are adjacent. Metadata are in `development_model_metadata.json`. It was loaded successfully; a 20-row in-sample prediction smoke test is not an evaluation.
- Training history: `results/16/histories/cnn_lstm_development_training.csv`; fold/candidate histories are retained for spatial inner validation.
- Do not invoke the removed `evaluate-future` command, `final_evaluation.py`, or `run-all`. Existing future artifacts are frozen.
- The checkpoint above is newly trained on development data and did not generate the historical future scores. The historical checkpoint is unavailable.
- Frozen future artifact hashes: all five final SHA256 values match `results/16/future_artifacts_lock_initial.csv`; comparison details are in `results/16/future_artifacts_lock_final.csv`.
- The development TPE study uses pinned Optuna 5.0.0, an explicit seeded `TPESampler`, SQLite study storage, 50 trials per outer temporal fold, and one worker for reproducibility. See the [official TPESampler reference](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html) and [study API](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html).
- Before current-turn experiments, `reproducibility/baseline_snapshot_2026-11-11/snapshot_manifest.json` recorded hashes for 329 files (28,340,959 bytes) and software versions. The workspace has no Git metadata, so the snapshot manifest is the baseline state record.
- Origin-time persistence is the observed PM2.5 at t for a target at t+1. The corrected baseline predictions and paired comparisons were refreshed from saved diagnostic CNN predictions without retraining the diagnostic network.
- `run_extreme_event_weighting.py` uses development folds only. It applies MSE sample weights of 1x for PM2.5 ≤25, 2x for >25 through 50, and 4x for >50, with the same fold-selected configurations and epochs as the saved Optuna reference.
- `generate_review_evidence.py` reports station-month missingness and time-series tests on each station's longest uninterrupted development segment. ACF/PACF are reported for segments with at least 14 daily rows; formal tests require at least 30.
- `run_lead_lag_diagnostics.py` calculates descriptive pollutant-to-PM2.5 correlations at offsets from -7 to +7 days, using the longest uninterrupted development segment per station and at least 30 valid pairs. Sparse pollutant availability limits the contributing station count; these correlations do not establish causation or forecast-time availability.
- Exogenous predictors remain disabled because source records do not establish that weather, trace gases, or AOD are available by forecast issue time. `feature_group_ablation_status.csv` records those axes as blocked, not as evaluated results.
- `audit_saved_temporal_predictions.py` verifies horizon/target alignment, one-day raw-fold embargoes, actual metrics from saved arrays, model reload predictions, and locked artifact hashes. The maximum model reload vs recorded OOF prediction difference is 3.6e-15.
- `run_nested_architecture_experiment.py` compares direct/residual outputs and pooling on identical outer samples; `run_nested_lookback_experiment.py` compares 3- and 7-day histories on the same 2,867 samples. Both use development temporal folds only and save fold models/configs/histories/predictions. `analyze_temporal_oof_experiments.py` computes cluster intervals and paired five-fold tests; the five-fold tests are low power and not independent evidence of broad temporal generalization.
- `reconstruct_historical_checkpoint.py` refits using the recorded development candidate and epoch, then compares prediction values by saved sample ID only. It matches 1,716/1,716 recorded future predictions (maximum absolute difference 7.1e-15). It never reads future labels or calculates future metrics. Exact original weight-file identity cannot be established.
- `audit_upstream_join.py` records reproducible row-level PM/GEE correspondences and unmatched source rows; it cannot reconstruct the original ETL code or undocumented inclusion rules.


## Current-turn feature-family record (2026-11-12)
- `run_nested_feature_family_experiment.py` and `analyze_nested_feature_family_experiment.py` compare four predeclared available-feature families on the same five temporal folds and 2,867 targets. Selection stays within inner chronological validation; `feature_family_experiment/experiment_metadata.json` records that future labels/metrics were not used. Station-cluster intervals condition on the observed folds; paired tests have only five dependent observations and results are exploratory.
- `finalize_development_feature_model.py` chooses the feature family from a final chronological development split and refits using all 3,349 development sequences. The selected family is lag-1 only; seed 42016; 27 epochs. It saves `models/cnn_lstm_development_feature_selected.keras`, matching preprocessor/config, `development_feature_selected_model_metadata.json`, training history, and a 20-row in-sample smoke file. The smoke output is not evaluation. This new artifact does not replace the historical frozen predictions or the earlier base development model.
- MAPE has been calculated directly from the finite-grid OOF arrays: 50.1396% temporal and 42.1346% spatial, excluding only y<=0. MAPE is supplementary because small positive targets destabilize percentage error. Pooled spatial RMSE/MAE/R² are 8.7005/3.2874/.1539.
- Final full suite rerun on 2026-11-12: 55 passed, 0 failed, 11 Keras deprecation warnings. Detailed pytest output is archived at `reproducibility/pytest_2026-11-12.txt`. The full evidence record is `reproducibility/final_validation_record_2026-11-12.json`.



## 2026-11-12 cleaning and added-predictor evidence
- `scripts/audit_cleaning_quality.py` filters the saved development indices before reading/analysing the development-only table. It emits a JSON summary, environmental-field missingness and station-specific Q3+3×IQR high flags. Flags are not deletions.
- `run_issue_time_lagged_exogenous_experiment.py`/`analyze_issue_time_lagged_exogenous_experiment.py` compare target history against one-day-lagged pollutants/AOD, meteorology and all exogenous channels on identical outer temporal folds. All source values are no later than t−1 for a forecast issued at t; delivery timing itself remains unverified.
- `run_nested_spatial_predictor_experiment.py`/`analyze_spatial_predictor_experiment.py` compare target history against same-origin other-station PM means within 25/50/100 km, excluding the target station. All use development-only temporal folds and the persistence origin-time assumption. Neither tested expansion improved the target-history-only pooled result. Outputs and fitted fold artifacts are under the corresponding `results/16/` experiment folders.


Results of the added tests: lagged exogenous variants had pooled RMSE 9.480–9.598 / MAE 3.836–3.960; same-origin neighbor variants had pooled RMSE 9.361–9.439 / MAE 3.663–3.705. Both compared unfavorably with the PM-history-only reference (9.322 / 3.462). The nested selected procedures were nonsignificant versus persistence with five-fold exact Wilcoxon p=.813. Do not promote these variants.
