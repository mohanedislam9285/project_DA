# Final Quality Audit

**Disposition: Reproducible development evidence with material provenance and performance limits.** The package contains development-only comparisons and diagnostics. It does not claim cleared redistribution rights, recovery of the upstream ETL, or byte-identical original historical weights.

| Area | Verification | Status |
|---|---|---|
| Teacher requirements | Mapped in `TEACHER_REQUIREMENTS_MATRIX.md` | Met where supported by supplied data; blocked exogenous ablations are explicitly marked |
| Scope | CNN-LSTM only trained/tuned/evaluated as learned model | Pass |
| Architecture | Development-only historical reconstruction matches all 1,716 saved predictions; original binary hash unavailable | Pass with identity limitation |
| Dataset | Size, dates, station count, explicit PM bounds, key/coordinate checks, station-relative anomaly screening and rights caveat documented | Strong quality controls; upstream rights remain partial |
| Join provenance | Every bundled record matches source PM/GEE rows; omitted-row selection and original ETL remain undocumented | Partial/open |
| EDA | Required CSVs, distribution, trend, station summary, correlation, missingness and map exist | ACF/PACF and ADF/KPSS/Ljung-Box on sufficiently long uninterrupted development segments; station-month missingness and descriptive pollutant/PM2.5 lead-lag correlations saved |
| Leakage prevention | Fold-local preprocessing; sequence and split tests pass | Pass for audited code |
| Temporal CV | Five scored folds; 2,867 predictions; metrics saved | Pass |
| Spatial CV | Five grouped station folds; 3,349 predictions; station disjointness verified | Pass, conditional on historical station input |
| Hyperparameter tuning | Nested 50-trial TPE per temporal outer fold; trial log, fold configs and selected model saved | Pass; selection is development-only |
| Robustness | Three seeds, training-size sensitivity, parameter sensitivity and nested feature-family comparison saved | Pass with exploratory limits |
| Ablation | Four comparable CNN-LSTM feature sets, five folds; additional lagged-environment and spatial-neighbor tests | Pass; extra inputs did not improve OOF results |
| Feature importance | Temporal and spatial permutation values and ranking comparison | Pass with interpretability caveat |
| Error analysis | Residuals by date, station, season, concentration band and month | Pass |
| Metrics | RMSE, MAE, R² and supplementary zero-excluded MAPE trace to saved OOF arrays; `results.csv` also carries per-model/scheme `niveau=global` summary rows (mean ± std and 95% bootstrap intervals) aggregated from the saved folds | Pass |
| Development checkpoint | Saved, loaded; 20-row in-sample smoke test | Pass |
| Report | Editable source and 15-page PDF (14 main pages plus appendix) | Includes related work, temporal/spatial evidence, feature-family experiment, MAPE caveat and limitations |
| Presentation | 14 slides plus speaker notes | Includes feature-family sensitivity, spatial results, metric caveat and scope limitations |
| Tests | Full isolated suite: 55 passed, 11 Keras deprecation warnings | Pass |
| Fresh environment | Pinned requirements installed and project installed editable in a clean venv | Pass |
| Licensing | Rights for compiled data unresolved | Open |
| Frozen future artifacts | Initial/final SHA256 values compared; all five match | Pass |
| Forbidden future evaluation | No future labels or metrics used for selection; reconstruction compared predictions only | Pass |

## Remaining material limitations
1. The compiled Kaggle dataset's redistribution terms remain unresolved. The source is retained locally and cited; no unrestricted redistribution claim is made.
2. The publisher describes date/nearest-coordinate matching (~0.25°). All 41,718 bundled rows map to exact PM/GEE source records with coincident coordinates, but 12,074 additional source PM rows meet the approximate distance condition and are absent. Full inclusion logic and original ETL remain unrecovered.
3. Time-series diagnostics have limited station coverage: only six stations have a 30-day uninterrupted segment with enough non-missing pollutant pairs for lead-lag correlations. These correlations are descriptive and do not establish causation or forecast-time availability.
4. The original historical checkpoint binary and hash cannot be recovered. A development-only refit using the recorded candidate and selected epoch reproduces every saved prediction to 7.1e-15, but exact weight-file identity is unproven. Frozen metrics were not recalculated.
5. Predictive skill is modest and variable. Nested feature-family experiments show a modest temporal fold-average reduction against the reference features, but fold-level and baseline comparisons remain inconclusive after correction; no significant improvement over persistence is established.
6. Exogenous feature ablations remain blocked because forecast-origin availability is unverified.

All five final future-artifact hashes match the initial lock. The baseline manifest captured 329 files / 28,340,959 bytes before experiments. Architecture and lookback comparisons use development folds only and do not establish robust superiority over persistence.

No future labels were read or metrics recomputed. The historical model reconstruction only compared prediction arrays by sample ID.


## 2026-11-12 evidence addendum
A four-family nested feature experiment now uses the same five temporal outer folds and 2,867 targets. The best fixed family (without calendar terms, lag-1 plus rolling-3) has mean-fold RMSE 7.894 and MAE 3.396, versus 8.520/3.978 for the reference. Its exact five-fold RMSE p=.063 and Holm-adjusted p=.313 across the five family procedures; comparison with persistence is nonsignificant (exact p=.813; Holm p=1.00). Treat this as exploratory, not a confirmed gain. A separate final development fit selected lag-only on an inner split and refit all 3,349 development sequences. It has no independent score.

The report and slides now include spatial pooled metrics and supplementary zero-excluded MAPE calculated from the actual finite-grid OOF arrays. MAPE is clearly caveated for near-zero target values. The report contains 14 main pages plus one appendix (15 total); the deck has 14 slides. Related work was added to frame differences in inputs and validation design without claiming cross-study comparability.


The added cleaning QA is diagnostic and non-destructive: its Tukey-based station-relative flags are retained and documented rather than treated as false positives for deletion. The nested lagged-environment and spatial-neighbor experiments did not improve target-history-only forecasting, so no expanded feature set is promoted. The prediction-quality limitation remains substantive rather than hidden by scoring choices.


## Predictor expansion and cleaning audit addendum
The latest development-only cleaning audit covers 30,651 rows / 73 stations before the locked future period: no out-of-rule targets, duplicate station-day keys or invalid coordinates. A Q3+3×IQR station-relative screen flagged 352 values among 71 adequately sized station series; the flagged values remain retained for review.

One-day-lagged meteorological/pollutant/AOD variants had pooled RMSE 9.480–9.598 and MAE 3.836–3.960, above the PM-history-only reference at 9.322/3.462. Same-origin neighboring-station features at 25/50/100 km yielded RMSE 9.361/9.439/9.408, also above PM-history-only. The inner-selected variants remained nonsignificant against persistence (five-fold exact p=.813). Neither added feature group is used in the final model; the exogenous source delivery times are still unknown.


## 2026-11-12 review-hardening addendum
A post-hoc, non-destructive audit reconciled `results/16/results.csv` with the reporting contract. No model was refit and the single-use future holdout was **not** re-evaluated; the five frozen future artifacts remain byte-identical to their initial lock. Results were produced by `finalize_review_hardening.py` and recorded with pre/post SHA-256 in `results/16/review_hardening_record.json`.

- **Fold aggregation.** `results.csv` previously contained only `niveau=fold` rows, so the requested mean ± standard deviation and 95% bootstrap intervals were empty. It now carries 39 `niveau=global` validation summary rows (one per model/scheme/metric group) aggregated from the 183 saved fold rows: std is populated for 36 groups (three single-fold groups correctly have no dispersion) and 95% bootstrap intervals for all 39. The underlying folds are unchanged; summaries were computed from them, not re-measured.
- **Future-test baselines.** The frozen holdout previously reported the CNN-LSTM score only, with no baseline comparison and no `dataset_split=future_test` rows in `results.csv`. Nine such rows are now present: the three CNN-LSTM metrics mirrored from the frozen artifact, plus `persistence` and `seasonal_naive` scored deterministically on the **identical 1,716 frozen sample IDs** (baselines depend only on observed PM2.5, so this is not a model refit). Persistence RMSE is 5.846071 and seasonal naïve RMSE is 8.026846; these are consolidated into `results.csv` (not hash-locked) and baseline predictions into a separate `future_test_baselines.csv`, so the locked `future_test_results.csv` stays untouched.
- **Honest limitation restated.** On the frozen holdout the CNN-LSTM (RMSE 6.027) is **worse** than one-day persistence (5.846) and better than seasonal naïve (8.027). The hardening adds the missing reporting fields and a like-for-like baseline; it does not establish skill over persistence, and this negative result is reported as-is rather than hidden by scoring choices.
- **Provenance note.** `future_test_results.csv` reports a different raw SHA-256 on a Windows checkout because `core.autocrlf=true` rewrites three `LF` to `CRLF`; after newline normalisation the content is byte-identical to the 2026-11-11 lock (1206 bytes, `ce732a39…`), and git treats the file as unmodified.
