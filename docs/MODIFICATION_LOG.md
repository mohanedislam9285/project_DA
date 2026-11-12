# Modification Log

Date: 2026-11-11. Scope: finalize and audit the Group 16 CNN-LSTM submission workspace. The original repository and external backup were not modified.

- Restricted the project CLI/workflow to Group 16 CNN-LSTM development work; no future evaluator, frozen holdout score calculation, comparator model, or `run-all` workflow was executed.
- Protected `results/16/future_test.json` from fold regeneration and added a regression test for byte preservation.
- Repaired the development results finalizer to preserve legacy rows, write temporal and spatial development fold rows idempotently, and report both unweighted fold means and pooled OOF metrics.
- Saved development-only temporal and grouped spatial CV, tuning, robustness, ablation, importance, error-analysis, timing, history, and checkpoint evidence under `results/16/`.
- Created a clean Python 3.12.10 environment, installed pinned requirements and the package, and reran the safe test suite: 45 passed. The five-test module that invokes the one-time evaluator was excluded.
- Updated the editable report source and rebuilt `reports/final_report.pdf` (12 pages including cover and appendix); rendered all pages to PNG and visually inspected the contact sheet.
- Rebuilt `presentation/final_presentation.pptx` (11 slides, each with notes) and updated speaker notes.
- Initially archived obsolete report/PDF drafts under `report/archived_drafts/`; on the later cleanup request, moved the obsolete presentation too, along with all legacy report material, into `garbeg/` for manual deletion.
- Added an official-source terms review to `docs/DATA_PROVENANCE.md`; compiled-data redistribution terms remain unresolved.
- Compared all five final frozen-artifact hashes with the initial lock; all match. No frozen future artifacts were recalculated or modified.
- Updated `README.md`, teacher requirements, methodology, reproducibility record, quality audit, project audit, data provenance, and inventory files to reflect observed evidence and remaining issues.
- On the user's follow-up, moved superseded report drafts, legacy assets and originals, review-only PDF render images/contact sheet, and Python test caches into `garbeg/` for manual deletion. The active report and presentation were left in their current folders.

## Remaining material issues

At the time of this earlier log entry, the package was marked NOT READY FOR FINAL SUBMISSION. It cited unresolved compiled-data rights and upstream join provenance, diagnostics that were not yet present, and the unavailable historical future checkpoint/configuration.

## Evidence reconciliation (2026-11-11)

The prior readiness note above is retained as a historical record. Later review evidence in `results/16/` includes station-month pollutant missingness, ACF/PACF plus ADF/KPSS/Ljung-Box diagnostics on sufficiently long uninterrupted development segments, and descriptive lead-lag correlations in `lead_lag_correlations_development.csv` and `lead_lag_summary_development.csv`. Sparse pollutant availability limits useful lead-lag evidence to six stations. The README, audit, report, and presentation reflect these artifacts. The compiled-bundle license, upstream join method, forecast-time availability of exogenous fields, and unrecoverable legacy checkpoint remain material limitations. The evidence does not show a significant temporal-CV improvement over persistence.
