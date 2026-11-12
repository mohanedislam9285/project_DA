# Speaker notes

1. Introduce Group 16 and state that development validation is measured, with open data and rubric limitations.
2. Explain the motivation and why time/station validation matters.
3. Describe dataset size, dates, station coverage and unresolved license terms.
4. Explain cleaning, missingness and skew. The station-relative IQR screen flagged observations for review but did not automatically delete them.
5. Define forecast origin, horizon, lookback and issue-time inputs.
6. Explain the saved Optuna CNN-LSTM architecture and training configuration.
7. Persistence uses PM2.5 at the forecast origin. Five-fold paired evidence does not show significant improvement.
8. Historical metrics remain frozen; a development-only reconstruction matches outputs but cannot prove original binary identity.
9. Available-feature ablation results use the same folds; unverified exogenous variables remain excluded.
10. Explain the small weighting trade-off and that only seven OOF targets exceeded 50 µg/m³.
11. Feature-family selection varied by fold. The fixed no-calendar lag-plus-roll3 family has the best observed mean-fold score, but corrected fold tests remain inconclusive and no variant is established as superior.
12. Temporal and spatial diagnostics are distinct. Spatial folds evaluate held-out station groups conditional on historical sequences. MAPE is supplementary and unstable near zero.
13. Six stations support the longer formal/lead-lag diagnostics; these associations are exploratory rather than validated predictive effects.
14. Close with the remaining license, inclusion-rule, issue-time availability and generalization limits.
