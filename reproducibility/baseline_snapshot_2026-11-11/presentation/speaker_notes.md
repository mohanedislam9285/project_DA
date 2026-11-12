# Speaker notes

1. Introduce Group 16 and state that development validation is now measured, with open data and rubric limitations.
2. Explain the public-health motivation and need for time/station validation.
3. Describe dataset size, dates, station locations and unresolved license terms.
4. Explain cleaning, missingness and skew; do not overinterpret correlation.
5. Define forecast origin, horizon, lookback and available inputs.
6. Explain the actual CNN-LSTM architecture and configured training settings.
7. Persistence uses PM2.5 at the forecast origin, not the next-day target. The five-fold CNN-LSTM/persistence difference is not statistically significant after correction.
8. Metrics are frozen historical values; the checkpoint was not recovered and the evaluation was not repeated.
9. Report the available-feature ablation and its five-fold confidence intervals; exogenous inputs were excluded.
10. Explain the tail-weighting trade-off and the fact that only seven validation predictions exceed 50 µg/m³.
11. Explain why AOD/gas/meteorology ablations cannot be claimed yet, and why licensing and join provenance remain open.
12. Close by distinguishing completed development evidence from unresolved provenance and legacy-evaluation limitations.
