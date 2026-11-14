# Speaker notes and oral-defense prompts

## Slide 1: Title

Open by defining the forecast origin: use observations through the end of day t to predict station PM2.5 at t+1. State that this is a regression study. Development CV and previously saved future-period metrics are distinct.

## Slide 2: Task and RQs

Explain why monitoring can benefit from forecasts while avoiding an unsupported claim that the model itself improves health. Repeat that the target is a real-valued concentration, not a category. Introduce the main research question and the four labelled questions.

## Slide 3: Data and identity

The active file is a publisher-prepared combined table. Explain the composite station ID and why the source’s numeric-ID count differs from the cleaned station count. Be explicit that compiled-data redistribution rights remain unresolved.

## Slide 4: Seasonality

Seasonal PM2.5 summaries show the highest mean and >25 and >50 shares in DJF. They are descriptive record-level statistics. The project’s 25 and 50 µg/m³ cutoffs are not described as normative thresholds.

## Slide 5: Workflow and leakage

Walk through the workflow from prepared CSV to fold scoring. Emphasize the end-of-day origin and t+1 label, past-only features, embargo, fold-local preprocessing and inner-only tuning. Exogenous fields are excluded because observation dates alone do not prove availability to a forecaster.

## Slide 6: Architecture

Explain causal convolution as local feature extraction over adjacent time positions. Explain the LSTM gates as a mechanism for retaining or updating sequential state. The scalar output is continuous regression. Justify the architecture as a hypothesis and note that evaluation does not prove it beats persistence.

## Slide 7: Training curve

Show both training and chronological inner-validation MSE. The best inner validation loss occurs at epoch 45, followed by modest improvement/flattening; the training loss continues lower, consistent with overfitting. This curve is a representative archived inner fit and must not be conflated with outer-fold metrics.

## Slide 8: Validation

Temporal folds test later periods with expanding training data. Spatial folds test held-out stations conditional on having historical sequences. These answer different questions and are not interchangeable. Hyperparameter search is nested in outer training data.

## Slide 9: Evaluation

The complete RMSE/MAE/R² table is a fixed predeclared CNN–LSTM compared with persistence and weekly seasonal-naive on matched folds. Separately, per-fold Optuna tuning gives mean-fold RMSE 8.520 and MAE 3.978. The paired test does not show a significant improvement over persistence.

## Slide 10: Frozen score

State clearly that this is a previously saved score, not a new experiment in this turn. No future labels were read and no new baseline score was computed. Prediction equivalence does not prove that the absent original checkpoint has identical bytes.

## Slide 11: Explainability

Permutation importance ranks lag-1 first in all five folds. Missingness indicators exist for derived rolling channels in the reference preprocessing. Pollutant-specific flags are not in the final selected model because the pollutants themselves are excluded pending issue-time availability evidence.

## Slide 12: Ablation

Removing target lag and rolling features worsens fold RMSE. Removing calendar or rolling channels produces small improvements in this fixed no-retuning experiment. Do not treat this as an optimized feature search. Exogenous ablation variants are blocked because availability is unverified.

## Slide 13: Error analysis

Use these examples as descriptive development OOF cases only. Positive actual-minus-predicted residual means the model underpredicts. There were only seven cases over 50 µg/m³, so tail error estimates are unstable, but missing severe events is an important practical limitation.

## Slide 14: Diagnostic limits

Explain that requiring uninterrupted daily segments avoids pretending missing days are consecutive. Most station series are too short for formal tests; only six meet the 30-day requirement and have adequate lead-lag pairs. Spatial validation allows history at each held-out station and therefore does not measure cold-start performance.

## Slide 15: Significance and limitations

Describe the paired test and cluster-bootstrap interval with the sample limitations. The confidence interval includes zero, and only five temporal folds are available. Also disclose the data-rights caveat and missing original weights.

## Slide 16: Research answers

Answer each question explicitly. For RQ1, say no significant improvement. For RQ2, separate temporal and spatial scopes. For RQ3, identify lag-1 and qualified ablation findings. For RQ4, connect seasonality, sparse diagnostics, extreme error and unresolved rights.

## Slide 17: Conclusion

Conclude that the architecture is a plausible research candidate but not yet preferable to persistence for deployment. Give a clear answer to whether it is worth using: not as a claimed winner on these data. Mention the minimum evidence needed to revisit the decision.
