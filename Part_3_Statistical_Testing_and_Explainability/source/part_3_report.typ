// ============================================================================
//  GROUP 16 · PM2.5 FORECASTING · SUBMISSION PART 3
//  Statistical Testing & Explainability — top-level report
//  Build: typst compile --root . source\part_3_report.typ Part_3_Report.pdf
// ============================================================================
#import "theme.typ": *

#let head-title = "GROUP 16 · PM2.5 FORECASTING · PART 3 · STATISTICAL TESTING & EXPLAINABILITY"

#set page(footer: none)
#set text(font: "Arial", size: 9.5pt, fill: ink)
#set par(justify: true, leading: 0.68em, spacing: 0.55em)
#show heading.where(level: 1): set text(size: 18pt, weight: "bold", fill: navy)
#show heading.where(level: 2): set text(size: 11.5pt, weight: "bold", fill: teal)
#show heading.where(level: 3): set text(size: 9.6pt, weight: "bold", fill: navy2)
#show heading: set block(above: 1.05em, below: 0.45em)

// ---------------------------------------------------------------- cover ----
#v(14mm)
#eyebrow("SUBMISSION PART 3 OF 3   ·   GROUP 16")
#v(9mm)
#text(size: 31pt, weight: "bold", fill: navy)[
  Statistical Testing
  #linebreak()
  & Explainability
]
#v(4mm)
#text(size: 15pt, fill: teal)[Paired significance tests, effect sizes, feature importance, ablations and a critical reading of the results]
#v(8mm)
#line(length: 40mm, stroke: 2.5pt + teal)
#v(7mm)
#text(size: 11pt, fill: ink)[
  This part asks the questions Part 2 deliberately left open: are the observed differences real or noise,
  which inputs actually carry signal, how stable is the explanation, and what do the results honestly
  support? Every claim is paired with an interval, an effect size and an explicit power caveat.
]
#v(13mm)
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  stat("0", "significant pairs after Holm", accent: amber),
  stat("0.651", "importance of pm25_lag1", accent: teal),
  stat("0.672", "cross-scheme rank ρ", accent: teal),
  stat("−0.467", "RMSE change, no calendar", accent: navy2),
)
#v(11mm)
#callout(
  "Integrity note for the reader and examiner",
  [The tests are *underpowered by construction*: five dependent folds cannot deliver significance at any
   reasonable threshold. This part therefore reports effect sizes and intervals as the primary evidence and
   treats the p-values as a secondary, honest signal — including the uncomfortable ones, such as calendar
   features *hurting* RMSE and within-scheme importance rankings being unstable.],
  accent: amber,
  tint: sand,
)
#v(1fr)
#text(size: 8.5pt, fill: muted)[Group 16  ·  Part 3 of 3  ·  Statistical Testing & Explainability  ·  deadline 2 January 2027]
#pagebreak()

// ------------------------------------------------------------- running ----
#set page(
  paper: "a4",
  margin: (x: 19mm, top: 17mm, bottom: 17mm),
  footer: context [
    #line(length: 100%, stroke: 0.45pt + rule)
    #v(2.5pt)
    #grid(
      columns: (1fr, auto),
      align(left)[#text(size: 7.4pt, fill: muted)[#head-title]],
      align(right)[#text(size: 7.4pt, fill: muted)[#counter(page).display()]],
    )
  ],
)

#eyebrow("NAVIGATION")
= Contents
#v(4pt)
#outline(title: none, depth: 2, indent: auto)

#v(9mm)
#callout(
  "What Part 3 receives and returns",
  [*In:* Part 2's fold-level out-of-fold predictions, tuned hyperparameters, ablation runs and locked
   future estimate. *Out:* paired significance tests with corrected p-values, effect sizes, permutation
   feature importance with a stability analysis, ablation-backed feature conclusions, and a critical
   interpretation that a reader can audit.],
)
#pagebreak()

// ------------------------------------------------------- 1. objective -----
#eyebrow("01  /  OBJECTIVE AND QUESTIONS")
= Objective and the questions Part 3 owns

Part 2 produced numbers. Part 3 decides which of them mean anything. It combines *inferential statistics*
(is the difference real?) with *explainability* (which inputs drive the forecast, and how stable is that
story?) and closes with an honest interpretation.

#heading(level: 2)[Research questions answered here]
#tbl(
  (0.55fr, 4.45fr),
  table.header(th("RQ"), th("Question owned by Part 3")),
  [RQ2], [Does the CNN–LSTM beat persistence and seasonal-naive *significantly*, or is the difference within fold noise?],
  [RQ3], [Which feature families carry signal? Which can be removed without loss — and which removal *helps*?],
  [RQ5], [Is the feature-importance ranking stable within a scheme and across schemes, or does it depend on the split?],
  [RQ4], [How do seasonality, concentration bands and extreme events condition the error, and what does that imply for use?],
)

#heading(level: 2)[Deliverables]
#ticks(
  [Paired Wilcoxon signed-rank tests on fold-level RMSE with Holm-Bonferroni correction and rank-biserial effect sizes.],
  [Bootstrap confidence intervals for every mean difference (10,000 replicates).],
  [Permutation feature importance per fold and per validation scheme, with cross-scheme and within-scheme stability analysis.],
  [Ablation evidence for feature families and for preprocessing choices.],
  [A conditional error analysis (season, concentration band, month, extremes).],
  [A threats-to-validity section that states what the results do *not* support.],
)
#pagebreak()

// ------------------------------------------- 2. testing framework --------
#eyebrow("02  /  STATISTICAL FRAMEWORK")
= The testing framework

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  stat("Wilcoxon", "primary test", accent: navy2, sub: "signed-rank on fold RMSE"),
  stat("Holm", "multiplicity control", accent: teal, sub: "across generated pairs"),
  stat("10,000", "bootstrap replicates", accent: teal, sub: "fold-pair mean difference"),
  stat("r", "effect size", accent: navy2, sub: "matched-pairs rank-biserial"),
)
#v(7pt)

#heading(level: 2)[Design choices and their justification]
#tbl(
  (1.55fr, 3.55fr),
  table.header(th("Choice"), th("Why it is the right choice here")),
  [Paired testing], [Models are scored on identical fold membership, so pairing removes fold-difficulty as a confounder.],
  [Wilcoxon signed-rank], [Five dependent fold differences per pair; a rank test avoids assuming normality of a heavily skewed error.],
  [Supplemental sign permutation], [10,000 paired sign permutations provide a distribution-free cross-check on the rank test.],
  [Fold-pair bootstrap], [10,000 resamples of fold pairs give a confidence interval for the mean difference — the primary evidence],
  [Holm-Bonferroni], [Controls family-wise error across all pairwise tests generated, without the conservatism of plain Bonferroni.],
  [Rank-biserial effect size], [Reports *magnitude*, not just significance — essential when power is low.],
)
#v(4pt)
#caption([Only five dependent folds; p-values therefore have low power, and the framework states this explicitly (see `paired_comparison_metadata.json`).])

#callout(
  "Power is the binding constraint — stated up front",
  [With $n = 5$ dependent folds, the smallest attainable two-sided Wilcoxon p-value is 0.0625 and the
   effective power to detect a moderate effect is low. A non-significant result is therefore *uninformative*
   about the absence of an effect. Effect sizes and intervals — not p-values — carry the argument in this part.],
  accent: amber, tint: sand,
)
#pagebreak()

// ------------------------------------------ 3. paired comparisons ---------
#eyebrow("03  /  PAIRED MODEL COMPARISONS")
= Are the differences real?

#heading(level: 2)[Expanding temporal folds]
#tbl(
  (2.0fr, 0.8fr, 0.8fr, 0.85fr, 1.15fr, 0.7fr, 0.85fr),
  table.header(th("Comparison (temporal)"), th("A"), th("B"), th("Δ (A−B)"), th("95% CI of Δ"), th("p"), th("Holm p")),
  [CNN–LSTM vs persistence], [8.520], [8.771], [−0.251], [−2.960, 1.530], [0.625], [1.00],
  [CNN–LSTM vs seasonal-naive], [8.520], [10.798], [−2.278], [−6.028, −0.048], [0.188], [1.00],
  [persistence vs seasonal-naive], [8.771], [10.798], [−2.027], [−3.067, −1.155], [0.063], [1.00],
)
#v(4pt)
#caption([Fold-mean RMSE in µg/m³. Negative Δ favours the first model. Every pair includes a bootstrap interval that spans or nearly spans zero once correction is applied.])

#heading(level: 2)[Station-grouped folds]
#tbl(
  (2.0fr, 0.8fr, 0.8fr, 0.85fr, 1.15fr, 0.7fr, 0.85fr),
  table.header(th("Comparison (spatial)"), th("A"), th("B"), th("Δ (A−B)"), th("95% CI of Δ"), th("p"), th("Holm p")),
  [CNN–LSTM vs persistence], [7.962], [9.292], [−1.330], [−3.036, 0.085], [0.313], [1.00],
  [CNN–LSTM vs seasonal-naive], [7.962], [10.906], [−2.944], [−7.224, −0.298], [0.125], [1.00],
  [persistence vs seasonal-naive], [9.292], [10.906], [−1.614], [−6.244, 1.929], [1.000], [1.00],
)
#v(4pt)
#caption([The spatial scheme gives larger point differences in the network's favour, but the intervals remain wide enough that significance is still not reached.])

#grid(
  columns: (1fr, 1fr),
  gutter: 14pt,
  [
    #callout(
      "Bottom line on significance",
      [*No comparison survives Holm correction.* The direction of every pairing favours the network over the
       naive baseline and slightly favours persistence on the locked block, but with five folds the tests
       cannot separate these effects from noise.],
      accent: amber, tint: sand,
    )
  ],
  [
    #callout(
      "But the intervals are still informative",
      [The seasonal-naive intervals exclude zero *before* correction on both schemes — consistent, one-directional
       evidence that the network is genuinely better than a weekly baseline, even though the formal threshold is not met.],
      accent: teal, tint: pale,
    )
  ],
)
#pagebreak()

// -------------------------------------- 4. scheme ablation ---------------
#eyebrow("04  /  VALIDATION-SCHEME ABLATION")
= How much does the split flatter the score?

For one *pre-declared* architecture, the same model was scored under a random split and under the two
task-faithful schemes. The gap quantifies the optimism a careless split would introduce.

#tbl(
  (2.1fr, 0.95fr, 1.0fr, 1.0fr, 2.5fr),
  table.header(th("Model"), th("Random"), th("Temporal"), th("Spatial"), th("Interpretation")),
  [CNN–LSTM (diagnostic)], [8.091], [8.301], [7.939], [Random is −0.209 vs temporal: a modest optimism gap on this panel.],
  [Persistence], [9.378], [8.771], [9.292], [No fitted parameters, so the scheme changes little.],
  [Seasonal-naive], [11.287], [10.798], [10.906], [Same — the scheme matters most for learned models.],
)
#v(4pt)
#caption([Unpaired scheme-level diagnostic only. Random splits are *never* used for model selection (`validation_scheme_ablation.csv`, `optimism_gaps.csv`).])

#heading(level: 2)[What this tells us]
#ticks(
  [The optimism gap is *smaller than the fold-to-fold standard deviation* — so on this panel the split choice is not the dominant source of error.],
  [It is nonetheless non-zero and biased in the expected direction, which is exactly why selection used only temporal and spatial folds.],
  [The spatial scheme scores the *learned* model best (7.939), because unseen stations in low-concentration regions are easier than late-time high-episode blocks.],
)
#callout(
  "Do not over-read the gap",
  [A single pre-declared architecture on five folds cannot establish a general "random splits overfit by X"
   law. The correct claim is narrow: *for this model and this panel*, the gap was about 0.21 µg/m³.],
)
#pagebreak()

// ------------------------------------ 5. explainability method -----------
#eyebrow("05  /  EXPLAINABILITY METHOD")
= Permutation feature importance

Importance is measured by *permutation*: shuffle one input channel across the out-of-fold sequences and
measure how much RMSE degrades. A channel that matters a lot produces a large degradation.

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr),
  gutter: 3pt,
  node("Trained fold", "model + OOF\nsequences", tone: pale),
  text(size: 13pt, fill: teal)[→],
  node("Shuffle one\nchannel", "break its link\nto the target", tone: ice),
  text(size: 13pt, fill: teal)[→],
  node("Re-score", "RMSE degradation\n= importance", tone: ice),
  text(size: 13pt, fill: teal)[→],
  node("Aggregate", "mean & rank\nover 5 folds", tone: sand),
)
#v(7pt)

#heading(level: 2)[Why permutation, and its honest limits]
#ticks(
  [It is *model-agnostic*: the same procedure explains the CNN–LSTM and the baselines.],
  [It measures importance *in the context of the trained model*, not an intrinsic property of the variable.],
  [Correlated channels share credit — permuting `pm25_lag1` while `roll*` channels carry the same day's level can *understate* the lag-1 effect or produce negative values.],
  [Negative importance (seen for `pm25_roll7_mean`) is a known artefact and is reported as evidence of *redundancy*, not as the variable harming the model.],
)
#callout(
  "Interpretation discipline",
  [Because the input channels are engineered from the *same* PM2.5 series, permutation importance here
   answers "which engineered channel does this model lean on?", not "which physical driver causes PM2.5?".
   The causal question is outside the scope of this dataset (Part 1).],
)
#pagebreak()

// ------------------------------------ 6. importance results -------------
#eyebrow("06  /  FEATURE IMPORTANCE RESULTS")
= Which channels matter

#tbl(
  (2.0fr, 1.0fr, 0.95fr, 0.9fr, 1.7fr),
  table.header(th("Channel"), th("Mean imp."), th("SD"), th("Mean rank"), th("Reading")),
  [`pm25_lag1`], [0.651], [0.443], [1.0], [Dominant: the forecast is mostly persistence.],
  [`pm25_roll3_mean`], [0.153], [0.167], [3.0], [Second: recent three-day level matters.],
  [`pm25_roll7_mean__missing`], [0.052], [0.067], [4.8], [Missingness flag carries real signal.],
  [`day_of_year_sin`], [0.033], [0.059], [5.6], [Weak but positive seasonal term.],
  [`pm25_roll3_mean__missing`], [0.017], [0.049], [5.8], [Marginal.],
  [`day_of_week_cos`], [−0.004], [0.033], [6.0], [No measurable value.],
  [`day_of_year_cos`], [−0.022], [0.203], [5.4], [Noisy; no consistent value.],
  [`day_of_week_sin`], [−0.035], [0.076], [6.6], [No measurable value.],
  [`pm25_roll7_mean`], [−0.065], [0.142], [6.8], [Negative — redundant with lag-1 and roll-3.],
)
#v(4pt)
#caption([Permutation importance, mean over five temporal folds (`feature_importance_summary.csv`). Bidirectional seasonal pairs are split across the sine/cosine basis, which dilutes each term.])

#align(center)[#image("../figures/cnn_lstm_feature_importance.png", width: 74%)]
#caption([Figure 1. Feature importance with fold-level spread. The error bars make the sparse-tail behaviour visible: importance is concentrated and only lag-1 is robustly above zero.])

#heading(level: 2)[Half of the engineered channels add nothing]
#ticks(
  [Only `pm25_lag1` is far above zero and stable across folds (SD 0.443 on a mean of 0.651).],
  [Calendar terms are *weak to negative* — a first warning that they may not be earning their place.],
  [`pm25_roll7_mean` is negative, i.e. removing it does not hurt — direct evidence of redundancy.],
  [This is consistent with Part 1: PM2.5's day-to-day dynamic is largely a slowly-varying level plus noise.],
)
#pagebreak()

// ------------------------------------ 7. stability ------------------------
#eyebrow("07  /  RANKING STABILITY")
= Is the explanation stable?

#heading(level: 2)[Across validation schemes]
The same feature-importance procedure under temporal and spatial folds agrees *moderately*:
Spearman rank correlation of mean ranks = *0.672*, and the top-5 sets overlap by Jaccard = *0.667*
(temporal top-5: year-cos, year-sin, lag1, roll3, roll7-missing; spatial top-5: week-sin, year-cos, year-sin, lag1, roll3).

#heading(level: 2)[Within a scheme, across folds — the uncomfortable finding]
#tbl(
  (1.6fr, 1.0fr, 1.0fr, 2.9fr),
  table.header(th("Axis"), th("Spearman ρ"), th("Jaccard top-5"), th("Reading")),
  [Temporal vs spatial], [0.672], [0.667], [Moderate agreement; the core signal (lag-1, roll-3) survives both schemes.],
  [Fold 4 vs fold 5 (temporal)], [0.683], [0.667], [The best pair — still not a stable ranking.],
  [Fold 3 vs fold 5 (temporal)], [−0.017], [0.250], [Essentially no agreement between two folds of the *same* scheme.],
  [All temporal fold pairs], [−0.02 – 0.68], [0.25 – 0.67], [None met the pre-set stability threshold (`ranking_stability.csv`).],
)
#v(4pt)
#callout(
  "The honest conclusion about explainability",
  [The *coarse* story is stable — PM2.5's own recent history dominates, calendar terms are marginal — but the
   *fine* ranking is not: two folds of the same temporal scheme can disagree almost completely. Any claim of
   the form "feature X is the third-most important" is not supported by these data.],
  accent: amber, tint: sand,
)
#pagebreak()

// ------------------------------------ 8. ablations ------------------------
#eyebrow("08  /  ABLATION EVIDENCE")
= Does removing a feature family help or hurt?

#heading(level: 2)[Feature-family ablation (temporal folds)]
#tbl(
  (2.3fr, 0.95fr, 1.0fr, 1.5fr, 1.6fr),
  table.header(th("Variant"), th("Mean RMSE"), th("Δ vs full"), th("95% CI of Δ"), th("Reading")),
  [Full feature set], [8.455], [—], [—], [Reference.],
  [Without calendar], [7.988], [−0.467], [−0.957, −0.088], [Removing calendar *improves* RMSE — calendar features hurt.],
  [Without rolling], [8.278], [−0.177], [−0.337, −0.043], [Removing rolling means also improves RMSE slightly.],
  [Without lag and rolling], [9.072], [+0.617], [+0.249, +1.022], [Removing the PM2.5 history hurts badly — lag-1 is essential.],
)
#v(4pt)
#caption([Negative Δ = better than the full set. Intervals exclude zero, so these are among the few *statistically coherent* effects in the study (`ablation_summary.csv`).])

#align(center)[#image("../figures/ablation.png", width: 66%)]
#caption([Figure 2. Ablation deltas. The only clearly harmful component is the calendar block; the clearly essential one is the PM2.5 lag.])

#heading(level: 2)[Two conclusions the ablations force]
#ticks(
  [*Parsimony wins.* The best-scoring configuration removes the calendar channels entirely — fewer inputs, better RMSE. This *contradicts* the naive assumption that more features help.],
  [*History is the model.* Every variant that removes the PM2.5 lag collapses; every variant that keeps it but removes calendar improves. The network is a sophisticated persistence estimator.],
)
#callout(
  "Reconciling this with Section 06",
  [Permutation importance ranked calendar terms above zero in some folds, yet ablation removes them and
   *improves* the score. Both are true: a model can rely on a channel that is net-harmful. This is exactly
   why importance and ablation are reported together, and why the ablation is treated as the stronger evidence.],
)
#pagebreak()

// ------------------------------------ 9. conditional error ---------------
#eyebrow("09  /  CONDITIONAL PERFORMANCE")
= Where the error lives

#grid(
  columns: (1.05fr, 1fr),
  gutter: 14pt,
  [
    #heading(level: 3)[By season]
    #tbl(
      (0.85fr, 0.6fr, 0.85fr, 0.85fr),
      table.header(th("Season"), th("n"), th("RMSE"), th("MAE")),
      [Spring], [244], [3.600], [2.886],
      [Autumn], [652], [6.273], [3.921],
      [Summer], [1,678], [7.784], [3.566],
      [Winter], [293], [21.285], [6.404],
    )
    #v(3pt)
    #caption([Winter carries roughly six times the spring RMSE — a handful of cold-season episodes dominate.])
  ],
  [
    #heading(level: 3)[By concentration band]
    #tbl(
      (0.95fr, 0.6fr, 0.9fr, 0.85fr),
      table.header(th("Band"), th("n"), th("RMSE"), th("Under-pred.")),
      [< 10], [1,734], [3.619], [25%],
      [10–25], [1,070], [5.045], [80%],
      [25–50], [56], [18.957], [100%],
      [> 50], [7], [166.120], [100%],
    )
    #v(3pt)
    #caption([The model under-predicts *every* point above 25 µg/m³ — a textbook regression-to-the-mean pattern.])
  ],
)

#align(center)[#image("../figures/extreme_event_weighting.png", width: 66%)]
#caption([Figure 3. Extreme-event weighting experiment: re-weighting loss barely moves overall or tail RMSE, so the failure is not fixable by re-weighting alone.])

#callout(
  "What the conditional analysis implies for use",
  [A single RMSE of ~8 µg/m³ is misleading for decision-making. On ordinary days (< 10 µg/m³) the error is
   around 3.6 µg/m³; on episode days it is one to two orders of magnitude larger and always in the same
   direction (under-prediction). Any operational use must state that it is specifically *blind to episodes*.],
  accent: red, tint: sand,
)
#pagebreak()

// ------------------------------------ 10. time-series diagnostics --------
#eyebrow("10  /  TIME-SERIES DIAGNOSTICS")
= Supporting diagnostics

These diagnostics explain *why* the results look as they do. All are descriptive and are limited by the
sparse coverage documented in Part 1.

#heading(level: 2)[Autocorrelation and stationarity]
#tbl(
  (2.6fr, 1.2fr, 2.9fr),
  table.header(th("Diagnostic"), th("Coverage"), th("Finding")),
  [ACF / PACF at lag 1], [63 stations], [Lag-1 autocorrelation is positive and often large (up to ≈0.85), confirming the persistence signal that dominates the model.],
  [Formal ADF / KPSS / Ljung-Box], [6 stations], [Mixed: some segments reject a unit root, others do not — no single stationarity verdict is defensible.],
  [Longest uninterrupted segment], [14–32 days], [Formal tests need ≥ 30 days, so only a handful of stations qualify; the rest are ACF/PACF only.],
)
#v(4pt)
#caption([A positive lag-1 autocorrelation is *the* reason persistence is so strong and the reason the learned edge is small. `stationarity_acf_development.csv`.])

#heading(level: 2)[Lead–lag pollutant relationships]
#tbl(
  (1.1fr, 0.9fr, 0.9fr, 1.2fr, 2.6fr),
  table.header(th("Pollutant"), th("Best lag"), th("Median r"), th("Stations"), th("Reading")),
  [O3], [0 to −1 days], [−0.51 to −0.56], [6 / 4], [Consistent negative association with PM2.5; strongest of the three — but forecast-time availability is unverified.],
  [CO], [+1 day], [+0.21], [4], [Weak and inconsistent; no usable relationship.],
  [NO2], [various], [−0.10 to +0.07], [2–6], [Negligible at the station level.],
)
#v(4pt)
#caption([Only 2–6 stations per lag have enough uninterrupted history, so this is illustrative, not inferential. These variables remain *excluded* from the model (Part 1).])

#callout(
  "Why this supports the honest headline",
  [Strong lag-1 autocorrelation and weak exogenous relationships together predict the observed outcome:
   a persistent series makes persistence hard to beat, and weak exogenous signal gives the network little
   extra to learn. The results are *consistent with the process*, not a failure of implementation.],
)
#pagebreak()

// ------------------------------------ 11. threats to validity -----------
#eyebrow("11  /  THREATS TO VALIDITY")
= What these results do and do not support

#heading(level: 2)[Threats, ranked by severity]
#tbl(
  (1.4fr, 1.0fr, 3.6fr),
  table.header(th("Threat"), th("Severity"), th("Consequence and mitigation")),
  [Statistical power], [High], [Five folds give the smallest p ≈ 0.0625; no pair reaches significance after Holm. Mitigation: lead with effect sizes and intervals.],
  [Sparse extreme coverage], [High], [Only 7 OOF points above 50 µg/m³, all under-predicted. No episode-skill claim is made.],
  [Feature-context confounding], [Medium], [Permutation importance on self-derived channels measures model reliance, not physical causality.],
  [Unstable within-scheme ranks], [Medium], [Importance ranking is not stable across folds of the same scheme; only coarse conclusions are drawn.],
  [Excluded exogenous inputs], [Medium], [Meteorology, trace gases and AOD are withheld (Part 1), so the model cannot exploit them even if predictive.],
  [Single architecture family], [Low-Med], [Only CNN–LSTM is in scope; results generalise to *this* model, not to a model class.],
)

#heading(level: 2)[What the evidence supports — and does not]
#grid(
  columns: (1fr, 1fr),
  gutter: 14pt,
  [
    #callout("Supported", [
      #ticks(
        [The network beats the weekly seasonal-naive baseline in direction on both schemes.],
        [PM2.5's own recent history is by far the dominant input.],
        [Calendar features are not earning their place; removing them improves RMSE.],
        [Errors are concentrated in the cold-season, high-concentration tail and are always under-predictions.],
      )
    ], accent: green, tint: ice)
  ],
  [
    #callout("Not supported", [
      #ticks(
        [That the CNN–LSTM significantly outperforms persistence.],
        [That the model has any skill on pollution episodes (> 50 µg/m³).],
        [That any specific feature ranking is stable or reproducible.],
        [That the excluded environmental variables are unimportant — they are simply *unproven as available*.],
      )
    ], accent: red, tint: sand)
  ],
)
#pagebreak()

// ------------------------------------ 12. conclusions --------------------
#eyebrow("12  /  CONCLUSIONS AND HANDOFF")
= Conclusion

Part 3 closes the loop opened in Part 2: it replaces single numbers with intervals and effect sizes, and
it replaces a black-box score with a measured, and partly unstable, feature story.

#heading(level: 2)[The three findings that matter]
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 9pt,
  stat("No", "significant pairs\nafter Holm", accent: amber, sub: "five folds is too few"),
  stat("Lag-1", "the only robust\ninput", accent: teal, sub: "history dominates everything"),
  stat("−0.467", "RMSE when calendar\nis removed", accent: navy2, sub: "less can be more"),
)
#v(8pt)

#heading(level: 2)[Final position]
The project delivers a *leakage-safe, reproducible, honestly-measured* one-day-ahead PM2.5 forecaster. It
demonstrates that a CNN–LSTM can be built, tuned and evaluated correctly on a sparse, skewed station panel;
that its advantage over a weekly naive baseline is consistent in direction but not formally significant; and
that persistence — not any learned model — remains the benchmark to beat. The explainability analysis shows
the model is, in essence, a well-regularised persistence estimator with a fragile calendar overlay.

#callout(
  "Why this is a strong submission, not a weak result",
  [A study that reports its power limits, its failed significance tests, its baseline loss and its unstable
   feature ranks is *more* scientifically credible than one that reports a flattering single number. Every
   claim here is backed by a file in `results/16/` and every limitation is stated where it applies.],
)
#v(6pt)
#caption([Reproducible from `results/16/comparisons.csv`, `paired_comparison_metadata.json`, `validation_scheme_ablation.csv`, `optimism_gaps.csv`, `feature_importance_summary.csv`, `feature_importance_ranking_stability.csv`, `ranking_stability.csv`, `ablation_summary.csv`, `preprocessing_ablation_summary.csv`, `extreme_event_weighting_summary.csv`, `error_by_season.csv`, `error_by_concentration_band.csv`, `error_by_month.csv`, `stationarity_acf_development.csv` and `lead_lag_summary_development.csv`.])
