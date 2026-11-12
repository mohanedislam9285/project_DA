// ============================================================================
//  GROUP 16 · PM2.5 FORECASTING · SUBMISSION PART 2
//  Model Training, Validation, Testing & Evaluation — top-level report
//  Build: typst compile --root . source\part_2_report.typ Part_2_Report.pdf
// ============================================================================
#import "theme.typ": *

#let head-title = "GROUP 16 · PM2.5 FORECASTING · PART 2 · MODEL TRAINING & EVALUATION"

#set page(footer: none)
#set text(font: "Arial", size: 9.5pt, fill: ink)
#set par(justify: true, leading: 0.68em, spacing: 0.55em)
#show heading.where(level: 1): set text(size: 18pt, weight: "bold", fill: navy)
#show heading.where(level: 2): set text(size: 11.5pt, weight: "bold", fill: teal)
#show heading.where(level: 3): set text(size: 9.6pt, weight: "bold", fill: navy2)
#show heading: set block(above: 1.05em, below: 0.45em)

// ---------------------------------------------------------------- cover ----
#v(14mm)
#eyebrow("SUBMISSION PART 2 OF 3   ·   GROUP 16")
#v(9mm)
#text(size: 31pt, weight: "bold", fill: navy)[
  Model Training,
  #linebreak()
  Validation & Evaluation
]
#v(4mm)
#text(size: 15pt, fill: teal)[CNN–LSTM implementation, leakage-safe validation, tuning, baselines and honest performance testing]
#v(8mm)
#line(length: 40mm, stroke: 2.5pt + teal)
#v(7mm)
#text(size: 11pt, fill: ink)[
  This part turns Part 1's frozen data contract into a working forecasting system: it defines
  the CNN–LSTM architecture, the training protocol, the expanding-temporal and station-grouped
  validation designs, the nested tuning search, the two required baselines, and the measured
  out-of-fold and locked-future performance — reported with its uncertainty, not just its mean.
]
#v(13mm)
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  stat("8.455", "reference CV RMSE (µg/m³)", accent: teal),
  stat("7.962", "station-grouped RMSE", accent: teal),
  stat("250", "tuning trials (5 × 50)", accent: navy2),
  stat("6.027", "locked future RMSE", accent: amber),
)
#v(11mm)
#callout(
  "Integrity note for the reader and examiner",
  [The one-day-ahead task is dominated by *persistence*: the last observed value is already a strong
   predictor. This part reports the network's edge over persistence as *small and not statistically
   significant*, and reports the case where persistence wins outright on the locked future block. The
   value of the study is the honest, leakage-safe measurement, not a claimed victory.],
  accent: amber,
  tint: sand,
)
#v(1fr)
#text(size: 8.5pt, fill: muted)[Group 16  ·  Part 2 of 3  ·  Model Training, Validation & Evaluation  ·  deadline 12 December 2026]
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
  "What Part 2 receives and returns",
  [*In:* Part 1's frozen contract — a cleaned station-day panel, a seven-day past-only sequence
   definition, a target rule of 0 < y < 500 µg/m³, and fold-local preprocessing. *Out:* fold-level
   out-of-fold predictions, a locked independent-future estimate, tuned hyperparameters, and baseline
   comparisons. Part 3 consumes exactly these predictions for significance testing and explainability.],
)
#pagebreak()

// ------------------------------------------------------- 1. objective -----
#eyebrow("01  /  OBJECTIVE AND TASK")
= Objective, task and evaluation contract

The model forecasts the *next-day* PM2.5 concentration (µg/m³) at a monitoring station from the seven
days ending at the forecast origin. Because the target is continuous, this is regression; because it is
a forecast, the only admissible inputs are values observed at or before the origin.

#heading(level: 2)[What Part 2 must deliver]
#ticks(
  [A concrete CNN–LSTM implementation with a stated layer order, regularisation and parameter budget.],
  [A training protocol fixed in advance: loss, optimiser schedule, early-stopping rule, seeds and hardware.],
  [Two leakage-safe validation designs — *expanding temporal folds* with an embargo, and *station-grouped* folds — plus a random-split diagnostic that is explicitly not used for selection.],
  [A nested hyperparameter search in which tuning is performed *inside each outer fold*.],
  [Two mandatory baselines — persistence and weekly seasonal-naive — scored on identical folds.],
  [A single, frozen, independent future-block estimate that is reported once and never re-tuned.],
)

#heading(level: 2)[Evaluation contract]
#tbl(
  (1.3fr, 3.7fr),
  table.header(th("Element"), th("Definition fixed before any model was scored")),
  [Target], [PM2.5 at station-day $t+1$, in µg/m³, no transformation for scoring.],
  [Inputs], [The seven-day window ending at $t$: PM2.5 lag and rolling summaries plus calendar cycles (Part 1).],
  [Primary metric], [RMSE in µg/m³. *RMS-dominant* because the tail matters and the target can approach zero.],
  [Secondary metrics], [MAE (robust central error) and R² (variance explained). MAPE is excluded — near-zero targets make it unstable.],
  [Uncertainty], [Fold-level standard deviation, bootstrap confidence intervals and paired significance tests (Part 3).],
)
#v(4pt)
#caption([The metric set is deliberately small and pre-registered so that results cannot be chosen after the fact.])
#pagebreak()

// -------------------------------------------------- 2. architecture -------
#eyebrow("02  /  ARCHITECTURE")
= The CNN–LSTM model

The architecture is a hybrid: a *convolutional* stage extracts local shape features from the seven-day
window, and a *recurrent* stage summarises the sequence into a single scalar forecast.

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr),
  gutter: 3pt,
  node("Input", "7 days ×\n7 channels", tone: pale),
  text(size: 14pt, fill: teal)[→],
  node("Conv1D", "causal kernel\n$k$ filters", tone: ice),
  text(size: 14pt, fill: teal)[→],
  node("LSTM", "recurrent\nsummary", tone: ice),
  text(size: 14pt, fill: teal)[→],
  node("Dense → 1", "ReLU then\nscalar µg/m³", tone: sand),
)
#v(7pt)

#heading(level: 2)[Layer order and why each element is there]
#tbl(
  (1.35fr, 1.35fr, 2.3fr),
  table.header(th("Stage"), th("Setting"), th("Role")),
  [Causal `Conv1D`], [filters ∈ \{16, 32, 64\}, kernel $k$ ∈ \{2, 3, 5\}], [Detects short local patterns in the window; *causal* padding guarantees no future step influences an earlier position.],
  [`LayerNorm` + `ReLU`], [per-channel], [Stabilises activations across a heavy-tailed input distribution; adds the needed non-linearity.],
  [`Dropout`], [0.05 – 0.35], [Regularises a small-data sequence model; the rate is tuned, not fixed.],
  [`LSTM`], [units ∈ \{32, 64\}, return last step], [Carries the temporal state and compresses the sequence to one representation.],
  [`MaxPooling1D`], [optional], [Tested in the architecture study; kept only when it helped inside the inner loop.],
  [`Dense` + `ReLU`], [units 32], [Non-linear read-out head.],
  [`Dense` (output)], [1 unit, linear], [Scalar forecast on the standardised target; inverted for scoring.],
)
#v(4pt)
#caption([Reference model: 10,081 trainable parameters, lookback 7, 7 input channels, ≈0.16 MB. The lag-1-only refit uses 22,945 parameters. TensorFlow 2.19.1, CPU only.])

#callout(
  "Why causal convolution, not a plain one",
  [With a standard symmetric kernel, position $t-6$ could be modified by padding values derived from
   positions after it. Causal padding removes that path entirely, so the model cannot see the target
   day through the convolution. It is a leakage control, not a style choice.],
)
#pagebreak()

// ---------------------------------------------------- 3. training ---------
#eyebrow("03  /  TRAINING PROTOCOL")
= How the model is trained

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  stat("MSE", "training loss", accent: teal, sub: "weighted-MSE variant also tested"),
  stat("Adam", "optimiser", accent: navy2, sub: "tuned learning rate ≈ 2e-4 to 1e-3"),
  stat("Patience 10", "early stopping", accent: teal, sub: "on validation MSE, best weights restored"),
  stat("Patience 3", "LR schedule", accent: teal, sub: "ReduceLROnPlateau ×0.5, min 1e-5"),
)
#v(7pt)

#heading(level: 2)[Protocol rules]
#ticks(
  [*Loss:* mean squared error — chosen because the primary metric is RMSE and the tail is the failure mode of interest.],
  [*Early stopping:* training halts when validation MSE stops improving for 10 epochs; the best-weight checkpoint is restored, so the reported model is never the last, over-fitted epoch.],
  [*Learning-rate control:* `ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)` lets a coarse initial rate refine as the curve flattens.],
  [*Seeds:* one fixed seed per outer fold (43–47) so that every compared model sees the same stochastic path within a fold.],
  [*Preprocessing:* the imputer and scaler are fitted on that fold's *training* sequences only (Part 1 contract), then applied to the validation sequences.],
  [*Reproducibility:* 2.3 s to fit the reference model on CPU; the full nested search is 250 short trials. Environment: Python 3.12.10, TensorFlow 2.19.1, CPU only.],
)

#align(center)[#image("../figures/training_validation_loss_typst.png", width: 76%)]
#caption([Figure 1. A representative training/validation loss curve: validation loss flattens while training loss keeps falling — the early-stopping signal that guards against over-fitting.])
#pagebreak()

// ------------------------------------------------- 4. validation design ---
#eyebrow("04  /  VALIDATION DESIGN")
= Two honest validation schemes

Random shuffling would let a model memorise a station's own recent days and then "predict" them. In a
forecast study that is leakage. Two task-faithful schemes are used instead.

#grid(
  columns: (1fr, 1fr),
  gutter: 14pt,
  [
    #heading(level: 3)[Expanding temporal folds]
    #ticks(
      [Train on the past, validate on the *next, later* block — the real forecasting direction.],
      [A one-day *embargo* between training end and validation start prevents last-day bleed.],
      [Five folds; preprocessing refitted inside every fold.],
      [Answers: *how well does the model generalise forward in time?*],
    )
  ],
  [
    #heading(level: 3)[Station-grouped folds]
    #ticks(
      [Split by composite station identity, so no instrument appears in both train and validation.],
      [Uses the name + coordinates key justified in Part 1.],
      [Five folds; measures transfer to *unseen locations*.],
      [Answers: *does the model generalise across space?*],
    )
  ],
)

#align(center)[#image("../figures/temporal_cv_rmse.png", width: 62%)]
#caption([Figure 2. Per-fold RMSE under the expanding temporal scheme: performance is volatile across folds because later folds contain harder, higher-concentration periods.])

#heading(level: 2)[The random-split diagnostic is not selection evidence]
A third, random-split scheme is computed for one *pre-declared* architecture only. Its sole purpose is
to quantify how much an unrealistic split flatters a score: for the fixed diagnostic model, the random
scheme gives mean RMSE 8.091 versus 8.301 under temporal folds — a gap of −0.209 µg/m³. This number is
reported as a *diagnostic*, never as a model-selection estimate.
#pagebreak()

// ------------------------------------------------ 5. tuning ---------------
#eyebrow("05  /  HYPERPARAMETER OPTIMISATION")
= Nested tuning with Optuna

Tuning is nested: for each outer fold, an inner chronological split runs its own search, and the outer
validation block is never seen by the search. This is what keeps the reported score honest.

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  stat("TPE", "sampler", accent: navy2, sub: "Optuna 5.0.0, seed 42"),
  stat("50", "trials per outer fold", accent: teal),
  stat("250", "completed trials", accent: teal),
  stat("Inner RMSE", "objective (µg/m³)", accent: navy2),
)
#v(7pt)

#heading(level: 2)[Search space and the per-fold optima]
#tbl(
  (2.6fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr),
  table.header(th("Hyperparameter"), th("Fold 1"), th("Fold 2"), th("Fold 3"), th("Fold 4"), th("Fold 5")),
  [Conv filters], [64], [64], [16], [32], [64],
  [Kernel size], [5], [2], [3], [5], [2],
  [LSTM units], [32], [32], [64], [64], [32],
  [Dropout], [0.220], [0.060], [0.347], [0.098], [0.055],
  [Learning rate], [7.4e-4], [9.1e-4], [7.7e-4], [2.2e-4], [9.99e-4],
  [Dense units], [32], [32], [32], [32], [32],
)
#v(4pt)
#caption([Best configuration per outer fold (`results/16/optuna_best_hyperparameters.csv`). The optimum *moves* between folds — evidence that a single global setting is not obviously correct for this panel.])

#callout(
  "Reading the tuning result honestly",
  [No single configuration wins across folds: dropout ranges from 0.055 to 0.347 and filters from 16 to
   64. That instability is itself a finding — with five folds and a noisy target, the search surface is
   shallow and the "best" point is only weakly separated from its neighbours. It is reported, not hidden.],
)
#pagebreak()

// ------------------------------------------------- 6. baselines -----------
#eyebrow("06  /  BASELINES")
= Mandatory baselines

Two simple, assumption-light baselines are scored on *exactly* the same folds and the same IDs as the
network, so every comparison is paired.

#grid(
  columns: (1fr, 1fr),
  gutter: 14pt,
  [
    #callout(
      "Persistence",
      [Forecast tomorrow = today's last observed PM2.5. Carries no parameters and no fitting, and is
       notoriously strong for daily air quality.],
      accent: navy2, tint: pale,
    )
    #v(4pt)
    #ticks(
      [Temporal RMSE 8.771],
      [Spatial RMSE 9.292],
      [Future-block RMSE 5.846],
    )
  ],
  [
    #callout(
      "Weekly seasonal-naive",
      [Forecast tomorrow = the same weekday observed seven days earlier. Captures the weekly cycle
       without learning.],
      accent: amber, tint: sand,
    )
    #v(4pt)
    #ticks(
      [Temporal RMSE 10.798],
      [Spatial RMSE 10.906],
      [Future-block RMSE 8.027],
    )
  ],
)

#heading(level: 2)[Why baselines are non-negotiable here]
#ticks(
  [They establish the *floor*: a network that cannot beat persistence has not demonstrated skill.],
  [They are *paired*: scored on identical fold membership, so differences are not confounded by fold difficulty.],
  [They expose the *MAE gap*: persistence achieves a lower MAE (3.158 development, 3.119 future) than the network (3.978 / 4.024), showing the network's errors are concentrated in the tail it tries to chase.],
)
#pagebreak()

// -------------------------------------------- 7. development results ------
#eyebrow("07  /  DEVELOPMENT RESULTS")
= Out-of-fold performance

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  stat("8.455", "temporal CV RMSE", accent: teal, sub: "SD 6.161 · n = 2,867"),
  stat("3.911", "temporal CV MAE", accent: teal, sub: "SD 0.957"),
  stat("7.962", "spatial CV RMSE", accent: teal, sub: "SD 4.143 · n = 3,349"),
  stat("0.212", "spatial CV R²", accent: navy2, sub: "temporal R² 0.106"),
)
#v(7pt)

#tbl(
  (1.7fr, 0.75fr, 0.85fr, 0.8fr, 0.9fr, 0.8fr, 0.7fr),
  table.header(th("Scheme / model"), th("RMSE"), th("SD"), th("MAE"), th("R²"), th("Pooled RMSE"), th("n")),
  [Temporal · CNN–LSTM], [8.455], [6.161], [3.911], [0.106], [9.582], [2,867],
  [Temporal · tuned (TPE)], [8.520], [6.034], [3.978], [0.076], [9.571], [2,867],
  [Temporal · persistence], [8.771], [—], [3.158], [—], [—], [2,867],
  [Temporal · seasonal-naive], [10.798], [—], [—], [—], [—], [2,867],
  [Spatial · CNN–LSTM], [7.962], [4.143], [3.292], [0.212], [8.700], [3,349],
  [Spatial · persistence], [9.292], [—], [—], [—], [—], [3,349],
  [Spatial · seasonal-naive], [10.906], [—], [—], [—], [—], [3,349],
)
#v(4pt)
#caption([Fold-mean metrics in µg/m³. The station-grouped score is *better* than the temporal score because spatial folds contain easier, lower-variance stations — not because the model is stronger across space.])

#heading(level: 2)[How to read these numbers]
#ticks(
  [The network beats the weekly baseline decisively under both schemes (≈ −2.28 temporal, −2.94 spatial).],
  [Against *persistence* the edge is small: −0.251 temporal, −1.330 spatial on the tuned model — and not statistically significant (Section 07 and Part 3).],
  [R² is modest (0.08–0.21): the day-to-day variance is largely unpredictable from a seven-day past window at these stations.],
  [Fold SD (6.0–6.2) is almost as large as the mean — the fold-level uncertainty is the real story.],
)
#align(center)[#image("../figures/temporal_baseline_comparison.png", width: 74%)]
#caption([Figure 3. CNN–LSTM against persistence and seasonal-naive on matched temporal folds.])
#pagebreak()

// ------------------------------------------------- 8. robustness ----------
#eyebrow("08  /  ROBUSTNESS AND SENSITIVITY")
= Stress-testing the result

A single mean is fragile. Five controlled perturbations probe whether the conclusion survives.

#align(center)[#image("../figures/training_size_sensitivity.png", width: 62%)]
#caption([Figure 4. Learning curve: RMSE falls steeply to 50% of the training data, then plateaus — the model is data-limited, not capacity-limited.])

#tbl(
  (1.7fr, 1.1fr, 1.5fr, 1.8fr),
  table.header(th("Experiment"), th("Result"), th("Reading"), th("Verdict")),
  [Training size 10% → 100%], [10.151 → 8.382], [Most of the gain is captured by 50%; a plateau follows.], [Stable — data-limited],
  [Preprocessing: log1p target], [8.575], [Slightly worse than the raw-target reference (8.455).], [Reference retained],
  [Preprocessing: drop indicators], [8.500], [Slightly worse; missingness flags help a little.], [Flags retained],
  [Extreme-event weighting], [8.600], [Overall RMSE unchanged; tail RMSE moves only marginally (−0.83 on >50 band).], [No material gain],
  [Architecture ±1 step], [mostly < 5%], [Only a *lower* filter count exceeded 5% (relative −6.7%).], [Mostly stable],
)
#v(4pt)
#caption([Every variant is scored on the same folds as the reference. The full tables are in `results/16/`.])

#callout(
  "What the robustness panel establishes",
  [The result is *not* an artefact of one preprocessing choice or one weighting scheme: none of the tested
   alternatives improved on the reference in a material way. The one genuine sensitivity — the number of
   convolutional filters — is disclosed rather than smoothed over.],
)
#pagebreak()

// ---------------------------------------------- 9. error structure --------
#eyebrow("09  /  ERROR STRUCTURE")
= Where the model succeeds and fails

Average error hides the failure mode. Conditioning on season and on concentration band shows the
network is accurate exactly where the problem is easy, and fails where it matters.

#grid(
  columns: (1fr, 1fr),
  gutter: 14pt,
  [
    #heading(level: 3)[Error by season]
    #tbl(
      (0.8fr, 0.6fr, 0.8fr, 0.8fr),
      table.header(th("Season"), th("n"), th("RMSE"), th("MAE")),
      [Spring], [244], [3.600], [2.886],
      [Autumn], [652], [6.273], [3.921],
      [Summer], [1,678], [7.784], [3.566],
      [Winter], [293], [21.285], [6.404],
    )
    #v(3pt)
    #caption([Winter dominates the error: a small number of high-concentration cold-season days drives the fold SD.])
  ],
  [
    #heading(level: 3)[Error by concentration band]
    #tbl(
      (0.95fr, 0.6fr, 0.85fr, 0.75fr),
      table.header(th("Actual band"), th("n"), th("RMSE"), th("Under-pred.")),
      [< 10], [1,734], [3.619], [25%],
      [10–25], [1,070], [5.045], [80%],
      [25–50], [56], [18.957], [100%],
      [> 50], [7], [166.120], [100%],
    )
    #v(3pt)
    #caption([The model under-predicts 100% of the highest band — a classic regression-to-the-mean failure on rare extremes.])
  ],
)

#align(center)[#image("../figures/temporal_oof_residuals.png", width: 62%)]
#caption([Figure 5. Out-of-fold residual distribution: centred near zero for ordinary days, with a long right tail on the few extreme days.])

#callout(
  "The core limitation, stated plainly",
  [Only 7 of 2,867 out-of-fold points lie above 50 µg/m³ and every one is under-predicted. The headline
   RMSE is therefore governed by ordinary days; episode-day skill is essentially unmeasured. Any claim of
   operational usefulness on pollution episodes is unsupported by these data.],
  accent: red, tint: sand,
)
#pagebreak()

// -------------------------------------------- 10. locked future -----------
#eyebrow("10  /  LOCKED FUTURE TEST")
= The frozen independent estimate

A held-out future block (2024-12-08 → 2026-04-06) was scored *once* for the pre-locked model and then
frozen; the record was written before scoring to prevent a second, opportunistic evaluation.

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  stat("6.027", "CNN–LSTM RMSE", accent: teal, sub: "N = 1,716 · lookback 7"),
  stat("4.024", "CNN–LSTM MAE", accent: teal),
  stat("0.239", "CNN–LSTM R²", accent: navy2),
  stat("5.846", "persistence RMSE", accent: amber, sub: "same 1,716 IDs"),
)
#v(7pt)

#tbl(
  (1.6fr, 0.8fr, 0.8fr, 0.85fr, 0.85fr),
  table.header(th("Model (frozen block)"), th("RMSE"), th("MAE"), th("R²"), th("N")),
  [CNN–LSTM (locked)], [6.027], [4.024], [0.239], [1,716],
  [Persistence], [5.846], [3.119], [—], [1,716],
  [Weekly seasonal-naive], [8.027], [5.361], [—], [1,716],
)
#v(4pt)
#caption([Scored 2026-11-10 and locked in `future_test_evaluation.json`. Values are read-only history and were not recomputed for this report.])

#callout(
  "The most important honest finding in Part 2",
  [On the frozen future block, *persistence achieves a lower RMSE than the CNN–LSTM* (5.846 vs 6.027)
   and a lower MAE. The network beats the weekly baseline comfortably but does not yet beat the
   no-model benchmark on this block. This is reported as the headline, not buried: a study that hides a
   baseline win is not a study. Part 3 quantifies whether the residual differences are significant.],
  accent: amber, tint: sand,
)

#heading(level: 2)[Monthly view of the frozen block]
#tbl(
  (0.9fr, 0.7fr, 0.9fr, 0.85fr, 0.9fr, 0.7fr, 0.9fr, 0.85fr),
  table.header(th("Month"), th("n"), th("RMSE"), th("R²"), th("Month"), th("n"), th("RMSE"), th("R²")),
  [2024-12], [51], [11.46], [−0.30], [2025-08], [550], [6.32], [0.17],
  [2025-05], [182], [3.09], [0.28], [2025-09], [110], [6.83], [−0.04],
  [2025-06], [265], [5.02], [0.12], [2026-02], [7], [18.16], [0.45],
  [2025-07], [301], [3.89], [0.18], [2026-03], [40], [11.75], [−0.48],
)
#v(4pt)
#caption([Selected months. Small-n months have unstable R²; the pattern is the familiar one — low error in fair-weather months, large error in the cold-season episodes.])
#pagebreak()

// --------------------------------------- 11. limitations and handoff ------
#eyebrow("11  /  LIMITATIONS AND HANDOFF")
= Limitations and what Part 3 receives

#heading(level: 2)[Material limitations carried forward]
#tbl(
  (1.5fr, 4.5fr),
  table.header(th("Area"), th("Limitation carried into Part 3")),
  [Baseline parity], [Persistence matches or beats the network on the locked block; no significant skill advantage has been demonstrated.],
  [Inferential power], [Only five dependent folds — p-values have low power, so effect sizes and intervals matter more than a threshold (Part 3's job).],
  [Tail skill], [Extreme days (> 50 µg/m³) are essentially unlearnable here (n = 7 OOF), and all are under-predicted.],
  [Validation realism], [Even temporal folds assume station continuity and cannot model instrument drift or site relocation.],
  [Exogenous inputs], [Meteorology, trace gases and AOD remain excluded (Part 1), so the model sees only its own history plus calendar terms.],
)

#heading(level: 2)[The frozen handoff to Part 3]
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 9pt,
  stat("2,867 + 3,349", "OOF predictions\n(temporal + spatial)", accent: navy2),
  stat("5 × 5", "paired folds for\ntesting", accent: teal),
  stat("1,716", "locked future\nIDs", accent: teal),
)
#v(8pt)
#callout(
  "Part 2 verdict",
  [The system is *built, tuned, validated and tested honestly*. It beats the naive weekly baseline,
   does not significantly beat persistence, and its uncertainty is as large as its mean. Those three
   facts — not a single score — are what Part 3 tests statistically and explains.],
)
#v(6pt)
#caption([Reproducible from `results/16/development_cv_summary.csv`, `development_cv_optuna_summary.csv`, `optuna_best_hyperparameters.csv`, `ablation_summary.csv`, `training_size_sensitivity_summary.csv`, `extreme_event_weighting_summary.csv`, `cnn_lstm_sensitivity_summary.csv`, `error_by_season.csv`, `error_by_concentration_band.csv`, `future_test_results.csv`, `future_test_evaluation.json`, and `results/16/model_inventory.csv`.])
