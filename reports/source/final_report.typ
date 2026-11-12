#set page(
  paper: "a4",
  margin: (x: 18mm, top: 17mm, bottom: 18mm),
  footer: context [
    #line(length: 100%, stroke: 0.45pt + rgb("#D5DEE7"))
    #v(3pt)
    #align(right)[#text(size: 8pt, fill: rgb("#607184"))[GROUP 16  ·  PM2.5 FORECASTING  ·  #counter(page).display()]]
  ],
)
#set text(font: "Arial", size: 9.5pt, fill: rgb("#243447"))
#set par(justify: true, leading: 0.68em, spacing: 0.55em)
#set heading(numbering: "1.")
#show heading.where(level: 1): set text(size: 18pt, weight: "bold", fill: rgb("#10243A"))
#show heading.where(level: 2): set text(size: 12pt, weight: "bold", fill: rgb("#087F80"))
#show heading.where(level: 3): set text(size: 10pt, weight: "bold", fill: rgb("#34495E"))

#let navy = rgb("#10243A")
#let teal = rgb("#087F80")
#let amber = rgb("#D9822B")
#let ink = rgb("#243447")
#let muted = rgb("#607184")
#let pale = rgb("#EEF4F6")
#let line-color = rgb("#D5DEE7")
#let cite(n) = text(size: 8pt, fill: muted)[#n]
#let section-label(label) = text(size: 8pt, weight: "bold", fill: teal, tracking: 1.1pt)[#label]
#let callout(title, body) = block(
  fill: pale,
  stroke: (left: 3pt + teal),
  inset: (x: 10pt, y: 8pt),
  radius: 3pt,
)[
  #text(weight: "bold", fill: navy)[#title]
  #linebreak()
  #body
]
#let node(title, detail, tone: pale) = block(
  fill: tone,
  stroke: 0.7pt + line-color,
  inset: (x: 6pt, y: 7pt),
  radius: 4pt,
)[
  #align(center)[
    #text(size: 8.5pt, weight: "bold", fill: navy)[#title]
    #linebreak()
    #text(size: 7.2pt, fill: muted)[#detail]
  ]
]
#let th(s) = text(size: 8pt, weight: "bold", fill: white)[#s]
#let tbl(cols, ..cells) = table(
  columns: cols,
  inset: (x: 6pt, y: 5pt),
  stroke: (x, y) => if y == 0 { 0pt } else { 0.45pt + line-color },
  fill: (x, y) => if y == 0 { navy } else if calc.rem(y, 2) == 0 { pale } else { white },
  align: left,
  ..cells,
)

// 1. Cover
#set page(footer: none)
#v(18mm)
#text(size: 9pt, weight: "bold", fill: teal, tracking: 2pt)[TECHNICAL REPORT  /  GROUP 16]
#v(12mm)
#text(size: 34pt, weight: "bold", fill: navy)[
  One-day-ahead
  #linebreak()
  PM2.5 forecasting
]
#v(5mm)
#text(size: 17pt, fill: teal)[A leakage-aware CNN–LSTM evaluation]
#v(10mm)
#line(length: 36mm, stroke: 2.5pt + teal)
#v(8mm)
#text(size: 12pt, fill: ink)[
  A station-level regression study using past-only features, expanding temporal validation, and matched baseline comparisons.
]
#v(16mm)
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 8pt,
  node("40,983", "clean station-day records"),
  node("95", "cleaned stations"),
  node("5", "expanding temporal folds"),
)
#v(12mm)
#callout(
  "Submission integrity note",
  [The reported future-period score is a frozen historical estimate, not a newly run benchmark. It is presented separately from development cross-validation. Compiled-data redistribution rights and part of the upstream row-inclusion logic remain unresolved.]
)
#v(1fr)
#text(size: 9pt, fill: muted)[
  Group 16  ·  English report  ·  12 November 2026
]
#pagebreak()

// 2. Executive summary and questions
#set page(footer: context [
  #line(length: 100%, stroke: 0.45pt + line-color)
  #v(3pt)
  #align(right)[#text(size: 8pt, fill: muted)[GROUP 16  ·  PM2.5 FORECASTING  ·  #counter(page).display()]]
])
#section-label("EXECUTIVE SUMMARY")
#heading(level: 1)[Study at a glance]
#text(size: 9.8pt)[
  This study predicts next-day station PM2.5 concentration from observations available through the end of the current day. PM2.5 forecasting is a regression problem, not classification: the output is a continuous concentration in µg/m³, not a class label. The cleaned panel contains 40,983 station-day observations at 95 coordinate-defined stations, from 25 November 2016 to 6 April 2026. Development model selection and cross-validation stop before 8 December 2024.
]
#callout(
  "Main research question",
  [How accurately can next-day PM2.5 be forecast at a monitoring station from information available at forecast issue time, and does a CNN–LSTM improve on simple past-only forecasts under leakage-safe validation?]
)
#v(5pt)
#tbl(
  (0.55fr, 4.45fr),
  table.header(th("RQ"), th("Research question")),
  [RQ1], [How does CNN–LSTM compare with persistence and seasonal-naive baselines on later development periods?],
  [RQ2], [What do temporal and station-grouped evaluations say about generalization across time and locations?],
  [RQ3], [Which available feature groups contribute to forecast performance, and how stable is their contribution?],
  [RQ4], [How do seasonality, sparse coverage, extreme events, missingness, and data rights constrain trustworthy use?],
)
#v(5pt)
#heading(level: 2)[Headline findings]
#list(
  [Five-fold temporal estimates are highly variable. The tuned CNN–LSTM mean-fold RMSE was 8.520 µg/m³; persistence was 8.771. The CNN–LSTM had higher mean-fold MAE (3.978 vs 3.158), and the paired fold test did not establish a significant RMSE gain (exact Wilcoxon *p* = .625).],
  [A separate, fixed-configuration diagnostic on a matched cohort gave mean-fold RMSE / MAE / R² of 8.301 / 3.845 / 0.156 for CNN–LSTM and 8.771 / 3.158 / 0.207 for persistence. These diagnostic values are not the tuned-model estimates above.],
  [The frozen historical future estimate is RMSE 6.027, MAE 4.024, R² 0.239 across 1,716 predictions. The missing original weight file cannot be verified byte-for-byte; the stored predictions were reproduced by a development-only refit without evaluating holdout labels.]
)
#pagebreak()

// 3. Motivation and task
#section-label("01  /  CONTEXT")
#heading(level: 1)[Why forecast PM2.5?]
#text[
  Fine particulate matter (PM2.5) is a key air-quality indicator because small particles can penetrate deeply into the respiratory system and are associated with adverse health outcomes. Short-horizon forecasts can help monitoring teams anticipate deteriorating conditions, prioritize review of stations, communicate timely risk information, and plan follow-up measurements. A forecast is decision support, not a clinical or regulatory determination; its value depends on calibrated, location-appropriate performance and reliable data latency. [#cite("1")]
]
#heading(level: 2)[A continuous target and a defined forecast origin]
#text[
  For station $s$ on day $t$, the model uses data available through the end of day $t$ to estimate $y_(s,t+1)$, the next day’s PM2.5 concentration in µg/m³. Since the response is numeric and continuous, evaluating it as classification would discard concentration magnitude and make the stated RMSE, MAE, and R² inappropriate. We therefore treat this project explicitly as regression.
]
#grid(
  columns: (1fr, 1fr),
  gutter: 9pt,
  node("Forecast origin", "End of day t; no future target information enters the feature sequence.", tone: rgb("#EAF3F3")),
  node("Forecast horizon", "One day ahead: predict station concentration at t + 1.", tone: rgb("#F8F1E8")),
)
#heading(level: 2)[Why combine convolution and recurrence?]
#text[
  A one-dimensional temporal convolution scans a short local window and can learn recurring short-range shapes—such as a rise, decline, or local fluctuation—through shared filters. The LSTM then reads the ordered sequence of convolutional representations and uses gated memory (input, forget, and output gates) to retain or update information over time. The dense regression head maps the learned sequence representation to one real-valued concentration.
]
#text[
  This combination is structurally appropriate for a short temporal sequence: the convolution provides local pattern extraction, while recurrence represents ordered dependencies. It is not guaranteed to beat a simpler model. With only seven daily history steps, strong autocorrelation, sparse station panels, and few extreme events, the additional capacity may be unnecessary or unstable. That is why the study compares it with persistence and reports nonsignificant results without claiming architectural superiority. [#cite("2")][#cite("3")]
]
#callout(
  "Interpretation boundary",
  [The analysis estimates predictive association in this compiled station panel. It does not identify causal effects of pollutants, prove health benefits, or establish that a live warning system would have the same performance.]
)
#pagebreak()

// 4. Data, station identity, seasonality
#section-label("02  /  DATA AND PROVENANCE")
#heading(level: 1)[Dataset, cleaning, and seasonal context]
#text[
  The input is the publisher-prepared *Air Quality – Mediterranean Basin (2016–2026)* bundle, version 1. The combined file contains 41,718 rows; the project’s prescribed target filter removes 735 records with PM2.5 ≤ 0 or ≥ 500 µg/m³, leaving 40,983 records across 95 stations. The retained target median is 7.85 µg/m³; the 95th and 99th percentiles are 21.65 and 33.56 µg/m³. Duplicate station-day keys and malformed target/date rows were not found after cleaning.
]
#text[
  Kaggle’s captured metadata labels the compiled bundle “Other (specified in description)” rather than naming a standard redistribution license. The publisher does not provide a compiled-bundle license or a complete component-attribution map. Redistribution rights therefore remain unresolved; the dataset is not presented as cleared for redistribution. [#cite("4")][#cite("6")][#cite("7")][#cite("8")]
]
#heading(level: 2)[Station identity and count reconciliation]
#text[
  A station ID is constructed from the source sensor name plus latitude and longitude rounded to six decimal places. This preserves separately located instruments even when names repeat. The audit reconciles 98 numeric source IDs to 98 composite stations; three test entries were removed by target cleaning, leaving 95. There are 94 distinct sensor names, and four names map to two coordinate-defined stations. No source ID spans multiple composite stations. [#cite("4")]
]
#heading(level: 2)[Seasonal distribution and threshold frequency]
#text[
  Seasonal summaries are descriptive station-day statistics, not regulatory compliance rates. The project’s 25 and 50 µg/m³ cutoffs are used to describe the right tail; they are not asserted here as health standards. Winter has the highest seasonal mean and the largest share above 25 and 50 µg/m³.
]
#tbl(
  (0.7fr, 0.8fr, 0.9fr, 1fr, 1fr, 1fr),
  table.header(th("Season"), th("Records"), th("Mean"), th("Median"), th(">25"), th(">50")),
  [DJF], [8,938], [10.94], [8.77], [6.15%], [0.45%],
  [MAM], [9,276], [8.25], [6.83], [1.92%], [0.36%],
  [JJA], [12,766], [9.59], [8.50], [1.95%], [0.15%],
  [SON], [10,003], [9.08], [7.47], [2.46%], [0.25%],
)
#v(4pt)
#text(size: 8pt, fill: muted)[Mean and median in µg/m³; percentages are record shares above the stated concentration cutoffs.]
#pagebreak()

// 5. Workflow and leakage prevention
#section-label("03  /  PREPROCESSING")
#heading(level: 1)[Processing workflow and leakage controls]
#grid(
  columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr),
  gutter: 3pt,
  node("Original CSV", "41,718 prepared rows"),
  text(size: 18pt, fill: teal)[→],
  node("Cleaning", "Apply target bounds\ncheck station keys"),
  text(size: 18pt, fill: teal)[→],
  node("Preprocessing", "Fold-local imputation and scaling"),
  text(size: 18pt, fill: teal)[→],
  node("Features", "7-day past-only\nno date gaps"),
  text(size: 18pt, fill: teal)[→],
  node("Training", "TPE selection inside training; score outer fold"),
)
#heading(level: 2)[What was done to prevent leakage]
#list(
  [The primary temporal design uses five expanding folds over whole dates, with a one-day raw-date embargo. All feature windows end at the forecast origin; each label is exactly the following station-day.],
  [PM2.5 lag and rolling features are calculated from prior observations only, grouped by station. Sequences crossing station boundaries or missing-day gaps are rejected.],
  [Median imputation and standardization are fitted on the training partition of each fold only. Hyperparameter selection uses chronological inner validation drawn only from the outer training partition.],
  [Meteorology, AOD, NO₂, CO, and O₃ were not used in the final model because timestamped availability at the forecast origin is unverified. Their observation dates alone do not prove that values could be operationally received by issue time.],
  [Automated leakage tests verify origin/target alignment, recompute past-only rolling values, mutate values at and after a cutoff and check that earlier features do not change, and confirm fold-local preprocessing. The frozen future block was not used for feature/model selection.]
)
#callout(
  "Verification boundary",
  [The checks support leakage control in the executed pipeline. They do not prove that the publisher’s original upstream join was leakage-free or reconstruct the source publisher’s complete row-inclusion process.]
)
#pagebreak()

// 6. Architecture
#section-label("04  /  MODEL")
#heading(level: 1)[CNN–LSTM architecture]
#text[
  The reference sequence has seven daily steps. The tuned configuration starts from seven calendar/target-history channels; fold-fitted preprocessing can add missingness flags for the rolling target summaries. The sequence is passed through a causal convolution stack, then a recurrent encoder and a scalar output layer.
]
#grid(
  columns: (1fr, auto, 1.1fr, auto, 1fr, auto, 1.05fr, auto, 0.85fr),
  gutter: 3pt,
  node("Input", "7 days × 9 processed channels\n(7 features + 2 flags)", tone: rgb("#EAF3F3")),
  text(size: 18pt, fill: teal)[→],
  node("Causal Conv1D", "16 filters · kernel 3\nlocal temporal motifs", tone: rgb("#E6EEF6")),
  text(size: 18pt, fill: teal)[→],
  node("Norm + ReLU", "LayerNorm\nDropout 0.347", tone: rgb("#EEF4F6")),
  text(size: 18pt, fill: teal)[→],
  node("LSTM", "64 units\nsequence memory", tone: rgb("#F8F1E8")),
  text(size: 18pt, fill: teal)[→],
  node("Dense head", "32 ReLU\nscalar regression", tone: rgb("#EAF3F3")),
)
#v(5pt)
#callout(
  "How the two components work together",
  [The convolution shares filters across nearby time positions, extracting local temporal motifs. Layer normalization and dropout regularize the learned channels. Max pooling reduces the sequence length. The LSTM processes the remaining ordered representations and its gates control what history to retain, forget, or expose. A dense head returns one continuous PM2.5 prediction. This design is suitable for local-plus-sequential structure, but the experiments do not establish that it is superior to a naive forecast.]
)
#heading(level: 2)[Training configuration]
#tbl(
  (1fr, 2fr, 1fr, 2fr),
  table.header(th("Setting"), th("Value"), th("Setting"), th("Value")),
  [Lookback], [7 consecutive days], [Objective], [Mean squared error],
  [Optimizer], [Adam], [Learning rate], [TPE-selected; fold-specific],
  [Conv filters / kernel], [16 / 3 in selected reference], [LSTM / dense], [64 / 32 units],
  [Dropout], [0.347], [Tuning], [50 Optuna TPE trials per outer fold],
)
#v(4pt)
#text(size: 8pt, fill: muted)[Hyperparameters vary by temporal fold; the table summarizes the selected reference configuration recorded in the model metadata. The final feature-selected development refit uses only PM2.5 lag-1, and is not an independent evaluation.]
#pagebreak()

// 7. Training and validation evidence
#section-label("05  /  FITTING AND VALIDATION")
#heading(level: 1)[Training trace and validation design]
#text[
  The archived inner-fit history below shows training and chronological inner-validation MSE for a representative selected CNN–LSTM fit. The best recorded inner-validation loss is 86.86 (µg/m³)² at epoch 45; training loss continues lower. The widening gap is consistent with overfitting in this fit. This curve is not the five-fold outer-validation score and is not a training curve for the frozen historical checkpoint.
]
#align(center)[#image("../../results/16/figures/training_validation_loss_typst.png", width: 94%)]
#text(size: 8pt, fill: muted)[Figure 1. Archived 55-epoch inner-fit history (`cnn_lstm_optuna_final_inner_history.csv`). MSE is shown in squared target units.]
#heading(level: 2)[Outer temporal folds]
#tbl(
  (0.55fr, 1fr, 1fr, 0.9fr, 0.8fr),
  table.header(th("Fold"), th("Validation period"), th("Train sequences"), th("Validation n"), th("TPE trials")),
  [1], [30 Apr 2019–16 Oct 2020], [482], [322], [50],
  [2], [17 Oct 2020–24 Oct 2021], [801], [172], [50],
  [3], [25 Oct 2021–18 Nov 2022], [976], [377], [50],
  [4], [19 Nov 2022–30 Jan 2024], [1,353], [733], [50],
  [5], [31 Jan–7 Dec 2024], [2,086], [1,263], [50],
)
#text(size: 8pt, fill: muted)[Five expanding folds; 2,867 scored outer predictions total. Tuning occurs within each outer training window. Training and validation fold composition varies by time and available station sequences.]
#heading(level: 2)[Why the estimates are uncertain]
#text[
  Fold 3 has a large PM2.5 episode and a substantially higher RMSE than the other folds. The target distribution and available stations change over time, so mean-fold metrics have large standard deviations and should not be read as a precise estimate of performance in every season or location. We report matched-fold results, pooled prediction metrics, and uncertainty separately rather than presenting one split as universal.
]
#pagebreak()

// 8. Evaluation results
#section-label("06  /  RESULTS")
#heading(level: 1)[Evaluation against simple baselines]
#text[
  Primary comparisons use the same five temporal folds and target convention. RMSE penalizes large errors; MAE describes typical absolute error; R² compares residual variation with a fold’s observed target variance. The tuned fold-wise CNN–LSTM mean RMSE is 8.520 (SD 6.034) and mean MAE is 3.978 (SD 0.892). Persistence predicts the last observed concentration at the forecast origin. Weekly seasonal-naive predicts from the corresponding prior week.
]
#tbl(
  (1.55fr, 1fr, 1fr, 0.9fr),
  table.header(th("Matched diagnostic model"), th("RMSE"), th("MAE"), th("R²")),
  [Fixed CNN–LSTM], [8.301 ± 6.221], [3.845 ± 0.921], [0.156 ± 0.101],
  [Persistence], [8.771 ± 8.984], [3.158 ± 1.468], [0.207 ± 0.476],
  [Weekly seasonal-naive], [10.798 ± 10.039], [4.721 ± 1.718], [−0.287 ± 0.525],
)
#text(size: 8pt, fill: muted)[Mean ± sample SD across five folds; µg/m³ for RMSE and MAE. This is the fixed-configuration diagnostic table; it is deliberately not mixed with the per-fold TPE-selected scores.]
#heading(level: 2)[Paired inference and confidence intervals]
#list(
  [For the TPE-selected CNN–LSTM versus persistence, mean-fold RMSE was 8.520 versus 8.771, but mean-fold MAE favored persistence (3.978 versus 3.158). The exact paired five-fold Wilcoxon test gave *p* = .625; Holm-adjusted *p* = 1.00.],
  [The pooled RMSE difference (CNN–LSTM minus persistence) was −1.131 µg/m³, with a station-cluster bootstrap 95% interval [−2.824, 0.797] across 72 station clusters. The interval includes zero.],
  [Only five dependent fold pairs provide low inferential power. Bootstrap intervals quantify resampling uncertainty conditional on this dataset and split design; they do not include dataset shift, model-search multiplicity, or licensing/provenance uncertainty.]
)
#heading(level: 2)[Frozen historical future estimate]
#callout(
  "Separate locked result — not a baseline comparison",
  [RMSE 6.0272 µg/m³ · MAE 4.0240 µg/m³ · R² 0.2394 · n = 1,716. The project record says the result was previously scored; this task did not access labels or compute new holdout metrics. A development-only reconstruction matched the stored predictions to maximum absolute difference 7.1 × 10⁻¹⁵, but the original weight bytes are unavailable. No persistence or seasonal-naive future score exists in the saved locked result, so no future-period model superiority claim is made.]
)
#pagebreak()

// 9. Seasonal and diagnostic limits
#section-label("07  /  ROBUSTNESS")
#heading(level: 1)[Seasonality, temporal diagnostics, and coverage]
#text[
  The observed winter mean (10.94 µg/m³) and 6.15% share above 25 µg/m³ exceed the corresponding values in the other seasons. Winter also has the highest >50 share (0.45%). These are pooled record-level descriptions; stations contribute different counts and the distribution is highly right-skewed. An observed seasonal pattern does not establish a meteorological cause.
]
#align(center)[#image("../../results/16/figures/pm25_by_season.png", width: 78%)]
#text(size: 8pt, fill: muted)[Figure 2. Seasonal PM2.5 summaries from the cleaned station-day panel. Threshold percentages are provided on the preceding page.]
#heading(level: 2)[Why only six stations support some tests]
#text[
  To avoid treating missing dates as if they were consecutive observations, station diagnostics use each station’s longest uninterrupted daily development segment. ACF/PACF summaries require at least 14 consecutive days and are available for 63 stations. Formal ADF, KPSS, and Ljung–Box tests require at least 30 consecutive daily observations and are possible for only six stations. Sparse and irregular records, combined with the development cutoff, leave most station segments too short. Results from six stations are low-coverage diagnostics, not basin-wide evidence of stationarity or serial dependence.
]
#heading(level: 2)[Lead–lag coverage]
#text[
  Pollutant/PM2.5 lead–lag correlations use uninterrupted segments and require sufficient non-missing pairs. The saved summary has usable pairs at six stations; per-lag sample sizes range from two to six stations. NO₂/CO/O₃ have approximately 5.9–6.0% missing values overall, with station-month gaps. These correlations are descriptive, underpowered, and non-causal. They do not justify adding pollutant values as predictors without a timestamped delivery/availability audit.
]
#callout(
  "Spatial-validation interpretation",
  [Five station-grouped folds yielded pooled RMSE 8.700, MAE 3.287, and R² 0.154 on 3,349 predictions. The folds test held-out stations conditional on having historical station sequences; they do not test a genuinely cold-start station with no history.]
)
#pagebreak()

// 10. Explainability and ablation
#section-label("08  /  EXPLAINABILITY")
#heading(level: 1)[What information mattered?]
#text[
  Temporal channel permutation importance measures the change in fold RMSE when a channel’s values are permuted. It is a model-conditional diagnostic, not a causal effect. Lag-1 PM2.5 ranked first in each of the five folds, consistent with persistence being a strong baseline. Rolling summaries had weaker and less stable importance; calendar signals were small or negative on average.
]
#tbl(
  (2fr, 0.9fr, 0.9fr, 0.8fr),
  table.header(th("Input channel"), th("Mean ΔRMSE"), th("SD"), th("Mean rank")),
  [PM2.5 lag-1], [0.651], [0.443], [1.0],
  [PM2.5 rolling-3 mean], [0.153], [0.167], [3.0],
  [Rolling-7 missing flag], [0.052], [0.067], [4.8],
  [Day-of-year sine], [0.033], [0.059], [5.6],
  [PM2.5 rolling-3 missing flag], [0.017], [0.049], [5.8],
)
#text(size: 8pt, fill: muted)[Importance in µg/m³ RMSE change; five temporal folds. Small/negative values should not be interpreted as proof that a feature is harmful.]
#heading(level: 2)[Missingness indicators and inactive pollutant channels]
#text[
  In the reference preprocessing, missingness flags are available for derived PM2.5 rolling features. The final feature-selected development refit uses only PM2.5 lag-1. NO₂, CO, and O₃ indicators are not active in that final model because those pollutant channels themselves are excluded: their observation timestamps do not establish that values could be received by forecast issue time. Activating missingness indicators for unavailable pollutants could encode station coverage or measurement practices without providing an operationally available predictor. Pollutant missingness is reported descriptively, not imputed into the final model.
]
#heading(level: 2)[Ablation of approved feature groups]
#align(center)[#image("../../results/16/figures/available_feature_ablation.png", width: 75%)]
#text[
  Removing lag and rolling history increased mean-fold RMSE by 0.617 (fold-bootstrap 95% interval 0.249 to 1.022). Removing calendar terms decreased it by 0.467 (interval −0.957 to −0.088). Removing rolling features alone decreased it by 0.177 (interval −0.337 to −0.043). These fixed, non-retuned comparisons suggest target history is useful and calendar/rolling expansions are not consistently helpful. Exogenous-family ablations are marked blocked, not reported as negative results, because forecast-time availability is unverified.
]
#pagebreak()

// 11. Error analysis and limitations
#section-label("09  /  ERROR ANALYSIS")
#heading(level: 1)[Where does the model fail?]
#text[
  The OOF residual convention below is actual minus prediction; a positive residual means underprediction. Errors rise sharply with concentration, and the sample count above 50 µg/m³ is very small. The examples below are representative development OOF cases, not future-period observations and not causal explanations.
]
#tbl(
  (1.3fr, 0.95fr, 0.95fr, 0.95fr, 1.1fr),
  table.header(th("Case"), th("Actual"), th("Predicted"), th("Absolute error"), th("Interpretation")),
  [Low: 28 Jun 2020], [5.0], [3.96], [1.04], [Near-background level; relatively small miss],
  [Moderate: 21 Jan 2022], [17.5], [12.96], [4.54], [Missed part of a concentration rise],
  [Elevated: 4 Feb 2022], [37.92], [13.88], [24.04], [Strong underprediction],
  [Extreme: 13 Feb 2022], [124.33], [15.73], [108.60], [Severe event underprediction],
  [Largest error: 14 Feb 2022], [338.80], [16.16], [322.64], [Peak far outside the learned typical range],
)
#text(size: 8pt, fill: muted)[Concentrations and errors in µg/m³. Selected CNN–LSTM temporal OOF examples from `local_development_cases.csv`; the extreme cases are only seven observations above 50 µg/m³.]
#heading(level: 2)[Limitations]
#list(
  [The compiled data bundle’s redistribution license and source-by-source attribution mapping are unresolved. The publisher’s row-level correspondences can be matched, but the full inclusion rule for omitted source rows and original ETL cannot be proven.],
  [Sparse, irregular station coverage limits formal temporal diagnostics and pollutant lead–lag analysis to six stations; high-concentration predictions are unstable due to few examples.],
  [The final model uses a short past-only history and has no verified meteorology, satellite, or trace-gas forecasts. It cannot represent unobserved future emissions or atmospheric conditions.],
  [Five dependent temporal folds give weak statistical power. The historical checkpoint is missing, so prediction equivalence is documented but binary model identity is not.]
)
#callout(
  "Practical conclusion",
  [The CNN–LSTM is a reasonable research candidate, but the evidence does not justify preferring it for deployment over persistence. It has not demonstrated a reliable overall gain, its MAE is worse in the paired mean-fold comparison, and it misses rare high-concentration events. Use only with a simpler baseline, calibrated uncertainty, verified real-time input availability, and explicit human oversight.]
)
#pagebreak()

// 12. Discussion and conclusion
#section-label("10  /  DISCUSSION AND CONCLUSION")
#heading(level: 1)[Discussion: answers to RQ1–RQ4]
#tbl(
  (0.52fr, 4.48fr),
  table.header(th("RQ"), th("Answer")),
  [RQ1], [No reliable improvement is established. Mean-fold RMSE is slightly lower for the tuned CNN–LSTM than persistence (8.520 vs 8.771), but MAE is higher (3.978 vs 3.158); the paired test is nonsignificant (*p* = .625) and the pooled RMSE interval includes zero.],
  [RQ2], [Performance varies sharply across time. Station-grouped CV gives pooled RMSE 8.700 and MAE 3.287, but uses historical sequences at held-out stations; it does not demonstrate cold-start transfer.],
  [RQ3], [Lag-1 is the clearest useful feature. Removing target history worsens error; calendar and rolling features provide no consistent benefit. Feature-family results vary by fold, and unverified exogenous variables are excluded rather than treated as proven useful.],
  [RQ4], [Winter records have the highest mean and threshold exceedance shares, yet extreme events are badly underpredicted. Only six stations support formal temporal and lead–lag diagnostics. Unresolved redistribution rights and upstream inclusion logic further limit reuse.]
)
#heading(level: 2)[Conclusion: concise answers and recommendation]
#text[
  In conclusion, RQ1 is answered “no reliable gain established”; RQ2 is answered “temporal performance varies and spatial evidence is conditional on station history”; RQ3 is answered “lag-1 is the clearest contributor, while other feature groups are mixed”; and RQ4 is answered “seasonality, rare-event error, sparse diagnostics, and unresolved data rights materially constrain use.”
]
#text[
  The study establishes a leakage-aware one-day-ahead regression protocol and provides auditable development-time comparisons. CNN–LSTM’s local temporal filters and recurrent memory are a plausible way to learn short-range patterns plus ordered dependencies, but architectural plausibility is not evidence of superior forecasting. Current results are mixed, uncertainty is large, and rare high-concentration episodes remain poorly captured.
]
#text[
  Therefore, CNN–LSTM is *not yet worth selecting over persistence as an operational recommendation*. Keep it as an experimental candidate; use persistence as the minimum comparator and require prospective, time-stamped evaluation, larger high-pollution coverage, calibrated intervals, and a documented compiled-data license before deployment or redistribution. This answer is deliberately conservative and does not infer a future-period baseline win from the frozen historical score.
]
#heading(level: 2)[Submission and reproducibility note]
#text[
  The report is built from `reports/source/final_report.typ`; the training-loss figure is generated from the saved inner history by `build_typst_report.py`. Evidence tables and plots are retained under `results/16/`. The source dataset should not be redistributed until the compiled bundle’s rights and attribution are clarified. References and additional evidence notes follow in the appendix and are excluded from the main-text page count.
]
#pagebreak()

// Appendix
#set heading(numbering: none)
#section-label("APPENDIX A")
#heading(level: 1)[References]
#set par(justify: false, leading: 0.62em, spacing: 0.42em)
#text(size: 8.3pt)[
  [1] World Health Organization. (2021). *WHO global air quality guidelines: particulate matter (PM₂.₅ and PM₁₀), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide.* https://www.who.int/publications/i/item/9789240034228

  [2] Kim, H. S., et al. (2022). Development of a CNN+LSTM Hybrid Neural Network for Daily PM2.5 Prediction. *Atmosphere, 13*(12), 2124. https://doi.org/10.3390/atmos13122124

  [3] Bai, X., et al. (2024). Prediction of PM2.5 concentration based on a CNN-LSTM neural network algorithm. *PeerJ, 12*, e17811. https://doi.org/10.7717/peerj.17811

  [4] Ameziane, S. (v1). *Air Quality – Mediterranean Basin (2016–2026).* Kaggle. https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026

  [5] Hewamalage, H., Ackermann, K., & Bergmeir, C. (2023). Forecast evaluation for data scientists: common pitfalls and best practices. *Data Mining and Knowledge Discovery.* https://doi.org/10.1007/s10618-022-00894-5

  [6] OpenAQ. Terms of use and license resources. https://docs.openaq.org/about/terms ; https://docs.openaq.org/resources/licenses

  [7] Copernicus Climate Data Store. Sentinel data legal notice and products license. https://cds.climate.copernicus.eu/licences/ec-sentinel ; https://cds.climate.copernicus.eu/licences/licence-to-use-copernicus-products

  [8] NASA. Science information policy. https://science.nasa.gov/researchers/science-information-policy/

  [9] Group 16 project materials: *Updated version PM25 Forecasting Mini Project*; *PM25 CNN-LSTM Layers and Libraries*; and *All files CSV guide*. Supplied course materials, 2026.

  [10] Project evidence: `results/16/temporal_cv_optuna_results.csv`, `diagnostic_cv_results.csv`, `development_cv_summary.csv`, `spatial_cv_results.csv`, `seasonal_summary.csv`, `feature_importance_summary.csv`, `ablation_summary.csv`, `local_development_cases.csv`, `future_test_results.csv`, and `station_identity_reconciliation_summary.json`.
]
#pagebreak()
#section-label("APPENDIX B")
#heading(level: 1)[Evidence boundaries and defense points]
#tbl(
  (1.1fr, 3.9fr),
  table.header(th("Topic"), th("Defensible statement")),
  [Data rights], [The publisher metadata says “Other (specified in description)”; the compiled bundle’s redistribution rights and attribution mapping are unresolved.],
  [Leakage], [Temporal order, past-only features, fold-local preprocessing, and one-day target alignment were tested. Exogenous channels were excluded because issue-time availability is unverified.],
  [Frozen future result], [Previously saved metrics are reported as locked historical evidence; this run did not read holdout labels or compute new test scores.],
  [Significance], [Five dependent fold pairs are underpowered; report effect sizes and intervals, not a binary winner.],
  [Model choice], [A CNN–LSTM is suitable to test local-plus-sequential structure, but it has not shown a reliable operational advantage over persistence.],
  [Known failure mode], [Very high PM2.5 events are underpredicted; only seven OOF targets exceed 50 µg/m³, so tail estimates are unstable.],
)
#v(8pt)
#callout(
  "Oral defense in one sentence",
  [“We defined a one-day-ahead PM2.5 regression task, generated seven-day station sequences without crossing time gaps, fit preprocessing within expanding folds, and found no statistically reliable advantage over persistence—especially with weak performance on rare extreme events.”]
)
