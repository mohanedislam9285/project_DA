// ============================================================================
//  GROUP 16 · PM2.5 FORECASTING · SUBMISSION PART 1
//  Data Exploration & Preprocessing — top-level report
//  Build: typst compile part_1_report.typ ..\Part_1_Report.pdf
// ============================================================================
#import "theme.typ": *

#let head-title = "GROUP 16 · PM2.5 FORECASTING · PART 1 · DATA & PREPROCESSING"

#set page(footer: none)
#set text(font: "Arial", size: 9.5pt, fill: ink)
#set par(justify: true, leading: 0.68em, spacing: 0.55em)
#show heading.where(level: 1): set text(size: 18pt, weight: "bold", fill: navy)
#show heading.where(level: 2): set text(size: 11.5pt, weight: "bold", fill: teal)
#show heading.where(level: 3): set text(size: 9.6pt, weight: "bold", fill: navy2)
#show heading: set block(above: 1.05em, below: 0.45em)

// ---------------------------------------------------------------- cover ----
#v(14mm)
#eyebrow("SUBMISSION PART 1 OF 3   ·   GROUP 16")
#v(9mm)
#text(size: 31pt, weight: "bold", fill: navy)[
  Data Exploration &
  #linebreak()
  Preprocessing
]
#v(4mm)
#text(size: 15pt, fill: teal)[Audit, cleaning, EDA and leakage-safe preparation of a 40,983-record PM2.5 station panel]
#v(8mm)
#line(length: 40mm, stroke: 2.5pt + teal)
#v(7mm)
#text(size: 11pt, fill: ink)[
  This part documents where the data come from, what was removed and why, how the
  station-day panel behaves over time and space, and exactly how the modelling inputs
  are constructed so that no future information leaks into a forecast.
]
#v(13mm)
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  stat("40,983", "clean station-day records"),
  stat("95", "coordinate-defined stations"),
  stat("2,299", "distinct UTC daily dates"),
  stat("9", "candidate environmental variables"),
)
#v(11mm)
#callout(
  "Integrity note for the reader and examiner",
  [Redistribution rights for the compiled Kaggle bundle are *unresolved*, and the publisher's full
   upstream row-inclusion rule cannot be reconstructed. Meteorology, trace gases and AOD are retained
   in the audit but *excluded from the model inputs* because their availability at the forecast origin
   is not documented. Both points are treated as first-class limitations, not footnotes.],
  accent: amber,
  tint: sand,
)
#v(1fr)
#text(size: 8.5pt, fill: muted)[Group 16  ·  Part 1 of 3  ·  Data Exploration & Preprocessing  ·  deadline 14 November 2026]
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
  "How Part 1 connects to the rest of the project",
  [Part 1 fixes the *data contract*: the cleaned panel, the station-identity rule, the target bounds,
   the seven-day past-only sequence definition, and the fold-local preprocessing rule. Part 2 consumes
   exactly this contract to train, validate and test the CNN–LSTM; Part 3 consumes the resulting
   out-of-fold predictions for statistical testing and explainability. Every count and statistic used
   downstream originates in the audit files cited here.],
)
#pagebreak()

// ------------------------------------------------------- 1. objective -----
#eyebrow("01  /  OBJECTIVE")
= Objective and scope

The project forecasts the next-day PM2.5 concentration at a monitoring station from observations
available at the end of the current day. PM2.5 is fine particulate matter; the forecast target is a
*continuous concentration in µg/m³*, so the task is regression, not classification. Part 1 does not
model anything: it establishes that the data are trustworthy enough to model and that the inputs are
constructed without leakage.

#heading(level: 2)[What Part 1 must deliver]
#ticks(
  [A verified dataset description: source, licence status, variable units, dimensions and coverage.],
  [A documented cleaning policy: exactly which records were removed, on what rule, and what was deliberately *kept* despite being unusual.],
  [A quantitative EDA: target distribution, seasonality, monthly structure, correlations, missingness and time-series behaviour.],
  [A missing-value strategy fitted inside each training fold rather than on the whole dataset.],
  [A precise feature definition: seven past-only daily steps per station, with no gap crossing and no future information.],
  [A leakage-control statement that downstream parts can rely on.],
)

#heading(level: 2)[Research questions anchored in this part]
#tbl(
  (0.55fr, 4.45fr),
  table.header(th("RQ"), th("Question owned or informed by Part 1")),
  [RQ1], [Is the cleaned panel complete and consistent enough for matched-fold comparison against simple baselines?],
  [RQ3], [Which *available* feature groups carry signal, and which are excluded because they are unavailable at forecast time?],
  [RQ4], [How do seasonality, sparse coverage, extreme events, missingness and data rights constrain trustworthy use?],
)
#v(4pt)
#caption([Parts 2 and 3 answer RQ1–RQ4 with fold-level evidence; Part 1 defines what "available" and "clean" mean.])

#callout(
  "Framing decision",
  [The project is deliberately scoped to *one learned model family only* — CNN–LSTM — because that is
   Group 16's assigned architecture. Part 1 prepares a single, reproducible input contract for it.],
)
#pagebreak()

// ----------------------------------------------------- 2. dataset info ----
#eyebrow("02  /  DATASET AND PROVENANCE")
= Dataset, licence and coverage

The active input is the publisher-prepared *Air Quality – Mediterranean Basin (2016–2026)* bundle
(version 1), obtained from Kaggle and stored locally as `data/raw/extracted/combined_gee_pm25.csv`.
Retrieval metadata, source checksums and the live Kaggle metadata snapshot are retained in
`data/manifest.yaml` and `data/provenance/`.

#heading(level: 2)[Dimensions and coverage]
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  stat("41,718", "raw combined rows", accent: navy2),
  stat("40,983", "rows after cleaning", accent: teal),
  stat("95", "stations (lat/lon unique)", accent: teal),
  stat("0", "duplicate station-day keys", accent: green),
)
#v(6pt)
#tbl(
  (1.25fr, 1.55fr, 1fr, 1.6fr),
  table.header(th("Attribute"), th("Value"), th("Unit"), th("Source of record")),
  [Source], [Kaggle · Sarah Ameziane, v1], [—], [`data/manifest.yaml`],
  [Temporal span], [2016-11-25 → 2026-04-06], [UTC day], [audit + EDA],
  [Spatial span], [lat 36.13–43.53 · lon −5.66–9.14], [degrees], [EDA station summary],
  [Target], [PM2.5 concentration], [µg/m³], [`Exploratory_Data.csv`],
  [Target bounds], [retain 0 < PM2.5 < 500], [µg/m³], [teacher cleaning rule],
  [Rows removed], [735 sentinel / out-of-rule], [rows], [audit],
)

#heading(level: 2)[Variable inventory and missingness]
#tbl(
  (1.35fr, 1.15fr, 1fr, 0.85fr, 1.35fr),
  table.header(th("Variable"), th("Family"), th("Unit"), th("Missing"), th("Role in the project")),
  [pm25], [Target], [µg/m³], [0.00%], [Forecast target],
  [temperature_celsius], [Meteorological], [°C], [0.00%], [Excluded — timing unverified],
  [pressure_mb], [Meteorological], [hPa], [0.00%], [Excluded — timing unverified],
  [wind_u / wind_v], [Meteorological], [m/s], [0.00%], [Excluded — timing unverified],
  [no2], [Trace gas (GEE)], [mol/m²], [5.87%], [Excluded — timing unverified],
  [co], [Trace gas (GEE)], [mol/m²], [5.87%], [Excluded — timing unverified],
  [o3], [Trace gas (GEE)], [mol/m²], [5.97%], [Excluded — timing unverified],
  [aod], [Satellite (GEE)], [dimensionless], [0.00%], [Excluded — timing unverified],
)
#v(4pt)
#caption([Only the PM2.5 target and calendar terms enter the model. Environmental columns are audited for completeness but withheld because forecast-origin availability is undocumented.])

#heading(level: 2)[Licence and provenance status]
#callout(
  "Unresolved: redistribution rights",
  [Kaggle labels the licence *"Other (specified in description)"*. The description points to OpenAQ,
   Copernicus and NASA sources but provides no compiled-bundle licence and no field-to-upstream
   attribution map. The bundle is therefore *not* presented as cleared for redistribution, and the
   original ETL is not recoverable. Sources are cited so a supervisor can reproduce the retrieval.],
  accent: amber,
  tint: sand,
)
#pagebreak()

// ------------------------------------------------------ 3. quality audit --
#eyebrow("03  /  DATA QUALITY AUDIT")
= Cleaning and quality controls

#heading(level: 2)[The cleaning rule, in one sentence]
Only rows whose PM2.5 value violates the teacher-specified physical rule were removed: records with
PM2.5 ≤ 0 or PM2.5 ≥ 500 µg/m³ (735 rows). *No row was deleted for being statistically unusual* —
extreme but physically plausible concentrations are retained, because statistical extremeness alone
cannot establish an instrument fault.

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  stat("735", "rows removed (0 < y < 500)", accent: amber),
  stat("727 / 8", "low / high sentinel removals", accent: amber),
  stat("0", "out-of-rule values post-clean", accent: green),
  stat("0", "invalid coordinates", accent: green),
)
#v(6pt)

#heading(level: 2)[Checks performed and their outcomes]
#tbl(
  (1.9fr, 1.15fr, 2.45fr),
  table.header(th("Check"), th("Result"), th("Interpretation")),
  [Duplicate station-day keys], [0], [Each (station, date) pair is unique — the panel is a clean keyed table.],
  [Out-of-rule / non-finite targets], [0], [After cleaning, every retained target is inside the physical rule.],
  [Coordinate validity], [0 invalid], [All 95 stations carry finite latitude and longitude.],
  [Station identity reconciliation], [98 → 95], [Three test entries removed by target cleaning; the identity logic is preserved.],
  [Station-relative Tukey screen], [352 flagged], [Q3 + 3×IQR flags on 71 stations; flags are *retained for review*, never auto-deleted.],
  [Development high-value counts], [90 / 36 / 11], [Values > 50 / > 100 / > 250 µg/m³ kept in the development partition.],
)
#v(4pt)
#caption([Development-scope audit covers 30,651 rows and 73 stations up to 2024-12-07. See `results/16/cleaning_quality_audit/` and `results/16/audit.json`.])

#heading(level: 2)[Station identity: why the count moves from 98 to 95]
A station ID is built from the source sensor name plus latitude and longitude rounded to six decimals.
This keeps two separately located instruments apart even when a sensor name repeats. The audit
reconciles 98 numeric source identifiers to 98 coordinate-defined stations; cleaning removes three
test entries, leaving 95. There are 94 distinct sensor names, and four of them map to two
coordinate-defined stations each.
#callout(
  "Why this matters for validation",
  [Because a *name* is not a *station*, station-grouped cross-validation in Part 2 must split on the
   composite station ID (name + coordinates). Splitting on the name alone would leak the same instrument
   across train and test and inflate spatial scores.],
)
#pagebreak()

// ---------------------------------------------------------------- 4. EDA --
#eyebrow("04  /  EXPLORATORY DATA ANALYSIS")
= Target distribution

The cleaned PM2.5 target is strongly right-skewed: a long, low body of ordinary days and a thin
tail of pollution episodes. This shape shapes every modelling decision that follows — why absolute
error metrics dominate, why MAPE is unsafe, and why rare high-concentration days are hard to learn.

#align(center)[#image("../figures/pm25_distribution.png", width: 80%)]
#caption([Figure 1. Distribution of cleaned daily PM2.5 (µg/m³) across 40,983 station-day records.])

#tbl(
  (0.8fr, 0.85fr, 0.85fr, 0.85fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr),
  table.header(th("n"), th("Mean"), th("SD"), th("Min"), th("Median"), th("P95"), th("P99"), th("Max")),
  [40,983], [9.458], [9.730], [0.007], [7.851], [21.653], [33.559], [484.98],
)
#v(4pt)
#caption([Summary statistics in µg/m³. Skewness 17.83 and excess kurtosis ≈ 649 confirm a heavy right tail; the maximum is ≈62× the median.])

#heading(level: 2)[What the distribution implies]
#ticks(
  [Half of all station-days sit at or below *7.85 µg/m³*; the problem is mostly about ordinary days.],
  [The top 1% exceeds *33.6 µg/m³* and the maximum reaches *484.98 µg/m³* — roughly two orders of magnitude above the median.],
  [Because targets can approach zero, *MAPE is unstable*; RMSE, MAE and R² are the primary metrics in Parts 2 and 3.],
)
#pagebreak()

// ---------------------------------------------------------- 5. temporal ---
#eyebrow("05  /  TEMPORAL STRUCTURE")
= Seasonality and time trends

#align(center)[#image("../figures/pm25_by_season.png", width: 70%)]
#caption([Figure 2. Seasonal PM2.5 summaries across the cleaned panel (DJF, MAM, JJA, SON).])

#grid(
  columns: (1.05fr, 1fr),
  gutter: 14pt,
  [
    #heading(level: 3)[Seasonal profiles]
    #tbl(
      (0.7fr, 0.85fr, 0.75fr, 0.8fr, 0.85fr, 0.85fr),
      table.header(th("Season"), th("Records"), th("Mean"), th("Med."), th(">25"), th(">50")),
      [DJF], [8,938], [10.94], [8.77], [6.15%], [0.45%],
      [MAM], [9,276], [8.25], [6.83], [1.92%], [0.36%],
      [JJA], [12,766], [9.59], [8.50], [1.95%], [0.15%],
      [SON], [10,003], [9.08], [7.47], [2.46%], [0.25%],
    )
    #v(3pt)
    #caption([Mean/median in µg/m³; percentages are record shares above the cutoffs — descriptive, not regulatory compliance rates.])
  ],
  [
    #heading(level: 3)[Selected monthly means]
    #tbl(
      (0.7fr, 0.85fr, 0.85fr, 0.85fr),
      table.header(th("Month"), th("Mean"), th("Med."), th(">25")),
      [Jan], [10.39], [8.15], [4.85%],
      [Feb], [11.28], [9.25], [6.62%],
      [Mar], [9.49], [7.26], [4.00%],
      [Jun], [9.09], [7.81], [1.90%],
      [Nov], [10.17], [7.91], [3.59%],
      [Dec], [11.15], [8.91], [7.05%],
    )
    #v(3pt)
    #caption([Winter months (Nov–Feb) carry the highest means and the largest share of high days; full table in `eda/monthly_trend.csv`.])
  ],
)

#heading(level: 2)[Coverage is sparse and irregular — the single most important data fact]
The panel is *not* a dense grid. Dates and stations are present unevenly, which is why Part 1 measures
behaviour on uninterrupted segments instead of pretending every calendar day is an observation.
#tbl(
  (2.4fr, 1.1fr, 2.0fr),
  table.header(th("Property"), th("Value"), th("Consequence")),
  [ACF/PACF availability], [63 stations], [Stations with ≥ 14 consecutive daily development observations.],
  [Formal ADF / KPSS / Ljung-Box], [6 stations], [Only stations with ≥ 30 consecutive days can be tested; low-coverage diagnostics.],
  [Lead–lag pollutant analysis], [2–6 stations per lag], [Descriptive only; no causal or availability claim.],
)
#v(4pt)
#caption([Development partition only. Sequences in the model never bridge a date gap, so irregular coverage is handled by construction.])
#pagebreak()

// ------------------------------------------- 6. relationships/missingness --
#eyebrow("06  /  RELATIONSHIPS AND MISSINGNESS")
= Correlations and missing values

#heading(level: 2)[Association with the target]
All linear and rank associations between PM2.5 and the environmental variables are *weak*. The
strongest is the rank correlation with NO₂ (Spearman ρ = 0.247); no pair reaches |r| > 0.9, so there
is no multicollinearity red flag, and no environmental variable is a strong same-day predictor here.

#tbl(
  (1.3fr, 1.1fr, 1.1fr, 1.1fr, 1.4fr),
  table.header(th("Variable"), th("Pearson r"), th("Spearman ρ"), th("|r| > 0.9?"), th("Reading")),
  [no2], [0.159], [0.247], [No], [Strongest, but still weak in absolute terms],
  [aod], [0.094], [0.180], [No], [Weak positive association],
  [pressure_mb], [0.113], [0.167], [No], [Weak positive association],
  [wind_u], [−0.079], [−0.164], [No], [Weak negative association],
  [o3], [−0.087], [−0.135], [No], [Weak negative association],
  [co], [0.041], [0.036], [No], [Essentially none],
  [temperature_celsius], [−0.006], [0.064], [No], [Essentially none],
)
#v(4pt)
#caption([Computed on the cleaned panel (`results/16/correlations.csv`). Because availability at forecast time is undocumented, these associations are reported as *context*, not as model inputs.])

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [#image("../figures/spearman_correlations.png", width: 100%)],
  [#image("../figures/missingness.png", width: 100%)],
)
#caption([Figure 3 (left). Spearman rank-correlation structure. Figure 4 (right). Missingness profile of the cleaned table.])

#heading(level: 2)[Missing-value profile]
#tbl(
  (1.5fr, 1fr, 1fr, 2fr),
  table.header(th("Column"), th("Missing"), th("Share"), th("Handling decision")),
  [no2], [2,407], [5.87%], [Not modelled; audited only],
  [co], [2,407], [5.87%], [Not modelled; audited only],
  [o3], [2,448], [5.97%], [Not modelled; audited only],
  [pm25 and calendar], [0], [0.00%], [Core modelling columns — no missingness],
)
#v(4pt)
#callout(
  "Missingness is not imputed globally",
  [There is no dataset-wide fill. Any imputation happens *inside a training fold*, so the value used to
   fill a gap can never be computed from the validation or test period. This is a leakage control, not a
   convenience (Section 8).],
)
#pagebreak()

// -------------------------------------------------- 7. missing value plan --
#eyebrow("07  /  MISSING VALUES")
= Missing-value strategy

Because the *model* columns (PM2.5 target and calendar terms) are complete, the missing-value problem
is small and confined to derived statistics. The strategy is therefore deliberately conservative.

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 9pt,
  stat("Median", "imputation statistic", accent: teal, sub: "robust to the heavy right tail"),
  stat("Fold-local", "where it is fitted", accent: teal, sub: "training partition only"),
  stat("+ flags", "indicator channels", accent: teal, sub: "record where a gap was filled"),
)
#v(7pt)

#heading(level: 2)[Rules]
#ticks(
  [*Median imputation* is used rather than the mean, because the target is heavily skewed and a mean fill would be dragged by episodes.],
  [The imputer and the scaler are *fitted on each training fold's sequences* and then applied to that fold's validation sequences — never fitted on the full panel.],
  [*Missingness indicators* are added for derived PM2.5 rolling summaries so the network can distinguish a real low value from a filled one.],
  [Standardisation (zero mean, unit variance) uses the same fold-local fit; the scalar output head is trained on the standardised target and predictions are inverse-transformed for scoring.],
  [Reading the target scale: all reported metrics are computed *after* inverse transformation, i.e. on the original µg/m³ scale.],
)

#callout(
  "Why the reference preprocessing is frozen here",
  [The target is kept on its original *µg/m³* scale, both rolling missingness indicators are retained,
   and imputation and scaling are fitted fold-locally. These choices are declared as the reference data
   contract so that every later stage consumes identical inputs. Their justification is leakage-safety
   and the strong right skew of the target — not model scores, which are outside Part 1's scope.],
)
#pagebreak()

// ------------------------------------------------------ 8. feature design --
#eyebrow("08  /  FEATURE ENGINEERING")
= Feature construction

#heading(level: 2)[The sequence contract]
For station $s$ and forecast origin day $t$, one training example is a *seven-day* window of daily
rows for that station, ending at $t$, paired with the single target value at $t+1$.

#grid(
  columns: (1fr, auto, 1.15fr, auto, 1fr),
  gutter: 3pt,
  node("7 past days", "[t−6 … t]\nindexed per station", tone: ice),
  text(size: 16pt, fill: teal)[→],
  node("Past-only features", "calendar cycles\nPM2.5 lag & rolling", tone: pale),
  text(size: 16pt, fill: teal)[→],
  node("Target", "PM2.5 at t+1\n(scalar)", tone: sand),
)
#v(7pt)

#heading(level: 2)[Feature channels]
#tbl(
  (1.5fr, 1.1fr, 2.5fr),
  table.header(th("Channel"), th("Type"), th("Definition and why it is safe")),
  [pm25_lag1], [Past value], [PM2.5 observed at the forecast origin $t$ — the persistence signal, and the strongest known predictor.],
  [pm25_roll3_mean], [Past rolling], [Mean of the three days ending at $t$; uses only observed values ≤ $t$.],
  [pm25_roll7_mean], [Past rolling], [Mean of the seven days ending at $t$; captures the recent level.],
  [day_of_year_sin/cos], [Calendar], [Annual cycle encoded as a smooth pair; known in advance, so it cannot leak.],
  [day_of_week_sin/cos], [Calendar], [Weekly cycle encoded as a smooth pair; known in advance.],
  [rolling missing flags], [Indicator], [Declare where a rolling summary was filled, so filled and real values are distinguishable.],
)
#v(4pt)
#caption([Nine processed channels in the reference model: seven features plus two rolling missingness flags. The channel set is fixed here as the reference contract consumed by all later stages.])

#heading(level: 2)[Three hard rules that prevent leakage]
#ticks(
  [*No gap crossing.* A sequence is only built from consecutive daily rows; a missing date breaks the window instead of being skipped silently.],
  [*No station crossing.* A window never spans two stations, so one instrument's history cannot inform another's forecast.],
  [*No future information.* Every feature is a function of data at or before $t$; the label is a single value at $t+1$. Automated tests mutate values at and after a cutoff and confirm earlier features do not change.],
)
#pagebreak()

// --------------------------------------------------- 9. pipeline/leakage --
#eyebrow("09  /  PIPELINE AND LEAKAGE CONTROL")
= Preprocessing pipeline

#grid(
  columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr),
  gutter: 3pt,
  node("Raw CSV", "41,718 prepared rows", tone: pale),
  text(size: 15pt, fill: teal)[→],
  node("Cleaning", "target bounds +\nkey checks", tone: pale),
  text(size: 15pt, fill: teal)[→],
  node("Split", "date/station-aware\nfolds (Part 2)", tone: ice),
  text(size: 15pt, fill: teal)[→],
  node("Fold-local", "impute + scale\non training only", tone: ice),
  text(size: 15pt, fill: teal)[→],
  node("Sequences", "7 days, no gaps", tone: sand),
)
#v(8pt)

#heading(level: 2)[Leakage-control checklist]
#tbl(
  (1.9fr, 1.2fr, 2.4fr),
  table.header(th("Control"), th("Status"), th("Evidence")),
  [Fold-local imputation and scaling], [Pass], [Fitted on the training partition of each fold only.],
  [Past-only rolling features], [Pass], [Recomputed from values ≤ the forecast origin.],
  [Origin/target alignment], [Pass], [Each label is exactly the next station-day; verified by automated tests.],
  [Gap and station boundaries], [Pass], [Sequence and split unit tests reject cross-station or gapped windows.],
  [Environmental timing], [Blocked], [Exogenous channels withheld rather than assumed to be available at issue time.],
  [Frozen holdout untouched], [Pass], [The reserved future block is excluded from all feature and model selection.],
)
#v(5pt)
#callout(
  "Verification boundary",
  [The checks support leakage control in *the executed pipeline*. They do not prove that the publisher's
   original upstream join was leakage-free, nor reconstruct the complete row-inclusion process. That
   residual uncertainty is disclosed rather than hidden.],
  accent: amber,
  tint: sand,
)
#pagebreak()

// ----------------------------------------------- 10. limitations/handoff --
#eyebrow("10  /  LIMITATIONS AND HANDOFF")
= Limitations and what Part 2 receives

#heading(level: 2)[Material limitations carried forward]
#tbl(
  (1.5fr, 4.5fr),
  table.header(th("Area"), th("Limitation carried into Parts 2 and 3")),
  [Licence], [Compiled-bundle redistribution rights unresolved; no unrestricted-redistribution claim is made.],
  [Upstream ETL], [All 41,718 bundled rows map to source PM/GEE rows, but the publisher's omitted-row rule and tie handling remain unreconstructed.],
  [Coverage], [Sparse, irregular station-days limit formal temporal tests to six stations and make lead–lag analysis descriptive.],
  [Environmental inputs], [Meteorology, NO₂, CO, O₃ and AOD are withheld because forecast-origin availability is undocumented.],
  [Tail behaviour], [Only a handful of development observations exceed 100 µg/m³, so high-concentration performance is inherently unstable.],
)

#heading(level: 2)[The frozen handoff to Part 2]
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 9pt,
  stat("40,983 × 9", "cleaned panel\ntarget + 8 audit variables", accent: navy2),
  stat("7 days", "past-only sequence length", accent: teal),
  stat("0 < y < 500", "target rule (µg/m³)", accent: teal),
)
#v(8pt)
#callout(
  "Part 1 verdict",
  [The panel is *clean enough to model and documented well enough to audit*. It is not a dense grid and
   its licence is unresolved, so Part 2 must (a) validate with expanding temporal folds and station-grouped
   folds, (b) keep preprocessing fold-local, and (c) report uncertainty honestly rather than a single number.],
)
#v(6pt)
#caption([All Part 1 statistics are reproducible from `results/16/audit.json`, `results/16/Exploratory_Data.csv`, `results/16/Exploratory_temporal.csv`, `results/16/eda/`, `results/16/correlations.csv` and `results/16/cleaning_quality_audit/`.])
