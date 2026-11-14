// ============================================================================
//  GROUP 16 · PM2.5 FORECASTING · SUBMISSION PART 1
//  Data Exploration & Preprocessing — top-level report
//  Build: typst compile --root . reports\part_1\source\part_1_report.typ reports\part_1\Part_1_Report.pdf
// ============================================================================
#import "theme.typ": *

#set document(
  title: "Part 1 - Data Exploration and Preprocessing",
  author: "Group 16",
  keywords: ("PM2.5", "data auditing", "exploratory analysis", "preprocessing", "CNN-LSTM"),
)

#let head-title = "GROUP 16 · PM2.5 FORECASTING · PART 1 · DATA & PREPROCESSING"

// ---- academic typography ----------------------------------------------------
#set page(footer: none)
#set text(font: "Cambria", size: 10pt, fill: ink, hyphenate: true)
#set par(justify: true, leading: 0.72em, spacing: 0.62em)

#show heading.where(level: 1): it => block(
  above: 1.35em,
  below: 0.6em,
  sticky: true,
)[
  #text(font: "Calibri", size: 19pt, weight: "bold", fill: navy)[#it.body]
  #v(2pt)
  #line(length: 24mm, stroke: 2pt + teal)
]
#show heading.where(level: 2): it => block(
  above: 1.05em,
  below: 0.35em,
  sticky: true,
)[
  #text(font: "Calibri", size: 11.4pt, weight: "bold", fill: teal)[#it.body]
]
#show heading.where(level: 3): it => block(
  above: 0.85em,
  below: 0.3em,
)[
  #text(font: "Calibri", size: 9.8pt, weight: "bold", fill: navy2)[#it.body]
]

// Tables use a compact sans face and no hyphenation, so narrow columns never
// split technical terms mid-word.
#show table: set text(font: "Calibri", size: 8.2pt, hyphenate: false)

// Long identifiers are wrapped at underscore boundaries, so artifact paths and
// column names stay inside narrow table cells instead of overflowing the margin.
#let codeid(s, size: 7.6pt, max-chars: 26) = {
  let parts = s.split("_")
  let lines = ()
  let cur = ""
  for (i, p) in parts.enumerate() {
    let piece = if i < parts.len() - 1 { p + "_" } else { p }
    if cur == "" {
      cur = piece
    } else if cur.len() + piece.len() > max-chars {
      lines.push(cur)
      cur = piece
    } else {
      cur = cur + piece
    }
  }
  if cur != "" { lines.push(cur) }
  for (i, l) in lines.enumerate() {
    if i > 0 { linebreak() }
    text(font: "DejaVu Sans Mono", size: size)[#l]
  }
}

#let keep(body) = block(breakable: false, width: 100%)[#body]

// ---------------------------------------------------------------- cover ----
#v(0.85fr)
#eyebrow("SUBMISSION PART 1 OF 3   ·   GROUP 16")
#v(9mm)
#text(size: 31pt, weight: "bold", fill: navy)[
  Data Exploration &
  #linebreak()
  Preprocessing
]
#v(4mm)
#text(size: 14.5pt, fill: teal)[
  Data auditing, cleaning, exploratory analysis and leakage-safe preprocessing
  of a PM2.5 station panel.
]
#v(8mm)
#line(length: 40mm, stroke: 2.5pt + teal)
#v(7mm)
#text(size: 10.8pt, fill: ink)[
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
  stat("8", "environmental variables", accent: navy2, sub: "+ 1 forecast target"),
)
#v(11mm)
#callout(
  "Integrity note",
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
  margin: (x: 20mm, top: 18mm, bottom: 18mm),
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

#heading(level: 1, outlined: false)[Contents]
#v(2pt)
#show outline.entry: set block(above: 0.42em, below: 0.42em)
#outline(title: none, depth: 1, indent: auto)

#v(2pt)

// ------------------------------------------------------- 1. objective -----
= Objective and scope
#block[
The project forecasts the next-day PM2.5 concentration at a monitoring station from observations
available at the end of the current day. PM2.5 is fine particulate matter; the forecast target is a
*continuous concentration in µg/m³*, so the task is regression, not classification. Part 1 does not
model anything: it establishes that the data are trustworthy enough to model and that the inputs are
constructed without leakage.
]

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
#keep[
#tbl(
  (0.55fr, 4.45fr),
  table.header(th("RQ"), th("Question owned or informed by Part 1")),
  [RQ1], [Is the cleaned panel complete and consistent enough for matched-fold comparison against simple baselines?],
  [RQ3], [Which *available* feature groups carry signal, and which are excluded because they are unavailable at forecast time?],
  [RQ4], [How do seasonality, sparse coverage, extreme events, missingness and data rights constrain trustworthy use?],
)
]

#callout(
  "Framing decision",
  [The modelling stage uses the CNN–LSTM architecture assigned to Group 16. Part 1 prepares a single,
   reproducible input contract for it.],
)

#v(2pt)

// ----------------------------------------------------- 2. dataset info ----
= Dataset, licence and coverage
#block[
The active input is the publisher-prepared *Air Quality – Mediterranean Basin (2016–2026)* bundle
(version 1), obtained from Kaggle and stored locally as `data/raw/extracted/combined_gee_pm25.csv`.
Retrieval metadata, source checksums and the live Kaggle metadata snapshot are retained in
`data/manifest.yaml` and `data/provenance/`.
]

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
#keep[
#tbl(
  (1.25fr, 1.55fr, 1fr, 1.6fr),
  table.header(th("Attribute"), th("Value"), th("Unit"), th("Source of record")),
  [Source], [Kaggle · Sarah Ameziane, v1], [—], [`data/manifest.yaml`],
  [Temporal span], [2016-11-25 → 2026-04-06], [UTC day], [audit + EDA],
  [Spatial span], [lat 36.13–43.53 · lon −5.66–9.14], [degrees], [EDA station summary],
  [Target], [PM2.5 concentration], [µg/m³], [`Exploratory_Data.csv`],
  [Target bounds], [retain 0 < PM2.5 < 500], [µg/m³], [cleaning rule],
  [Rows removed], [735 sentinel / out-of-rule], [rows], [audit],
)
]

#heading(level: 2)[Source-to-bundle matching audit]
Because the publisher's extraction code is unavailable, the retained bundle was cross-checked against
its two source tables. The check covers the rows that were kept; it does not reconstruct the publisher's
row-inclusion rule.
#keep[
#tbl(
  (3.7fr, 1.5fr),
  table.header(th("Provenance check"), th("Result")),
  [Original PM2.5 source rows], [88,041],
  [Original GEE source rows], [164,208],
  [Combined bundle rows], [41,718],
  [Combined rows matched to PM2.5 source], [41,718],
  [Combined rows matched to GEE source], [41,718],
  [Combined rows matched to both sources], [41,718],
  [Maximum PM2.5 matching difference], [3.6×10#super[-15] µg/m³],
  [Combined rows with exact GEE coordinate match], [41,718],
  [PM2.5 source rows not included in bundle], [46,323],
)
]
#v(4pt)
#caption([Recomputed from `results/provenance_audit/upstream_join_audit_summary.json`. Matching uses the source PM2.5 key (sensor name, date, coordinates) and the GEE key (ID, date).])
#v(3pt)
Every retained bundle row corresponds to an exact PM2.5 source record and an exact GEE ID/date record;
the matched PM2.5 targets agree to numerical precision and the GEE coordinates coincide. The audit
does not reconstruct the publisher's original extraction procedure or explain why 46,323 PM2.5 source
rows were omitted.

#v(2pt)

#heading(level: 2)[Variable inventory and missingness]
#keep[
#tbl(
  (1.3fr, 1.05fr, 0.9fr, 0.75fr, 0.9fr),
  table.header(th("Variable"), th("Family"), th("Unit"), th("Missing"), th("Role")),
  [pm25], [Target], [µg/m³], [0.00%], [Target],
  [temperature_celsius], [Meteorology], [°C], [0.00%], [Excluded],
  [pressure_mb], [Meteorology], [hPa], [0.00%], [Excluded],
  [wind_u / wind_v], [Meteorology], [m/s], [0.00%], [Excluded],
  [no2], [Trace gas], [mol/m²], [5.87%], [Excluded],
  [co], [Trace gas], [mol/m²], [5.87%], [Excluded],
  [o3], [Trace gas], [mol/m²], [5.97%], [Excluded],
  [aod], [Satellite], [dimensionless], [0.00%], [Excluded],
)
]
#v(3pt)
#caption([Column names follow the CSV. All eight environmental variables are excluded because their availability at the forecast origin has not been verified.])
#v(5pt)

#heading(level: 2)[Licence and provenance status]
#callout(
  "Unresolved: redistribution rights",
  [Kaggle labels the licence *"Other (specified in description)"*. The description points to OpenAQ,
   Copernicus and NASA sources but provides no compiled-bundle licence and no field-to-upstream
   attribution map. The bundle is therefore *not* presented as cleared for redistribution, and the
   original ETL is not recoverable. The source references support retrieval and provenance verification.],
  accent: amber,
  tint: sand,
)

#v(2pt)

// ------------------------------------------------------ 3. quality audit --
= Cleaning and quality controls

#heading(level: 2)[Cleaning rule]
#block[
Only rows whose PM2.5 value violates the specified physical rule were removed: records with
PM2.5 ≤ 0 or PM2.5 ≥ 500 µg/m³ (735 rows). *No row was deleted for being statistically unusual* —
extreme but physically plausible concentrations are retained, because statistical extremeness alone
cannot establish an instrument fault.
]

#grid(
  columns: (1fr, auto, 1.25fr, auto, 1fr, auto, 1.25fr),
  gutter: 3pt,
  node("Raw combined", "41,718 rows", tone: pale),
  text(size: 14pt, fill: teal)[→],
  node("Target validation", "missing · invalid\nbounds", tone: ice),
  text(size: 14pt, fill: teal)[→],
  node("Removed", "727 ≤ 0\n8 ≥ 500", tone: sand),
  text(size: 14pt, fill: teal)[→],
  node("Clean panel", "40,983 rows\n95 stations", tone: pale),
)
#v(5pt)
#caption([Cleaning flow. Physically valid values — including plausible extremes — pass through; only out-of-rule targets are removed.])
#v(4pt)
Retaining plausible extremes is deliberate. A high concentration is not automatically a measurement
error, and deleting unusual values would bias the target toward ordinary days and remove the episodes
the model is meant to learn. The station-relative Tukey screen identified 352 potentially unusual
values across 71 stations. These observations were retained and flagged for review rather than
automatically removed.

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
#keep[
#tbl(
  (1.9fr, 1.15fr, 2.45fr),
  table.header(th("Check"), th("Result"), th("Interpretation")),
  [Duplicate station-day keys], [0], [Each (station, date) pair is unique — the panel is a clean keyed table.],
  [Out-of-rule / non-finite targets], [0], [After cleaning, every retained target is inside the physical rule.],
  [Coordinate validity], [0 invalid], [All 95 stations carry finite latitude and longitude.],
  [Station identity reconciliation], [98 → 95], [Three station identities (nine zero-target rows) removed by the target rule; the identity logic is preserved.],
  [Station-relative Tukey screen], [352 flagged], [Q3 + 3×IQR flags on 71 stations; flags are *retained for review*, never auto-deleted.],
  [Development high-value counts], [90 / 36 / 11], [Values > 50 / > 100 / > 250 µg/m³ kept in the development partition.],
)
]
#v(4pt)
#caption([Development-scope audit covers 30,651 rows and 73 stations up to 2024-12-07. See `data/processed/audit.json` and `results/cleaning_quality_audit/cleaning_quality_audit.json`.])

#heading(level: 2)[Station identity reconciliation (98 to 95 stations)]
#block[
A station ID is built from the source sensor name plus latitude and longitude rounded to six decimals.
This keeps two separately located instruments apart even when a sensor name repeats. The audit
reconciles 98 numeric source identifiers to 98 coordinate-defined stations; cleaning removes three
station identities whose nine rows all have a zero target, leaving 95. The source contains 94 distinct
sensor names; after cleaning, 91 names remain represented. Four source names map to two
coordinate-defined stations each, which is why the station count is not the sensor-name count.
]
#callout(
  "Importance of composite station identifiers",
  [Because a *name* is not a *station*, station-grouped cross-validation in Part 2 must split on the
   composite station ID (name + coordinates). Splitting on the name alone would leak the same instrument
   across train and test and inflate spatial scores.],
)

#v(2pt)

// ---------------------------------------------------------------- 4. EDA --
= Target distribution
#block[
The cleaned PM2.5 target is strongly right-skewed: a long, low body of ordinary days and a thin
tail of pollution episodes. This distribution influences subsequent modelling decisions, particularly
metric selection and the treatment of rare high-concentration events.
]

#align(center)[#image("../../../figures/pm25_distribution.png", width: 76%)]
#caption([Figure 1. Distribution of cleaned daily PM2.5 (µg/m³) across 40,983 station-day records.])

#align(center)[#image("../../../figures/pm25_boxplot.png", width: 86%)]
#caption([Figure 2. Boxplot of the same target on linear and log scales. The box is compressed on the linear scale because the tail extends to 484.98 µg/m³.])

#keep[
#tbl(
  (0.8fr, 0.85fr, 0.85fr, 0.85fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr),
  table.header(th("n"), th("Mean"), th("SD"), th("Min"), th("Median"), th("P95"), th("P99"), th("Max")),
  [40,983], [9.458], [9.730], [0.007], [7.851], [21.653], [33.559], [484.98],
)
]
#v(4pt)
#caption([Summary statistics in µg/m³. Skewness 17.83 and excess kurtosis 651.67 confirm a heavy right tail; the maximum is ≈62× the median.])

#heading(level: 2)[How often the target exceeds the cutoffs]
#grid(
  columns: (1.15fr, 1fr),
  gutter: 14pt,
  [
    #tbl(
      (1.15fr, 1fr, 0.85fr),
      table.header(th("Threshold"), th("Records above"), th("Share")),
      [> 25 µg/m³], [1,223], [2.98%],
      [> 50 µg/m³], [117], [0.29%],
    )
    #v(3pt)
    #caption([Counted over all 40,983 cleaned records, not from seasonal averages. A further 37 records (0.09%) exceed 100 µg/m³.])
  ],
  [
    #heading(level: 3)[What the distribution implies]
    #ticks(
      [Half of all station-days sit at or below *7.85 µg/m³*; the problem is mostly about ordinary days.],
      [The top 1% exceeds *33.6 µg/m³* and the maximum reaches *484.98 µg/m³* — roughly two orders of magnitude above the median.],
      [Because targets can approach zero, *MAPE is unstable*; RMSE, MAE and R² are the primary metrics in Parts 2 and 3.],
    )
  ],
)

#v(2pt)

// ---------------------------------------------------------- 5. temporal ---
= Seasonality and time trends

#align(center)[#image("../../../figures/pm25_by_season.png", width: 68%)]
#caption([Figure 3. Seasonal PM2.5 summaries across the cleaned panel (DJF, MAM, JJA, SON).])

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
    #caption([Higher PM2.5 levels are observed during winter months, particularly November to February; full table in `eda/monthly_trend.csv`.])
  ],
)

#heading(level: 2)[Where the stations are]
#grid(
  columns: (0.85fr, 1.15fr),
  gutter: 10pt,
  [#image("../../../figures/station_map.png", width: 100%)],
  [#image("../../../figures/station_coverage_development.png", width: 100%)],
)
#caption([Figure 4 (left). Locations of the 95 coordinate-defined stations in the cleaned panel. Figure 5 (right). Development coverage per station, with the longest consecutive daily observation period highlighted.])

#heading(level: 2)[Sparse and irregular temporal coverage]
#block[
The dataset contains irregular station-level observations rather than a complete daily panel. Temporal
diagnostics are therefore restricted to uninterrupted daily segments.
]
#keep[
#tbl(
  (2.3fr, 1.0fr, 2.2fr),
  table.header(th("Property"), th("Value"), th("Consequence")),
  [ACF/PACF availability], [63 stations], [≥ 14 consecutive development days.],
  [Formal ADF / KPSS / Ljung-Box], [6 stations], [Only ≥ 30-day runs are testable.],
  [Lead–lag pollutant analysis], [2–6 stations per lag], [Descriptive only; no causal claim.],
)
]
#v(4pt)
#caption([Development partition only. Sequences in the model never bridge a date gap, so irregular coverage is handled by construction.])
#v(4pt)
Across the development partition, the median longest uninterrupted run is *17 days* and the maximum
is *32 days*; 63 stations reach at least 14 consecutive days and only 6 reach 30. Applying conventional
time-series diagnostics to irregularly spaced observations would make the lag interpretation unreliable.
Where ACF or PACF lags are reported, lag k means k daily steps — lag 24 is 24 days, not 24 hours.

#v(2pt)

// ------------------------------------------- 6. relationships/missingness --
= Correlations and missing values

#heading(level: 2)[Association with the target]
#block[
All linear and rank associations between PM2.5 and the environmental variables are weak; the strongest
is the rank correlation with NO₂ (Spearman ρ = 0.247). No pair exceeds |r| = 0.9. This pairwise
correlation screen does not rule out nonlinear or multivariate dependencies, and the observed same-day
correlations should not be interpreted as evidence of predictive performance.
]

#[#show table: set table(inset: (x: 5pt, y: 2.6pt))
#keep[
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
]
#v(4pt)
#caption([Computed on the cleaned panel (`results/correlations.csv`); reported as *context*, not model inputs, because availability at forecast time is undocumented.])]

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  [#image("../../../figures/spearman_correlations.png", width: 100%)],
  [#image("../../../figures/missingness.png", width: 100%)],
)
#caption([Figure 6 (left). Spearman rank-correlation structure. Figure 7 (right). Missingness profile of the cleaned table.])

#align(center)[#image("../../../figures/pollutant_missingness_heatmap_development.png", width: 100%)]
#caption([Figure 8. Percentage of station-days missing by station and month for NO₂, CO and O₃ in the development partition (73 stations, 2016-11 to 2024-12). Gray cells have no station-month record; darker red means a larger share of missing values. Full table: `results/pollutant_missingness_station_month_development.csv`.])
#v(4pt)
Missingness is concentrated rather than uniform. For each gas, 19 of the 73 development stations have
at least one affected station-month; 258 of 2,179 NO₂ station-months, 258 of 2,179 CO station-months,
and 261 of 2,179 O₃ station-months contain at least one missing day. Missingness is therefore
concentrated in a minority of stations and months rather than spread evenly across the dataset.

#heading(level: 2)[Missing-value profile]
#keep[
#tbl(
  (1.5fr, 1fr, 1fr, 2fr),
  table.header(th("Column"), th("Missing"), th("Share"), th("Handling decision")),
  [no2], [2,407], [5.87%], [Not modelled; audited only],
  [co], [2,407], [5.87%], [Not modelled; audited only],
  [o3], [2,448], [5.97%], [Not modelled; audited only],
  [pm25 and calendar], [0], [0.00%], [Core modelling columns — no missingness],
)
]
#v(4pt)
#callout(
  "Missingness is not imputed globally",
  [There is no dataset-wide fill. Any imputation is fitted inside the training fold, so the fill value
   cannot be computed from validation or test data.],
)

#v(2pt)

// -------------------------------------------------- 7. missing value plan --
= Missing-value strategy

Because the original PM2.5 target and calendar terms are complete, the missing-value problem is
confined to derived rolling features. The strategy is therefore deliberately conservative, and the
raw target is never imputed.

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
  [*Median imputation* is applied to missing rolling-feature values because these features are derived from a strongly right-skewed PM2.5 series.],
  [The imputer and the scaler are *fitted on each training fold's sequences* and then applied to that fold's validation sequences — never fitted on the full panel.],
  [*Missingness indicators* are added for derived PM2.5 rolling summaries so the network can distinguish a real low value from a filled one.],
  [*Feature standardisation* (zero mean, unit variance) uses the same fold-local fit. The target remains on its original µg/m³ scale; no target scaling is applied in the reference pipeline.],
  [All reported metrics are computed on the original µg/m³ scale, after any model output is returned to that scale.],
)

#heading(level: 2)[What is missing, and what is not]
#keep[
#tbl(
  (1.6fr, 1.15fr, 2.45fr),
  table.header(th("Column group"), th("Missing?"), th("Handling")),
  [Original PM2.5 target], [No], [Used as the label; never imputed.],
  [Calendar terms], [No], [Computed from the date; complete by construction.],
  [pm25_lag1], [No], [Observed at the forecast origin t.],
  [pm25_roll3_mean], [At segment starts], [First two rows of an uninterrupted segment lack a full three-day window.],
  [pm25_roll7_mean], [At segment starts], [First six rows of an uninterrupted segment lack a full seven-day window.],
)
]
#v(3pt)
#caption([Missingness in the selected model inputs. The imputer learns its fill value and indicator channels from the training fold only.])
#v(5pt)
#grid(
  columns: (1fr, auto, 1fr),
  gutter: 4pt,
  node("Before imputation", "rolling features contain gaps\nat the start of a segment", tone: sand),
  text(size: 14pt, fill: teal)[→],
  node("After imputation", "training-fold median fills gaps\n+ missingness indicator = 1", tone: ice),
)
#v(4pt)
#caption([Schematic of the two rolling-feature indicators. The original target column is not part of this fill operation.])

#callout(
  "Preprocessing configuration for subsequent modelling",
  [The original PM2.5 measurements are preserved in µg/m³. Both rolling missingness indicators are
   retained, median imputation and feature standardisation are fitted only on the training fold, and no
   target scaling is applied. These choices define the reference data contract consumed by later stages.],
)

#v(2pt)

// ------------------------------------------------------ 8. feature design --
= Feature construction

#heading(level: 2)[Sequence definition]
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

#heading(level: 2)[A concrete origin/target alignment]
#keep[
#tbl(
  (0.9fr, 1.5fr, 1.8fr),
  table.header(th("Day"), th("Input PM2.5"), th("Role")),
  [1], [8], [Input],
  [2], [9], [Input],
  [3], [10], [Input],
  [4], [7], [Input],
  [5], [11], [Input],
  [6], [12], [Input],
  [7], [13], [Origin / input],
  [8], [Unknown], [Target to predict],
)
]
#v(3pt)
#caption([Illustrative values only. Here $X = (8, 9, 10, 7, 11, 12, 13)$ and $y = "PM2.5 at day 8"$. The network also receives the calendar and rolling channels listed below, not only the raw values shown here.])
#v(4pt)

#heading(level: 2)[Feature channels]
#keep[
#tbl(
  (1.5fr, 1.1fr, 2.5fr),
  table.header(th("Channel"), th("Type"), th("Definition and why it is safe")),
  [pm25_lag1], [Past value], [PM2.5 observed at the forecast origin $t$ — a persistence feature representing the most recent observed value.],
  [pm25_roll3_mean], [Past rolling], [Mean of the three days ending at $t$; uses only observed values ≤ $t$.],
  [pm25_roll7_mean], [Past rolling], [Mean of the seven days ending at $t$; captures the recent level.],
  [day_of_year_sin/cos], [Calendar], [Annual cycle encoded as a smooth pair; known in advance, so it cannot leak.],
  [day_of_week_sin/cos], [Calendar], [Weekly cycle encoded as a smooth pair; known in advance.],
  [rolling missing flags], [Indicator], [Declare where a rolling summary was filled, so filled and real values are distinguishable.],
)
]
#v(4pt)
#caption([The input tensor contains nine channels: seven engineered features and two rolling-feature missingness indicators. The channel set is fixed here as the reference contract consumed by all later stages.])

#heading(level: 2)[Leakage prevention rules]
#ticks(
  [*No gap crossing.* A sequence is only built from consecutive daily rows; a missing date breaks the window instead of being skipped silently.],
  [*No station crossing.* A window never spans two stations, so one instrument's history cannot inform another's forecast.],
  [*No future information.* Every feature is a function of data at or before $t$; the label is a single value at $t+1$. Automated tests verify that modifying future observations does not affect previously constructed features.],
)

#v(2pt)

// --------------------------------------------------- 9. pipeline/leakage --
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
#keep[
#tbl(
  (1.6fr, 0.85fr, 3.2fr),
  table.header(th("Control"), th("Status"), th("Automated check")),
  [Fold-local imputation and scaling], [Pass], [Automated fold-local preprocessing tests confirm that statistics are fitted on the training fold only.],
  [Past-only rolling features], [Pass], [Future-value mutation tests confirm that earlier lag and rolling features remain unchanged.],
  [Origin/target alignment], [Pass], [Sequence alignment tests confirm that each window ends at the forecast origin and the label is the following day.],
  [Gap and station boundaries], [Pass], [Automated tests verify station isolation, consecutive-date requirements and correct input/target alignment.],
  [Environmental timing], [Blocked], [Exogenous channels are withheld rather than assumed to be available at issue time.],
  [Frozen holdout untouched], [Pass], [Holdout integrity tests confirm that the future test period is the latest chronological block and is excluded from development folds.],
)
]
#v(5pt)
#callout(
  "Executed test evidence",
  [The full suite was run from the project root on 2026-11-14. `python -m pytest src/tests`
   produced *63 passed, 0 failed* across 17 test files in 22.9 s. Tests are located in `src/tests`
   in this repository; the saved log is `reports/reproducibility/pytest_2026-11-14_part1.txt`.],
  accent: green,
  tint: ice,
)
#v(4pt)
#callout(
  "Verification boundary",
  [These tests assess the implemented preprocessing pipeline only. They cannot establish whether the
   publisher's original upstream join was leakage-free or reconstruct its complete row-selection
   procedure.],
  accent: amber,
  tint: sand,
)

#v(2pt)

// ----------------------------------------------- 10. limitations/handoff --
= Limitations and what Part 2 receives

#heading(level: 2)[Material limitations carried forward]
#keep[
#tbl(
  (1.5fr, 4.5fr),
  table.header(th("Area"), th("Limitation carried into Parts 2 and 3")),
  [Licence], [Compiled-bundle redistribution rights unresolved; no unrestricted-redistribution claim is made.],
  [Upstream ETL], [All 41,718 bundled rows map to source PM/GEE rows, but the publisher's omitted-row rule and tie handling remain unreconstructed.],
  [Coverage], [Sparse, irregular station-days limit formal temporal tests to six stations and make lead–lag analysis descriptive.],
  [Environmental inputs], [Meteorology, NO₂, CO, O₃ and AOD are withheld because forecast-origin availability is undocumented.],
  [Tail behaviour], [Only a handful of development observations exceed 100 µg/m³, so high-concentration performance is inherently unstable.],
)
]

#heading(level: 2)[Data specification transferred to Part 2]
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 9pt,
  stat("40,983", "cleaned records", accent: navy2, sub: "19 columns in the cleaned table"),
  stat("9", "model input channels", accent: teal, sub: "7 features + 2 missingness flags"),
  stat("0 < y < 500", "target rule (µg/m³)", accent: teal),
)
#v(8pt)

The cleaned dataset is prepared for subsequent modelling, subject to the limitations listed above.
Part 2 should validate with expanding temporal folds and station-grouped folds, keep preprocessing
fold-local, and report performance across validation folds together with appropriate uncertainty
estimates.
#v(4pt)
#caption([The fold definitions and the frozen future block are committed JSON artifacts. Appendix D lists every reproducibility anchor.])

#v(2pt)

// ------------------------------------------------------------- appendices --
#pagebreak(weak: true)
= Appendix A — Predictor distributions

Distributional analysis was performed for all environmental variables using histograms and boxplots.
#v(4pt)
#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [#image("../../../figures/distribution_temperature_celsius.png", width: 100%)],
  [#image("../../../figures/distribution_pressure_mb.png", width: 100%)],
  [#image("../../../figures/distribution_wind_u.png", width: 100%)],
  [#image("../../../figures/distribution_wind_v.png", width: 100%)],
  [#image("../../../figures/distribution_no2.png", width: 100%)],
  [#image("../../../figures/distribution_co.png", width: 100%)],
  [#image("../../../figures/distribution_o3.png", width: 100%)],
  [#image("../../../figures/distribution_aod.png", width: 100%)],
)
#v(3pt)
#caption([Figures A1–A8. Temperature, pressure, wind components, NO₂, CO, O₃ and AOD. These variables are audited but excluded from the model because their availability at the forecast origin is undocumented.])

#v(2pt)

= Appendix B — Station-month missingness

#keep[
#tbl(
  (1.0fr, 1.15fr, 1.25fr, 1.3fr, 2.2fr),
  table.header(th("Variable"), th("Station-months"), th("With any missing"), th("Stations affected"), th("Maximum station-month missing")),
  [NO₂], [2,179], [258], [19], [100%],
  [CO], [2,179], [258], [19], [100%],
  [O₃], [2,179], [261], [19], [100%],
)
]
#v(3pt)
#caption([Development partition (73 stations, 2016-11 to 2024-12). The full station-month table is `results/pollutant_missingness_station_month_development.csv`; Figure 8 visualises it.])
#v(5pt)
The station-month table shows that missingness is concentrated in a subset of stations and months
rather than being uniformly distributed across the dataset. The gases are retained for audit but
excluded from the model.

= Appendix C — Configuration and data dictionary

#keep[
#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [
    #heading(level: 3)[Reference configuration]
    #tbl(
      (1.3fr, 1.8fr),
      table.header(th("Setting"), th("Value")),
      [Forecast origin], [End of day t],
      [Primary horizon], [1 day],
      [Lookback], [7 days],
      [Maximum gap], [1 day],
      [Temporal folds], [5],
      [Embargo], [1 day],
      [Target rule], [0 < PM2.5 < 500 µg/m³],
      [Random seed], [42],
      [Model family], [CNN–LSTM],
    )
  ],
  [
    #heading(level: 3)[Source columns]
    #tbl(
      (1.95fr, 1.1fr, 0.85fr, 0.95fr),
      table.header(th("Column"), th("Family"), th("Unit"), th("Role")),
      [#codeid("date", size: 7.4pt, max-chars: 22)], [Time], [UTC day], [Timestamp],
      [#codeid("datetime_utc", size: 7.4pt, max-chars: 22)], [Time], [UTC], [Timestamp],
      [#codeid("id", size: 7.4pt, max-chars: 22)], [Metadata], [integer], [Source ID],
      [#codeid("latitude / longitude", size: 7.4pt, max-chars: 22)], [Metadata], [degrees], [Station ID],
      [#codeid("sensor_name", size: 7.4pt, max-chars: 22)], [Metadata], [text], [Station ID],
      [#codeid("pm25", size: 7.4pt, max-chars: 22)], [Target], [µg/m³], [Target],
      [#codeid("temperature_celsius", size: 7.4pt, max-chars: 22)], [Meteorology], [°C], [Audited],
      [#codeid("pressure_mb", size: 7.4pt, max-chars: 22)], [Meteorology], [hPa], [Audited],
      [#codeid("wind_u / wind_v", size: 7.4pt, max-chars: 22)], [Meteorology], [m/s], [Audited],
      [#codeid("no2", size: 7.4pt, max-chars: 22)], [Trace gas], [mol/m²], [Audited],
      [#codeid("co", size: 7.4pt, max-chars: 22)], [Trace gas], [mol/m²], [Audited],
      [#codeid("o3", size: 7.4pt, max-chars: 22)], [Trace gas], [mol/m²], [Audited],
      [#codeid("aod", size: 7.4pt, max-chars: 22)], [Satellite], [unitless], [Audited],
    )
  ],
)
]
#v(4pt)
#caption([Configuration from `src/configs/project.yaml`; source columns from `data/processed/audit.json`. Column names follow the CSV; chemical symbols are used in prose. The model consumes only the target history and the calendar channels declared in the feature contract.])

#v(2pt)

= Appendix D — Executed evidence

#tbl(
  (1.45fr, 2.85fr, 1.35fr, 0.8fr),
  table.header(th("Evidence"), th("Command / artifact"), th("Result"), th("Date")),
  [Full test suite], [#codeid("python -m pytest src/tests")], [63 passed, 0 failed, 22.9 s], [2026-11-14],
  [Fold definitions], [#codeid("folds/folds_{temporal,spatial,standard}.json")], [5 / 5 / diagnostic], [committed],
  [Future holdout], [#codeid("results/future_test.json")], [evaluated once], [committed],
  [Cleaning audit], [#codeid("results/cleaning_quality_audit/")], [735 removed], [recomputed],
  [Processed audit], [#codeid("data/processed/audit.json")], [41,718 → 40,983], [committed],
  [Provenance audit], [#codeid("results/provenance_audit/")], [41,718 matched], [recomputed],
  [EDA statistics], [#codeid("results/Exploratory_Data.csv")], [19 variables], [recomputed],
  [Correlation table], [#codeid("results/correlations.csv")], [36 pairwise rows], [recomputed],
  [Part 1 evidence], [#codeid("scripts/generate_part1_report_evidence.py")], [boxplot, coverage, heatmap], [2026-11-14],
  [Evidence summary], [#codeid("results/part1_evidence_summary.json")], [regenerated], [2026-11-14],
)
#v(4pt)
#caption([The test suite was executed with the project's recorded Python environment; the run log is saved at `reports/reproducibility/pytest_2026-11-14_part1.txt`. The evidence script reads only saved data and artifacts and regenerates the new Part 1 figures and `results/part1_evidence_summary.json`.])
