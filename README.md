<div align="center">

# 🌫️ PM2.5 Forecasting with CNN–LSTM

### A reproducible, time-aware air-quality forecasting research project

**Group 16** · Daily station-level forecasts · One-day-ahead horizon

[![Python 3.11–3.12](https://img.shields.io/badge/Python-3.11--3.12-3776AB?logo=python&logoColor=white)](src/pyproject.toml)
[![Model-CNN--LSTM](https://img.shields.io/badge/primary%20model-CNN--LSTM-6A5ACD)](src/configs/project.yaml)
[![Validation-temporal](https://img.shields.io/badge/validation-time--aware-168C83)](src/splits.py)
[![License-data caveat](https://img.shields.io/badge/data%20license-review%20required-E6A700)](data/manifest.yaml)

**The goal:** use recent PM2.5 history to estimate the next day’s concentration at a monitoring station, and evaluate that estimate without letting future information leak into training.

</div>

> [!IMPORTANT]
> This is a research and coursework project, **not an operational air-quality warning system**. The project records unresolved source-data licensing and provenance questions, and the saved historical future-test score is not a fresh, reproducible evaluation. Read [Limitations and responsible use](#-limitations-and-responsible-use) before interpreting or redistributing the data or results.

## 🧭 Start here

New to the project? This order will help:

1. Read [What the project does](#-what-the-project-does) and [How a forecast is made](#-how-a-forecast-is-made).
2. Follow [Quick start on Windows](#-quick-start-on-windows) to install the environment.
3. Run [the safe development workflow](#-safe-development-workflow) one step at a time.
4. See [Understanding the results](#-understanding-the-results), then open the [technical report](reports/technical_report.pdf) or the [presentation](presentation/slides.pdf). Per-part reports remain available in [`reports/part_1/`](reports/part_1/), [`reports/part_2/`](reports/part_2/) and [`reports/part_3/`](reports/part_3/).

## 🌍 What the project does

PM2.5 is particulate matter with an aerodynamic diameter of 2.5 micrometres or smaller. This project studies whether the recent daily history at a monitoring station can help forecast that station’s PM2.5 concentration one day ahead.

| At a glance | Project setting |
|---|---|
| Prediction target | Daily PM2.5 concentration |
| Target unit | µg/m³ (micrograms per cubic metre) |
| Forecast origin | End of day *t* |
| Main forecast | PM2.5 on day *t + 1* |
| Input history | Seven consecutive daily steps in the configured CNN–LSTM setup |
| Main learned model | CNN–LSTM |
| Main validation | Five expanding temporal folds with a one-day embargo |
| Random seed | 42 |
| Main data source | [Air Quality – Mediterranean Basin (2016–2026)](https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026) |

The configured source CSV contains **41,718 rows**. The recorded cleaning audit retains **40,983 observations across 95 station identities**, spanning **2016-11-25 to 2026-04-06 (UTC)**. A station identity is formed from the sensor name and rounded latitude/longitude so that nearby sites are not automatically merged. See [`data/processed/audit.json`](data/processed/audit.json) and [`data/manifest.yaml`](data/manifest.yaml) for source and cleaning details.

## 🔭 How a forecast is made

At the end of day *t*, the model receives information from that station’s preceding seven-day sequence and outputs a numeric estimate for day *t + 1*:

```text
Past station history (7 daily steps)                  Forecast
┌───────────────────────────────────────────┐        ┌─────────────────┐
│ PM2.5 lags + past rolling values          │ ─────▶ │ PM2.5 at t + 1  │
│ Calendar cycles (day/week and day/year)   │        │     in µg/m³    │
└───────────────────────────────────────────┘        └─────────────────┘
                      Forecast made at the end of day t
```

In the configured experiment, the usable model inputs are calendar features, past PM2.5 values, and rolling summaries of past PM2.5. The source also contains meteorological and satellite-derived variables, but their delivery/availability time has not been verified. **Those environmental predictors are therefore not silently treated as known at forecast time.** Feature configuration lives in [`src/configs/project.yaml`](src/configs/project.yaml).

### 🧠 Model architecture

The main network combines a convolutional layer, which can learn short local patterns, with a recurrent LSTM layer, which can summarize sequence order:

```mermaid
flowchart LR
    A["Seven-day station sequence"] --> B["Fold-local imputation and scaling"]
    B --> C["Causal 1D convolution"]
    C --> D["Layer normalization · ReLU · dropout"]
    D --> E["Max pooling"]
    E --> F["LSTM sequence summary"]
    F --> G["Dense layer"]
    G --> H["One PM2.5 estimate for t + 1"]
```

The default network uses a causal `Conv1D`, layer normalization, ReLU activation, dropout, max pooling, an LSTM, a dense ReLU layer, and a single-value output. It is trained with Adam and mean squared error (MSE); early stopping monitors a chronological inner validation segment. The preprocessing pipeline is fitted on each training fold only, then applied to its validation fold. See [`src/pm25_project/models/cnn_lstm.py`](src/pm25_project/models/cnn_lstm.py) and [`src/pm25_project/preprocessing.py`](src/pm25_project/preprocessing.py).

### 🛡️ Why time-aware validation matters

With time-series data, a random split can let a model learn from dates that occur after the dates it is asked to predict. This project instead evaluates on later time periods and separates training from validation with an embargo:

```mermaid
flowchart LR
    A["Earlier dates<br/>training window"] --> B["One-day embargo"]
    B --> C["Later dates<br/>validation window"]
    C -. "next expanding fold moves forward" .-> A
```

The temporal folds are the primary development evidence. Separate station-grouped folds explore performance on held-out station groups; they answer a different question from forecasting later dates at known stations. Results from standard/randomized folds are diagnostic only and are not the primary basis for model selection. Fold definitions and their purpose are saved in [`folds/folds_temporal.json`](folds/folds_temporal.json) and [`folds/folds_spatial.json`](folds/folds_spatial.json).

## 📏 How to read the evaluation

The project reports metrics on the original PM2.5 scale:

| Metric | Plain-language meaning | Better value |
|---|---|---|
| **RMSE** | Typical error with extra penalty for large misses | Lower |
| **MAE** | Average absolute difference between forecast and observation | Lower |
| **R²** | How much variation is explained relative to a simple mean reference | Higher |

MAPE (percentage error) is not used because values close to zero can make it unstable. A forecast should also be compared with simple baselines such as persistence (reuse the latest observed PM2.5) and a seasonal-naive forecast; a neural network is not automatically useful just because it is more complex.

> [!CAUTION]
> The saved historical future-test metrics are **frozen legacy evidence**, not a score to reproduce by rerunning the evaluator. The recorded values are RMSE **6.0272 µg/m³**, MAE **4.0240 µg/m³**, and R² **0.2394** over **1,716 saved predictions**. The historical checkpoint/configuration needed to reproduce that evaluation is unavailable, and a recorded date-alignment gap remains unresolved. Do not use this score to tune models or claim a verified improvement over persistence. See [`results/future_test_evaluation.json`](results/future_test_evaluation.json) for the recorded values and context.

## 💻 Quick start on Windows

The commands below use **PowerShell**. Python 3.12 (64-bit) is the recorded environment; the package metadata allows Python 3.11 and 3.12.

### 1. Open the project folder

Open PowerShell in the folder containing this README, or navigate to it:

```powershell
cd "C:\path\to\PM25_CNN_LSTM_Submission_Group16"
```

### 2. Create and activate a virtual environment

A virtual environment keeps this project’s Python packages separate from other projects on your computer.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, allow it for **this PowerShell process only**, then activate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install the project and its recorded dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e src
```

The dependency file includes TensorFlow and analysis, reporting, and testing packages, so installation may take a few minutes. The editable install makes the `pm25_project` package available to Python while you work in this folder.

### 4. Check the configuration

```powershell
python -m pm25_project.cli validate-config --config src/configs/project.yaml
```

You should see `Configuration valid`. The project config expects the source CSV at `data/raw/extracted/combined_gee_pm25.csv`. If the file is absent in your copy, obtain it from the [dataset page](https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026), review its usage terms, and place the CSV at the configured path. There is no automatic data download in this project.

## ▶️ Safe development workflow

Run these commands **in order** from the project root. Each command uses the same explicit configuration:

```powershell
# 1. Read, validate, and clean the configured source data.
python -m pm25_project.cli audit --config src/configs/project.yaml

# 2. Create exploratory summaries and plots.
python -m pm25_project.cli eda --config src/configs/project.yaml

# 3. Create or inspect the time-aware development fold definitions.
python -m pm25_project.cli folds --config src/configs/project.yaml

# 4. Score simple baselines on the defined folds.
python -m pm25_project.cli baselines --config src/configs/project.yaml

# 5. Run the primary CNN–LSTM temporal development experiment.
python -m pm25_project.cli cnn-lstm --config src/configs/project.yaml
```

The `audit` step creates the cleaned daily file under `data/processed/`. The later commands use that file. EDA, fold creation, baseline scoring, and model training can take longer depending on the computer; CNN–LSTM training requires the TensorFlow dependency.

The five template entry points wrap these commands and resolve the configuration path for you:

```powershell
python scripts/run_eda.py           # audit + eda + template figures
python scripts/run_training.py      # CNN–LSTM training with inner selection
python scripts/run_evaluation.py    # results.csv and comparisons.csv
python scripts/run_ablation.py      # ablation.csv
python scripts/run_explicability.py # importance, local cases, ranking stability
```

The notebooks in `notebooks/` reproduce the exploratory checks without retraining: `01_eda.ipynb`, `02_timeseries.ipynb` and `03_test_models.ipynb`.

### Optional development analyses

After the primary workflow, these commands run additional **development** analyses:

```powershell
# Evaluate the CNN–LSTM on station-grouped folds.
python -m pm25_project.cli cnn-lstm-spatial --config src/configs/project.yaml

# Examine variation across random seeds.
python -m pm25_project.cli cnn-lstm-seeds --config src/configs/project.yaml

# Explore selected CNN–LSTM hyperparameter sensitivity.
python -m pm25_project.cli cnn-sensitivity --config src/configs/project.yaml

# Compare configured feature ablations.
python -m pm25_project.cli cnn-feature-ablation --config src/configs/project.yaml

# Examine sensitivity to training-set size.
python -m pm25_project.cli cnn-train-size --config src/configs/project.yaml

# Build development-only comparison/report assets.
python -m pm25_project.cli cnn-lstm-finalize-dev --config src/configs/project.yaml
python -m pm25_project.cli report-assets --config src/configs/project.yaml
```

Use the scripts in `scripts/` for the additional studies represented in the results archive. For example, `python scripts/run_optuna_development.py` launches a nested development-only hyperparameter search. Check a script’s source before changing its settings or starting a long run.

### ✅ Run the safe test suite

```powershell
python -m pytest -c src/pyproject.toml src/tests -q --ignore=src/tests/test_final_evaluation.py
```

The one excluded test module exercises the single-use future evaluator. Do not run that evaluator as part of routine setup or development.

## 📦 Repository layout

The tree follows the course template; every path below exists in this repository.

```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                     source CSV (kept local, see data/manifest.yaml)
│   └── processed/               cleaned daily panel + cleaning audit (kept local)
├── notebooks/
│   ├── 01_eda.ipynb             descriptive statistics, missingness, association
│   ├── 02_timeseries.ipynb      coverage, ACF/PACF, ADF/KPSS, drift
│   └── 03_test_models.ipynb     fold check, naive baselines, linear benchmark
├── src/
│   ├── config.py                paths and SEED = 42
│   ├── preprocessing.py         cleaning, imputation, fold-local scaling
│   ├── features.py              lags, rolling means, cyclic calendar terms
│   ├── splits.py                temporal, spatial and diagnostic folds
│   ├── models.py                model registry (CNN–LSTM)
│   ├── test_evaluation.py       RMSE, MAE, R², paired bootstrap
│   ├── explainability.py        permutation importance, rank stability
│   ├── ablation.py              feature-family ablation
│   ├── pm25_project/            full implementation behind those entry points
│   ├── configs/                 project settings, feature families, search spaces
│   ├── tests/                   automated checks (splits, leakage, contracts, models)
│   └── pyproject.toml           package metadata and pytest settings
├── scripts/
│   ├── run_eda.py               EDA tables and figures
│   ├── run_training.py          CNN–LSTM training and inner selection
│   ├── run_evaluation.py        results.csv and comparisons.csv
│   ├── run_ablation.py          ablation.csv
│   ├── run_explicability.py     importance and ranking stability
│   └── build_*.py, audit_*.py   report/deck builds and evidence audits
├── folds/
│   ├── folds_temporal.json      5 expanding temporal folds
│   ├── folds_spatial.json       5 station-grouped folds
│   └── folds_standard.json      5 random diagnostic folds
├── results/
│   ├── Exploratory_Data.csv
│   ├── Exploratory_temporal.csv
│   ├── results.csv
│   ├── hyperparameters.csv
│   ├── feature_importance.csv
│   ├── comparisons.csv
│   ├── experiments.csv
│   ├── ablation.csv
│   ├── ranking_stability.csv
│   └── predictions/, tables/, eda/, models/, and the frozen historical evidence
├── figures/
│   ├── eda_histogrammes.png
│   ├── correlation_matrix.png
│   ├── robustness_plot.png
│   ├── ablation_barplot.png
│   └── the per-model and validation charts used in the reports
├── reports/
│   ├── technical_report.pdf     14-page English technical report
│   ├── part_1/ part_2/ part_3/  per-part reports and sources
│   ├── reproducibility/         validation records and the frozen baseline hash manifest
│   └── source/                  Typst source of the technical report
└── presentation/
    ├── slides.pdf               17-slide talk (about 12 minutes)
    ├── slides.pptx              editable source of the talk
    └── Part_1/2/3_Presentation.pptx
```

| Template path | What you will find there |
|---|---|
| [`src/config.py`](src/config.py) | Project root, data/results/figure paths and `SEED = 42`, plus the validated config loader |
| [`src/preprocessing.py`](src/preprocessing.py) | `clean_data`, `numeric_preprocessor`, `SequencePreprocessor` (fold-local imputation and scaling) |
| [`src/features.py`](src/features.py) | `build_origin_features`: past-only lags, rolling means and calendar cycles |
| [`src/splits.py`](src/splits.py) | `temporal_panel_folds`, `spatial_folds`, `random_diagnostic_folds`, `save_folds` |
| [`src/models.py`](src/models.py) | `model_registry` with the CNN–LSTM builder |
| [`src/test_evaluation.py`](src/test_evaluation.py) | `regression_metrics` and `paired_bootstrap_ci` |
| [`src/explainability.py`](src/explainability.py) | Permutation importance and Spearman/Jaccard rank stability |
| [`src/ablation.py`](src/ablation.py) | Feature-family ablation on matched samples |
| [`src/pm25_project/`](src/pm25_project/) | Full research implementation (sequences, training, diagnostics, benchmarks) |
| [`src/configs/`](src/configs/) | Project/data settings, feature families, and model search spaces |
| [`src/tests/`](src/tests/) | Automated checks for preprocessing, data contracts, splits, leakage, metrics, and models |
| [`notebooks/`](notebooks/) | The three exploratory notebooks, executed on the cleaned panel |
| [`scripts/`](scripts/) | The five template entry points plus report/deck builds and evidence audits |
| [`folds/`](folds/) | Temporal, station-grouped, and diagnostic fold definitions |
| [`data/manifest.yaml`](data/manifest.yaml) | Dataset origin, recorded checksums, and license caveat |
| [`data/processed/audit.json`](data/processed/audit.json) | Row-level cleaning audit behind the 40,983-row panel |
| [`results/`](results/) | Result tables at the root, plus predictions, tables, diagnostics and frozen historical evidence |
| [`figures/`](figures/) | Template summary figures and every chart used in the reports |
| [`reports/technical_report.pdf`](reports/technical_report.pdf) | The 14-page English technical report |
| [`presentation/slides.pdf`](presentation/slides.pdf) | The 17-slide presentation for a 10–15 minute talk |
| [`reports/reproducibility/`](reports/reproducibility/) | Validation records and integrity checks for saved artifacts |

## 📚 Key artifacts and further reading

- **Project configuration:** [`src/configs/project.yaml`](src/configs/project.yaml)
- **Data provenance and licensing notes:** [`data/manifest.yaml`](data/manifest.yaml)
- **Cleaning audit:** [`data/processed/audit.json`](data/processed/audit.json)
- **Temporal out-of-fold results:** [`results/temporal_cv_results.csv`](results/temporal_cv_results.csv)
- **Development summary:** [`results/development_cv_summary.csv`](results/development_cv_summary.csv)
- **Model comparisons:** [`results/comparisons.csv`](results/comparisons.csv)
- **Technical report (14 pages):** [`reports/technical_report.pdf`](reports/technical_report.pdf)
- **Presentation (17 slides):** [`presentation/slides.pdf`](presentation/slides.pdf)
- **Saved Part 1 report:** [`reports/part_1/Part_1_Report.pdf`](reports/part_1/Part_1_Report.pdf)
- **Saved Part 2 report:** [`reports/part_2/Part_2_Report.pdf`](reports/part_2/Part_2_Report.pdf)
- **Saved Part 3 report:** [`reports/part_3/Part_3_Report.pdf`](reports/part_3/Part_3_Report.pdf)
- **Saved Part 1 presentation:** [`presentation/Part_1_Presentation.pptx`](presentation/Part_1_Presentation.pptx)
- **Saved Part 2 presentation:** [`presentation/Part_2_Presentation.pptx`](presentation/Part_2_Presentation.pptx)
- **Saved Part 3 presentation:** [`presentation/Part_3_Presentation.pptx`](presentation/Part_3_Presentation.pptx)

## ⚠️ Limitations and responsible use

1. **Dataset rights:** Kaggle labels the compiled dataset license “Other (specified in description).” Its description refers to OpenAQ, Copernicus, and NASA data, but the compiled bundle’s redistribution rights are not clear. The project’s manifest recommends local academic analysis until source-by-source attribution and redistribution rights are confirmed. Do not assume that including a file in this repository grants permission to redistribute it.
2. **Data provenance:** The provided combined CSV does not include a complete transformation log establishing exactly how its upstream feature join was performed.
3. **Predictor availability:** Delivery latency for exogenous environmental measurements has not been established. Current configured prediction features therefore avoid treating these values as if they were known at the forecast origin.
4. **Predictive evidence:** The available evidence describes modest predictive skill and does not establish a statistically significant improvement over persistence. Development-fold results are not guarantees of future performance.
5. **Historical holdout:** The saved future-test result is frozen historical evidence. Its original checkpoint/configuration is not available for a fresh evaluation, and recorded alignment limitations should be disclosed.
6. **Not operational advice:** Predictions are research outputs, not regulatory measurements, exposure advice, or a substitute for official air-quality monitoring.

For integrity and leakage reasons, do not use future-test labels to select features, tune hyperparameters, or compare new models. The configuration marks finalization as locked; use the development folds for new experiments.

## 🧰 Troubleshooting

| Problem | What to check |
|---|---|
| `No module named pm25_project` | Activate `.venv`, confirm you are in the repository root, and rerun `python -m pip install -e src`. |
| Missing source CSV / cleaned data | Confirm the configured raw CSV exists. Run `audit` before `eda`, `folds`, or model commands. |
| TensorFlow is unavailable | Activate the intended environment and install dependencies with `python -m pip install -r requirements.txt`. |
| Configuration validation fails | Check the data path, date and target columns, target units, and station identity settings in [`src/configs/project.yaml`](src/configs/project.yaml). |
| PowerShell refuses to activate `.venv` | Use the process-scoped execution-policy command in [Quick start](#-quick-start-on-windows); avoid changing the machine-wide policy for this project. |

## 🙌 Acknowledgements

The source dataset is attributed to Sarah Ameziane on Kaggle and the providers identified in the dataset description, including OpenAQ, Copernicus, and NASA. See the [dataset page](https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026) and [`data/manifest.yaml`](data/manifest.yaml); confirm the applicable terms for each source before reuse or redistribution.
