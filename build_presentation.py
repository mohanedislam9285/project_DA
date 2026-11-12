from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "16"
PRESENTATION = ROOT / "presentation"
OUTPUT = PRESENTATION / "final_presentation_submission.pptx"

NAVY = RGBColor(16, 36, 58)
TEAL = RGBColor(8, 127, 128)
INK = RGBColor(36, 52, 71)
MUTED = RGBColor(96, 113, 132)
PALE = RGBColor(238, 244, 246)
ICE = RGBColor(232, 243, 243)
AMBER = RGBColor(217, 130, 43)
WHITE = RGBColor(255, 255, 255)
RULE = RGBColor(213, 222, 231)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
slide_notes: list[tuple[str, str]] = []


def set_background(slide, color=WHITE):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def text_box(slide, x, y, w, h, text, size=18, color=INK, bold=False, align=None):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.03)
    frame.margin_right = Inches(0.03)
    frame.margin_top = Inches(0.02)
    frame.margin_bottom = Inches(0.02)
    for i, line in enumerate(str(text).split("\n")):
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.font.name = "Aptos"
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = color
        paragraph.space_after = Pt(6)
        if align is not None:
            paragraph.alignment = align
    return shape


def bullet_list(slide, x, y, w, h, lines, size=17, color=INK, gap=9):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.05)
    frame.margin_right = Inches(0.05)
    for i, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        paragraph.text = f"• {line}"
        paragraph.level = 0
        paragraph.font.name = "Aptos"
        paragraph.font.size = Pt(size)
        paragraph.font.color.rgb = color
        paragraph.space_after = Pt(gap)
    return shape


def add_note(slide, title, note):
    slide.notes_slide.notes_text_frame.text = note
    slide_notes.append((title, note))


def new_slide(title, eyebrow="GROUP 16  |  CNN–LSTM"):
    slide = prs.slides.add_slide(BLANK)
    set_background(slide)
    text_box(slide, 0.62, 0.27, 11.8, 0.24, eyebrow.upper(), 9, TEAL, True)
    text_box(slide, 0.62, 0.55, 12.0, 0.57, title, 26, NAVY, True)
    rule = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.64), Inches(1.19), Inches(0.85), Inches(0.05)
    )
    rule.fill.solid()
    rule.fill.fore_color.rgb = TEAL
    rule.line.fill.background()
    text_box(
        slide,
        0.62,
        7.15,
        12.05,
        0.2,
        f"PM2.5 FORECASTING  •  {len(prs.slides):02d}",
        8,
        MUTED,
    )
    return slide


def card(slide, x, y, w, h, title, body, fill=PALE, title_size=17, body_size=13):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = RULE
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.14)
    frame.margin_right = Inches(0.14)
    frame.margin_top = Inches(0.10)
    frame.margin_bottom = Inches(0.07)
    p = frame.paragraphs[0]
    p.text = title
    p.font.name = "Aptos"
    p.font.size = Pt(title_size)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.space_after = Pt(5)
    for line in str(body).split("\n"):
        p = frame.add_paragraph()
        p.text = line
        p.font.name = "Aptos"
        p.font.size = Pt(body_size)
        p.font.color.rgb = INK
        p.space_after = Pt(3)
    return shape


def add_table(slide, x, y, w, h, headers, rows, font_size=12):
    table_shape = slide.shapes.add_table(
        len(rows) + 1,
        len(headers),
        Inches(x),
        Inches(y),
        Inches(w),
        Inches(h),
    )
    table = table_shape.table
    for column, header in enumerate(headers):
        cell = table.cell(0, column)
        cell.text = str(header)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for row_index, row in enumerate(rows, start=1):
        for column, value in enumerate(row):
            cell = table.cell(row_index, column)
            cell.text = str(value)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if row_index % 2 else PALE
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for row_index, row in enumerate(table.rows):
        for cell in row.cells:
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.045)
            cell.margin_bottom = Inches(0.045)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.name = "Aptos"
                paragraph.font.size = Pt(font_size)
                paragraph.font.color.rgb = WHITE if row_index == 0 else INK
                if row_index == 0:
                    paragraph.font.bold = True
    return table_shape


def add_image(slide, relative_path, x, y, w=None, h=None):
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Presentation figure not found: {path}")
    kwargs = {}
    if w is not None:
        kwargs["width"] = Inches(w)
    if h is not None:
        kwargs["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kwargs)


def build_ablation_figure():
    summary = pd.read_csv(RESULTS / "ablation_summary.csv")
    expected = [
        "full",
        "without_calendar",
        "without_lag_and_rolling",
        "without_rolling",
    ]
    keyed = summary.set_index("variant")
    missing = set(expected).difference(keyed.index)
    if missing:
        raise ValueError(f"Ablation summary is missing variants: {sorted(missing)}")
    data = keyed.loc[expected]
    labels = ["Full features", "No calendar", "No lag / rolling", "No rolling"]
    fig, axis = plt.subplots(figsize=(7.1, 3.9))
    bars = axis.barh(
        labels,
        data["mean"],
        xerr=data["std"],
        capsize=4,
        color=["#087F80", "#3B8E9A", "#D9822B", "#6F8297"],
        edgecolor="white",
        linewidth=0.8,
    )
    axis.invert_yaxis()
    axis.set_xlabel("Mean fold RMSE (µg/m³)", fontsize=10)
    axis.set_title(
        "Available-feature ablation · five temporal folds", fontsize=12, weight="bold"
    )
    axis.grid(axis="x", alpha=0.22)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=10)
    for bar, value in zip(bars, data["mean"]):
        axis.text(
            value + 0.13,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=9,
            color="#243447",
        )
    fig.tight_layout()
    path = RESULTS / "figures" / "available_feature_ablation_slides.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path.relative_to(ROOT)


def cover_slide():
    slide = prs.slides.add_slide(BLANK)
    set_background(slide, NAVY)
    text_box(
        slide,
        0.85,
        0.60,
        11.4,
        0.4,
        "GROUP 16  /  TECHNICAL PRESENTATION",
        12,
        RGBColor(98, 198, 194),
        True,
    )
    text_box(
        slide,
        0.85,
        1.45,
        11.6,
        1.6,
        "One-day-ahead\nPM2.5 forecasting",
        36,
        WHITE,
        True,
    )
    text_box(
        slide,
        0.90,
        3.35,
        10.8,
        0.55,
        "A leakage-aware CNN–LSTM regression study",
        22,
        RGBColor(98, 198, 194),
        True,
    )
    card(
        slide,
        0.9,
        4.55,
        11.3,
        1.32,
        "Bottom line",
        "A plausible sequence model, but no statistically reliable overall gain over persistence; extreme-event errors remain large.",
        RGBColor(231, 240, 244),
        16,
        15,
    )
    text_box(
        slide,
        0.9,
        6.65,
        10.8,
        0.3,
        "Historical future metrics are frozen; development validation is reported separately.",
        12,
        WHITE,
    )
    add_note(
        slide,
        "Title",
        "Open by defining the forecast origin: use observations through the end of day t to predict station PM2.5 at t+1. State that this is a regression study. Development CV and previously saved future-period metrics are distinct.",
    )


def main():
    ablation_figure = build_ablation_figure()
    cover_slide()

    slide = new_slide("Forecast task and research questions", "01  /  MOTIVATION")
    card(
        slide,
        0.72,
        1.48,
        5.85,
        1.05,
        "Why it matters",
        "Short-horizon forecasts can help air-quality monitoring prioritize review and timely communication; forecasts are decision support, not health or regulatory determinations.",
        ICE,
        16,
        13,
    )
    card(
        slide,
        6.78,
        1.48,
        5.82,
        1.05,
        "Regression, not classification",
        "Target is continuous PM2.5 in µg/m³ at t+1. RMSE, MAE and R² retain concentration magnitude.",
        PALE,
        16,
        13,
    )
    text_box(
        slide,
        0.78,
        2.85,
        11.8,
        0.55,
        "Main RQ  |  How accurately can issue-time history predict next-day PM2.5, and does CNN–LSTM beat past-only forecasts under leakage-safe validation?",
        15,
        NAVY,
        True,
    )
    add_table(
        slide,
        0.78,
        3.55,
        11.8,
        2.85,
        ["Label", "Research question"],
        [
            [
                "RQ1",
                "How does CNN–LSTM compare with persistence and seasonal-naive baselines?",
            ],
            [
                "RQ2",
                "What do temporal and station-grouped evaluations say about generalization?",
            ],
            [
                "RQ3",
                "Which available feature groups contribute, and how stable are they?",
            ],
            [
                "RQ4",
                "How do seasonality, sparse coverage, extreme events, missingness and rights limit use?",
            ],
        ],
        12,
    )
    add_note(
        slide,
        "Task and RQs",
        "Explain why monitoring can benefit from forecasts while avoiding an unsupported claim that the model itself improves health. Repeat that the target is a real-valued concentration, not a category. Introduce the main research question and the four labelled questions.",
    )

    slide = new_slide("Dataset and station identity", "02  /  DATA")
    add_table(
        slide,
        0.78,
        1.55,
        7.2,
        3.35,
        ["Item", "Audited value"],
        [
            ["Prepared CSV rows", "41,718"],
            ["After target cleaning", "40,983"],
            ["Cleaned station IDs", "95"],
            ["Date range", "2016-11-25 to 2026-04-06"],
            ["Invalid PM2.5 removed", "735 (≤0 or ≥500)"],
            ["Source", "Kaggle, Mediterranean Basin, v1"],
        ],
        13,
    )
    card(
        slide,
        8.25,
        1.55,
        4.2,
        2.0,
        "Station key",
        "Sensor name + latitude + longitude rounded to six decimals. Coordinates keep same-name sites distinct.",
        ICE,
        16,
        13,
    )
    card(
        slide,
        8.25,
        3.78,
        4.2,
        2.05,
        "Count reconciliation",
        "98 source IDs → 98 composite stations → 95 after cleaning. 94 sensor names; four names span two sites; three test entries removed.",
        PALE,
        16,
        13,
    )
    text_box(
        slide,
        0.82,
        5.25,
        7.1,
        0.85,
        "Rights caveat: publisher metadata says “Other (specified in description)”; redistribution rights for the compiled bundle remain unresolved.",
        14,
        AMBER,
        True,
    )
    add_note(
        slide,
        "Data and identity",
        "The active file is a publisher-prepared combined table. Explain the composite station ID and why the source’s numeric-ID count differs from the cleaned station count. Be explicit that compiled-data redistribution rights remain unresolved.",
    )

    slide = new_slide("Seasonality and PM2.5 thresholds", "03  /  EXPLORATORY ANALYSIS")
    add_image(slide, "results/16/figures/pm25_by_season.png", 0.75, 1.50, w=6.4, h=4.75)
    add_table(
        slide,
        7.42,
        1.68,
        5.1,
        3.95,
        ["Season", "Mean", "Median", ">25", ">50"],
        [
            ["DJF", "10.94", "8.77", "6.15%", "0.45%"],
            ["MAM", "8.25", "6.83", "1.92%", "0.36%"],
            ["JJA", "9.59", "8.50", "1.95%", "0.15%"],
            ["SON", "9.08", "7.47", "2.46%", "0.25%"],
        ],
        11,
    )
    text_box(
        slide,
        7.45,
        5.82,
        5.0,
        0.82,
        "Winter has the highest mean and >25/>50 shares. Cutoffs are descriptive, not presented as compliance standards.",
        12,
        MUTED,
    )
    add_note(
        slide,
        "Seasonality",
        "Seasonal PM2.5 summaries show the highest mean and >25 and >50 shares in DJF. They are descriptive record-level statistics. The project’s 25 and 50 µg/m³ cutoffs are not described as normative thresholds.",
    )

    slide = new_slide(
        "Processing workflow and leakage prevention", "04  /  METHODOLOGY"
    )
    flow = [
        ("Original CSV", "41,718 rows"),
        ("Cleaning", "Target bounds\nstation-day audit"),
        ("Preprocessing", "Fold-local\nimputer / scaler"),
        ("Features", "7-day past-only\nsequences"),
        ("Training", "Inner TPE\nouter scoring"),
    ]
    x_positions = [0.7, 3.2, 5.7, 8.2, 10.7]
    for i, (title, body) in enumerate(flow):
        card(
            slide,
            x_positions[i],
            1.62,
            1.92,
            1.22,
            title,
            body,
            ICE if i in (0, 4) else PALE,
            14,
            11,
        )
        if i < len(flow) - 1:
            text_box(
                slide, x_positions[i] + 1.95, 1.92, 0.50, 0.35, "→", 22, TEAL, True
            )
    bullet_list(
        slide,
        0.88,
        3.25,
        11.7,
        2.8,
        [
            "Temporal folds preserve date order and include a one-day raw-date embargo.",
            "Every sequence ends at day t; label is exactly station PM2.5 at t+1. Station changes and date gaps break sequences.",
            "Imputation and scaling are fitted only on each training fold. Hyperparameter selection uses chronological inner validation inside that fold.",
            "Inputs at/after cutoff were mutated in tests: earlier lag, rolling and sequence features remained unchanged.",
            "Meteorology, AOD, NO₂, CO and O₃ are excluded because forecast-issue-time availability is not verified.",
        ],
        15,
    )
    add_note(
        slide,
        "Workflow and leakage",
        "Walk through the workflow from prepared CSV to fold scoring. Emphasize the end-of-day origin and t+1 label, past-only features, embargo, fold-local preprocessing and inner-only tuning. Exogenous fields are excluded because observation dates alone do not prove availability to a forecaster.",
    )

    slide = new_slide("CNN–LSTM architecture", "05  /  MODEL")
    blocks = [
        (
            0.72,
            1.88,
            1.72,
            1.1,
            "7-day input",
            "9 processed channels\n(7 features + 2 flags)",
            ICE,
        ),
        (
            3.0,
            1.88,
            2.15,
            1.1,
            "Causal Conv1D",
            "16 filters · k=3\nlocal patterns",
            RGBColor(230, 238, 246),
        ),
        (5.75, 1.88, 1.8, 1.1, "Norm + ReLU", "Dropout 0.347\nMaxPool", PALE),
        (
            8.12,
            1.88,
            1.7,
            1.1,
            "LSTM",
            "64 units\ngated memory",
            RGBColor(248, 241, 232),
        ),
        (10.4, 1.88, 2.05, 1.1, "Dense → output", "32 ReLU\nscalar regression", ICE),
    ]
    for x, y, w, h, title, body, fill in blocks:
        card(slide, x, y, w, h, title, body, fill, 15, 12)
    for x in [2.47, 5.22, 7.66, 9.91]:
        text_box(slide, x, 2.22, 0.4, 0.35, "→", 22, TEAL, True)
    card(
        slide,
        0.86,
        3.5,
        5.65,
        1.42,
        "CNN: local temporal patterns",
        "Shared filters scan nearby days to learn short motifs such as local rises, falls and fluctuations.",
        PALE,
        16,
        14,
    )
    card(
        slide,
        6.78,
        3.5,
        5.65,
        1.42,
        "LSTM: ordered dependencies",
        "Gates update, retain or forget sequence information before a dense head maps the representation to PM2.5.",
        ICE,
        16,
        14,
    )
    text_box(
        slide,
        0.92,
        5.42,
        11.5,
        0.78,
        "Why this architecture? It combines local pattern extraction with sequential memory. That makes it plausible for short temporal sequences—not automatically better than persistence.",
        16,
        NAVY,
        True,
    )
    add_note(
        slide,
        "Architecture",
        "Explain causal convolution as local feature extraction over adjacent time positions. Explain the LSTM gates as a mechanism for retaining or updating sequential state. The scalar output is continuous regression. Justify the architecture as a hypothesis and note that evaluation does not prove it beats persistence.",
    )

    slide = new_slide("Training configuration and loss trace", "06  /  FITTING")
    add_table(
        slide,
        0.75,
        1.55,
        4.45,
        3.75,
        ["Setting", "Reference setting"],
        [
            ["History", "7 daily steps"],
            ["Conv1D", "16 filters, kernel 3"],
            ["LSTM / dense", "64 / 32 units"],
            ["Dropout", "0.347"],
            ["Optimizer", "Adam"],
            ["Loss", "MSE"],
            ["Search", "50 TPE trials/fold"],
        ],
        12,
    )
    add_image(
        slide,
        "results/16/figures/training_validation_loss_typst.png",
        5.48,
        1.52,
        w=7.1,
        h=3.65,
    )
    text_box(
        slide,
        0.82,
        5.63,
        11.6,
        0.72,
        "Archived inner-fit history: best validation MSE 86.86 (µg/m³)² at epoch 45; training loss continues down. This is not an outer-fold score or the frozen historical checkpoint.",
        13,
        MUTED,
    )
    add_note(
        slide,
        "Training curve",
        "Show both training and chronological inner-validation MSE. The best inner validation loss occurs at epoch 45, followed by modest improvement/flattening; the training loss continues lower, consistent with overfitting. This curve is a representative archived inner fit and must not be conflated with outer-fold metrics.",
    )

    slide = new_slide("Validation design", "07  /  ROBUSTNESS")
    add_table(
        slide,
        0.78,
        1.55,
        11.8,
        3.25,
        ["Fold", "Validation dates", "Train sequences", "Validation n", "Search"],
        [
            ["1", "2019-04-30 – 2020-10-16", "482", "322", "50 trials"],
            ["2", "2020-10-17 – 2021-10-24", "801", "172", "50 trials"],
            ["3", "2021-10-25 – 2022-11-18", "976", "377", "50 trials"],
            ["4", "2022-11-19 – 2024-01-30", "1,353", "733", "50 trials"],
            ["5", "2024-01-31 – 2024-12-07", "2,086", "1,263", "50 trials"],
        ],
        12,
    )
    card(
        slide,
        0.86,
        5.15,
        5.6,
        1.08,
        "Temporal CV",
        "2,867 outer OOF predictions; expanding training windows.",
        ICE,
        15,
        12,
    )
    card(
        slide,
        6.78,
        5.15,
        5.6,
        1.08,
        "Spatial CV",
        "Held-out station groups; each has historical input sequences.",
        PALE,
        15,
        12,
    )
    add_note(
        slide,
        "Validation",
        "Temporal folds test later periods with expanding training data. Spatial folds test held-out stations conditional on having historical sequences. These answer different questions and are not interchangeable. Hyperparameter search is nested in outer training data.",
    )

    slide = new_slide("Development metrics and baselines", "08  /  RESULTS")
    add_table(
        slide,
        0.78,
        1.55,
        11.8,
        2.55,
        [
            "Matched fixed diagnostic",
            "RMSE, mean ± SD",
            "MAE, mean ± SD",
            "R², mean ± SD",
        ],
        [
            ["CNN–LSTM", "8.301 ± 6.221", "3.845 ± 0.921", "0.156 ± 0.101"],
            ["Persistence", "8.771 ± 8.984", "3.158 ± 1.468", "0.207 ± 0.476"],
            [
                "Weekly seasonal-naive",
                "10.798 ± 10.039",
                "4.721 ± 1.718",
                "−0.287 ± 0.525",
            ],
        ],
        13,
    )
    card(
        slide,
        0.82,
        4.42,
        5.65,
        1.28,
        "TPE-selected temporal model",
        "Mean-fold RMSE 8.520 ± 6.034\nMean-fold MAE 3.978 ± 0.892",
        ICE,
        15,
        13,
    )
    card(
        slide,
        6.75,
        4.42,
        5.65,
        1.28,
        "Paired result vs persistence",
        "Exact fold Wilcoxon p = .625\nHolm-adjusted p = 1.00",
        PALE,
        15,
        13,
    )
    text_box(
        slide,
        0.87,
        6.0,
        11.6,
        0.48,
        "Fixed-configuration baseline table and TPE-selected results are distinct analyses; five dependent folds imply low inferential power.",
        12,
        MUTED,
    )
    add_note(
        slide,
        "Evaluation",
        "The complete RMSE/MAE/R² table is a fixed predeclared CNN–LSTM compared with persistence and weekly seasonal-naive on matched folds. Separately, per-fold Optuna tuning gives mean-fold RMSE 8.520 and MAE 3.978. The paired test does not show a significant improvement over persistence.",
    )

    slide = new_slide("Frozen historical future estimate", "09  /  LOCKED RESULT")
    metrics = [
        ("6.0272", "RMSE µg/m³"),
        ("4.0240", "MAE µg/m³"),
        ("0.2394", "R²"),
        ("1,716", "predictions"),
    ]
    for i, (value, label) in enumerate(metrics):
        card(
            slide,
            0.82 + i * 3.05,
            1.88,
            2.7,
            1.25,
            value,
            label,
            ICE if i % 2 == 0 else PALE,
            23,
            12,
        )
    card(
        slide,
        0.90,
        3.65,
        5.65,
        1.55,
        "What this number is",
        "Previously saved historical score; reported unchanged and separate from development CV.",
        PALE,
        16,
        14,
    )
    card(
        slide,
        6.78,
        3.65,
        5.65,
        1.55,
        "What it is not",
        "No new holdout evaluation or baseline comparison was run for this submission.",
        ICE,
        16,
        14,
    )
    text_box(
        slide,
        0.94,
        5.73,
        11.2,
        0.72,
        "A development-only refit reproduces all saved predictions (max |Δ| = 7.1 × 10⁻¹⁵); original weight-file identity cannot be verified.",
        14,
        MUTED,
    )
    add_note(
        slide,
        "Frozen score",
        "State clearly that this is a previously saved score, not a new experiment in this turn. No future labels were read and no new baseline score was computed. Prediction equivalence does not prove that the absent original checkpoint has identical bytes.",
    )

    slide = new_slide(
        "Explainability and inactive pollutant indicators", "10  /  INTERPRETATION"
    )
    add_table(
        slide,
        0.78,
        1.55,
        7.2,
        3.55,
        ["Feature channel", "Mean ΔRMSE", "Mean rank"],
        [
            ["PM2.5 lag-1", "0.651", "1.0"],
            ["Rolling-3 mean", "0.153", "3.0"],
            ["Rolling-7 missing flag", "0.052", "4.8"],
            ["Day-of-year sine", "0.033", "5.6"],
            ["Rolling-3 missing flag", "0.017", "5.8"],
        ],
        12,
    )
    card(
        slide,
        8.25,
        1.58,
        4.1,
        1.6,
        "Interpretation",
        "Lag-1 is consistently important; it also helps explain why persistence is hard to beat.",
        ICE,
        15,
        13,
    )
    card(
        slide,
        8.25,
        3.42,
        4.1,
        1.72,
        "Why no pollutant flags?",
        "NO₂/CO/O₃ values are excluded because issue-time availability is unverified. Their missingness flags are therefore not active in the final model.",
        PALE,
        15,
        13,
    )
    text_box(
        slide,
        0.85,
        5.58,
        11.4,
        0.7,
        "Permutation importance is model-conditional, not causal. The final feature-selected refit uses PM2.5 lag-1 only.",
        13,
        MUTED,
    )
    add_note(
        slide,
        "Explainability",
        "Permutation importance ranks lag-1 first in all five folds. Missingness indicators exist for derived rolling channels in the reference preprocessing. Pollutant-specific flags are not in the final selected model because the pollutants themselves are excluded pending issue-time availability evidence.",
    )

    slide = new_slide("Ablation and robustness", "11  /  SENSITIVITY")
    add_image(slide, str(ablation_figure), 0.72, 1.47, w=6.5, h=4.7)
    add_table(
        slide,
        7.55,
        1.72,
        4.95,
        3.7,
        ["Feature removal", "Δ mean RMSE", "95% interval"],
        [
            ["No calendar", "−0.467", "[−0.957, −0.088]"],
            ["No lag/rolling", "+0.617", "[+0.249, +1.022]"],
            ["No rolling", "−0.177", "[−0.337, −0.043]"],
        ],
        11,
    )
    text_box(
        slide,
        7.6,
        5.7,
        4.8,
        0.68,
        "Fixed folds, no retuning. Exogenous-family ablations are blocked, not negative results.",
        12,
        MUTED,
    )
    add_note(
        slide,
        "Ablation",
        "Removing target lag and rolling features worsens fold RMSE. Removing calendar or rolling channels produces small improvements in this fixed no-retuning experiment. Do not treat this as an optimized feature search. Exogenous ablation variants are blocked because availability is unverified.",
    )

    slide = new_slide(
        "Error analysis: typical and extreme cases", "12  /  FAILURE MODES"
    )
    add_table(
        slide,
        0.78,
        1.55,
        11.8,
        3.65,
        ["OOF case", "Actual", "Predicted", "|Error|", "Reading"],
        [
            ["Low, 2020-06-28", "5.00", "3.96", "1.04", "Small miss"],
            ["Moderate, 2022-01-21", "17.50", "12.96", "4.54", "Underpredicts rise"],
            ["Elevated, 2022-02-04", "37.92", "13.88", "24.04", "Large miss"],
            [
                "Extreme, 2022-02-13",
                "124.33",
                "15.73",
                "108.60",
                "Severe underprediction",
            ],
            [
                "Largest, 2022-02-14",
                "338.80",
                "16.16",
                "322.64",
                "Far outside learned range",
            ],
        ],
        11,
    )
    card(
        slide,
        0.92,
        5.55,
        11.45,
        0.88,
        "Tail limitation",
        "Only seven OOF targets exceed 50 µg/m³; high-band estimates are unstable, but examples show the model can miss abrupt severe events.",
        PALE,
        15,
        13,
    )
    add_note(
        slide,
        "Error analysis",
        "Use these examples as descriptive development OOF cases only. Positive actual-minus-predicted residual means the model underpredicts. There were only seven cases over 50 µg/m³, so tail error estimates are unstable, but missing severe events is an important practical limitation.",
    )

    slide = new_slide(
        "Coverage limitations and spatial validation", "13  /  GENERALIZATION"
    )
    card(
        slide,
        0.82,
        1.55,
        5.65,
        1.4,
        "Temporal diagnostics",
        "ACF/PACF: 63 stations with ≥14 uninterrupted daily observations.\nADF/KPSS/Ljung–Box: only 6 stations with ≥30 days.",
        ICE,
        15,
        13,
    )
    card(
        slide,
        6.78,
        1.55,
        5.65,
        1.4,
        "Lead–lag diagnostics",
        "Usable pollutant pairs at six stations; 2–6 stations per lag. Correlations are descriptive, sparse and non-causal.",
        PALE,
        15,
        13,
    )
    card(
        slide,
        0.82,
        3.35,
        5.65,
        1.35,
        "Spatial-grouped pooled scores",
        "RMSE 8.700 µg/m³  |  MAE 3.287 µg/m³  |  R² 0.154  |  n=3,349",
        PALE,
        15,
        13,
    )
    card(
        slide,
        6.78,
        3.35,
        5.65,
        1.35,
        "What the spatial split means",
        "Test stations are held out, but historical inputs at those stations remain available. Not a cold-start evaluation.",
        ICE,
        15,
        13,
    )
    text_box(
        slide,
        0.91,
        5.33,
        11.3,
        0.84,
        "Irregular gaps, sparse pollutant records and a pre-holdout development cutoff limit the number of long continuous station series. Six stations cannot support basin-wide claims.",
        14,
        MUTED,
    )
    add_note(
        slide,
        "Diagnostic limits",
        "Explain that requiring uninterrupted daily segments avoids pretending missing days are consecutive. Most station series are too short for formal tests; only six meet the 30-day requirement and have adequate lead-lag pairs. Spatial validation allows history at each held-out station and therefore does not measure cold-start performance.",
    )

    slide = new_slide(
        "Statistical interpretation and data limitations", "14  /  EVIDENCE BOUNDARIES"
    )
    add_table(
        slide,
        0.82,
        1.55,
        11.6,
        2.32,
        ["Evidence", "Estimate", "Interpretation"],
        [
            [
                "Mean-fold RMSE",
                "8.520 vs 8.771",
                "CNN–LSTM vs persistence; small difference",
            ],
            [
                "Paired exact Wilcoxon",
                "p=.625; Holm=1.00",
                "No significant gain established",
            ],
            [
                "Pooled RMSE delta",
                "−1.131; 95% CI [−2.824, 0.797]",
                "72-station cluster interval includes zero",
            ],
        ],
        12,
    )
    bullet_list(
        slide,
        0.92,
        4.28,
        11.4,
        1.75,
        [
            "Five dependent folds provide low power; intervals are conditional on this dataset and do not account for all model-search uncertainty.",
            "Compiled data redistribution rights, component attribution mapping and full upstream inclusion rule remain unresolved.",
            "The historical checkpoint is missing; prediction reproduction is not binary-weight identity.",
        ],
        14,
    )
    add_note(
        slide,
        "Significance and limitations",
        "Describe the paired test and cluster-bootstrap interval with the sample limitations. The confidence interval includes zero, and only five temporal folds are available. Also disclose the data-rights caveat and missing original weights.",
    )

    slide = new_slide("Discussion: direct answers to RQ1–RQ4", "15  /  SYNTHESIS")
    answers = [
        (
            "RQ1",
            "No reliable advantage over persistence is established; RMSE is slightly lower, but MAE is higher and the paired test is nonsignificant.",
        ),
        (
            "RQ2",
            "Performance changes sharply across periods; spatial folds are conditional on historical station sequences and do not prove cold-start transfer.",
        ),
        (
            "RQ3",
            "Lag-1 carries the clearest signal. Calendar/rolling additions are not consistently useful; environmental predictors remain excluded.",
        ),
        (
            "RQ4",
            "Winter is higher on average, but extreme events are underpredicted; diagnostics cover only six stations and reuse rights remain unresolved.",
        ),
    ]
    for i, (label, answer) in enumerate(answers):
        card(
            slide,
            0.9,
            1.45 + i * 1.13,
            11.5,
            0.9,
            label,
            answer,
            ICE if i % 2 == 0 else PALE,
            15,
            12,
        )
    add_note(
        slide,
        "Research answers",
        "Answer each question explicitly. For RQ1, say no significant improvement. For RQ2, separate temporal and spatial scopes. For RQ3, identify lag-1 and qualified ablation findings. For RQ4, connect seasonality, sparse diagnostics, extreme error and unresolved rights.",
    )

    slide = new_slide("Conclusion and recommendation", "16  /  TAKEAWAY")
    text_box(
        slide,
        0.95,
        1.65,
        11.2,
        1.0,
        "The CNN–LSTM is a reasonable model to test—not a demonstrated winner.",
        27,
        NAVY,
        True,
    )
    bullet_list(
        slide,
        1.0,
        3.0,
        11.0,
        2.15,
        [
            "Keep persistence as the minimum operational benchmark.",
            "Do not deploy the CNN–LSTM as superior on current evidence.",
            "Before deployment: verify real-time availability, collect more severe-event examples, quantify uncertainty and run a prospective evaluation.",
            "Before redistribution: resolve the compiled dataset license and attribution chain.",
        ],
        17,
        gap=12,
    )
    card(
        slide,
        0.98,
        5.75,
        11.0,
        0.82,
        "Defense summary",
        "Leakage-aware one-day-ahead regression; local CNN patterns plus LSTM memory; mixed results; conservative scientific conclusion.",
        ICE,
        15,
        13,
    )
    add_note(
        slide,
        "Conclusion",
        "Conclude that the architecture is a plausible research candidate but not yet preferable to persistence for deployment. Give a clear answer to whether it is worth using: not as a claimed winner on these data. Mention the minimum evidence needed to revisit the decision.",
    )

    prs.save(OUTPUT)
    PRESENTATION.mkdir(parents=True, exist_ok=True)
    notes_text = ["# Speaker notes and oral-defense prompts", ""]
    for index, (title, note) in enumerate(slide_notes, start=1):
        notes_text.extend([f"## Slide {index}: {title}", "", note, ""])
    (PRESENTATION / "speaker_notes.md").write_text(
        "\n".join(notes_text), encoding="utf-8"
    )
    print(f"Built {OUTPUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
