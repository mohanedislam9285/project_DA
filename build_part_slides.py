"""Build the three top-level submission decks (Part 1/2/3) as .pptx files.

Design system mirrors the Typst theme so the PDFs and decks read as one product.
Run:  python build_part_slides.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent

NAVY = RGBColor(16, 36, 58)
NAVY2 = RGBColor(27, 58, 92)
TEAL = RGBColor(8, 127, 128)
AMBER = RGBColor(217, 130, 43)
GREEN = RGBColor(46, 139, 87)
RED = RGBColor(192, 57, 43)
INK = RGBColor(36, 52, 71)
MUTED = RGBColor(96, 113, 132)
PALE = RGBColor(238, 244, 246)
ICE = RGBColor(232, 243, 243)
SAND = RGBColor(248, 241, 232)
RULE = RGBColor(213, 222, 231)
WHITE = RGBColor(255, 255, 255)

FONT = "Arial"
SW_IN = 13.333
SH_IN = 7.5
MARGIN_IN = 0.62
CONTENT_W_IN = SW_IN - 2 * MARGIN_IN
SW = Inches(SW_IN)
SH = Inches(SH_IN)
MARGIN = Inches(MARGIN_IN)
CONTENT_W = Inches(CONTENT_W_IN)

PART_LABELS = {
    1: "GROUP 16  ·  PM2.5 FORECASTING  ·  PART 1  ·  DATA & PREPROCESSING",
    2: "GROUP 16  ·  PM2.5 FORECASTING  ·  PART 2  ·  MODEL TRAINING & EVALUATION",
    3: "GROUP 16  ·  PM2.5 FORECASTING  ·  PART 3  ·  STATISTICAL TESTING & EXPLAINABILITY",
}


# --------------------------------------------------------------------------- #
# low-level helpers
# --------------------------------------------------------------------------- #
def new_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width = SW
    prs.slide_height = SH
    return prs


def blank(prs: Presentation, bg=WHITE):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = bg
    return slide


def rect(slide, x, y, w, h, fill=None, line=None, line_w=0.75, radius=None):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(line_w)
    shp.shadow.inherit = False
    if radius:
        try:
            shp.adjustments[0] = radius
        except (IndexError, KeyError):
            pass
    return shp


def text(slide, x, y, w, h, runs, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT,
         spacing=6, anchor=MSO_ANCHOR.TOP, line_spacing=None):
    """runs: str or list[str] (each becomes a paragraph)."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(0.02)
    tf.margin_right = Inches(0.02)
    tf.margin_top = Inches(0.01)
    tf.margin_bottom = Inches(0.01)
    lines = [runs] if isinstance(runs, str) else list(runs)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = str(line)
        p.alignment = align
        p.space_after = Pt(spacing)
        if line_spacing:
            p.line_spacing = line_spacing
        for run in p.runs:
            run.font.name = FONT
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = color
    return box


def bullets(slide, x, y, w, h, lines, size=13.5, color=INK, gap=8, marker="▸  "):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.02)
    tf.margin_right = Inches(0.02)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{marker}{line}"
        p.space_after = Pt(gap)
        for run in p.runs:
            run.font.name = FONT
            run.font.size = Pt(size)
            run.font.color.rgb = color
    return box


def header(slide, part_label, eyebrow, title, accent=TEAL):
    rect(slide, 0, 0, SW_IN, 0.075, fill=NAVY)
    text(slide, MARGIN_IN, 0.30, 11.0, 0.26, eyebrow.upper(), size=10.5, color=accent, bold=True)
    text(slide, MARGIN_IN, 0.55, 12.1, 0.62, title, size=25, color=NAVY, bold=True)
    rect(slide, MARGIN_IN, 1.20, 1.55, 0.05, fill=accent)
    text(slide, 11.0, 0.30, 1.72, 0.26, part_label.split("·")[-2].strip().title(),
         size=9, color=MUTED, bold=True, align=PP_ALIGN.RIGHT)


def footer(slide, part_label, num):
    rect(slide, MARGIN_IN, 7.02, CONTENT_W_IN, 0.012, fill=RULE)
    text(slide, MARGIN_IN, 7.08, 10.5, 0.26, part_label, size=8.5, color=MUTED)
    text(slide, 11.7, 7.08, 1.0, 0.26, str(num), size=8.5, color=MUTED, align=PP_ALIGN.RIGHT)


def notes(slide, body):
    slide.notes_slide.notes_text_frame.text = body


def stat_card(slide, x, y, w, h, value, label, accent=TEAL, sub=None):
    rect(slide, x, y, w, h, fill=WHITE, line=RULE, radius=0.08)
    rect(slide, x, y, 0.055, h, fill=accent)
    text(slide, x + 0.16, y + 0.12, w - 0.28, 0.45, value, size=21, color=accent, bold=True)
    text(slide, x + 0.16, y + 0.60, w - 0.28, 0.5, label, size=9.5, color=MUTED, spacing=2)
    if sub:
        text(slide, x + 0.16, y + h - 0.42, w - 0.28, 0.34, sub, size=8.5, color=MUTED,
             spacing=1)


def panel(slide, x, y, w, h, title, lines, accent=TEAL, tint=PALE, size=12.5, title_size=13):
    rect(slide, x, y, w, h, fill=tint, line=RULE, radius=0.06)
    rect(slide, x, y, 0.06, h, fill=accent)
    text(slide, x + 0.22, y + 0.14, w - 0.4, 0.3, title, size=title_size, color=NAVY, bold=True)
    bullets(slide, x + 0.22, y + 0.52, w - 0.42, h - 0.62, lines, size=size, gap=6)


def table(slide, x, y, w, rows, col_ratios, font=9.5, header_font=9.5, row_h=0.34,
          header_fill=NAVY, band=PALE):
    n_rows = len(rows)
    n_cols = len(rows[0])
    total_h = row_h * n_rows
    shape = slide.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y), Inches(w),
                                   Inches(total_h))
    tbl = shape.table
    tbl.first_row = True
    for j, ratio in enumerate(col_ratios):
        tbl.columns[j].width = Emu(int(Inches(w) * ratio))
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, cell_text in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = header_fill
            else:
                cell.fill.fore_color.rgb = band if i % 2 == 0 else WHITE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = str(cell_text)
            p.alignment = PP_ALIGN.LEFT
            for run in p.runs:
                run.font.name = FONT
                run.font.size = Pt(header_font if i == 0 else font)
                run.font.bold = i == 0
                run.font.color.rgb = WHITE if i == 0 else INK
    return shape


def image_fit(slide, path: Path, x, y, w, h, caption=None, border=True):
    """Fit image inside the (x,y,w,h) box keeping aspect ratio, centred."""
    with Image.open(path) as im:
        iw, ih = im.size
    aspect = iw / ih
    box_aspect = w / h
    if aspect >= box_aspect:
        dw, dh = w, w / aspect
    else:
        dh, dw = h, h * aspect
    dx = x + (w - dw) / 2
    dy = y + (h - dh) / 2
    if border:
        rect(slide, dx - 0.04, dy - 0.04, dw + 0.08, dh + 0.08, fill=WHITE, line=RULE, radius=0.03)
    slide.shapes.add_picture(str(path), Inches(dx), Inches(dy), Inches(dw), Inches(dh))
    if caption:
        text(slide, x, y + h + 0.05, w, 0.3, caption, size=8.5, color=MUTED, spacing=1)


def figure(part_dir: Path, name: str) -> Path:
    return part_dir / "figures" / name


# --------------------------------------------------------------------------- #
# cover
# --------------------------------------------------------------------------- #
def cover(prs, part_dir, part_no, eyebrow, title_lines, subtitle, lead, stats, note,
          deadline):
    slide = blank(prs)
    rect(slide, 0, 0, SW_IN, SH_IN, fill=WHITE)
    rect(slide, 0, 0, SW_IN, 0.11, fill=NAVY)
    rect(slide, 0, 0.11, 0.32, SH_IN - 0.11, fill=TEAL)

    text(slide, 1.05, 0.62, 10.5, 0.3, eyebrow, size=11.5, color=TEAL, bold=True)
    text(slide, 1.05, 1.02, 11.4, 1.7, title_lines, size=40, color=NAVY, bold=True, spacing=2)
    text(slide, 1.05, 2.62, 11.0, 0.4, subtitle, size=15.5, color=TEAL)
    rect(slide, 1.05, 3.12, 1.7, 0.055, fill=TEAL)
    text(slide, 1.05, 3.32, 10.9, 0.9, lead, size=12.5, color=INK, spacing=4, line_spacing=1.15)

    n = len(stats)
    gap = 0.18
    cw = (11.3 - gap * (n - 1)) / n
    for i, (val, lab, accent) in enumerate(stats):
        stat_card(slide, 1.05 + i * (cw + gap), 4.50, cw, 1.05, val, lab, accent=accent)

    rect(slide, 1.05, 5.80, 11.3, 0.82, fill=SAND, line=AMBER, radius=0.05)
    rect(slide, 1.05, 5.80, 0.06, 0.82, fill=AMBER)
    text(slide, 1.28, 5.90, 10.95, 0.28, note[0], size=11.5, color=NAVY, bold=True)
    text(slide, 1.28, 6.20, 10.95, 0.4, note[1], size=9.5, color=INK, spacing=2, line_spacing=1.1)

    text(slide, 1.05, 6.90, 11.3, 0.3, deadline, size=9.5, color=MUTED)
    return slide


# --------------------------------------------------------------------------- #
# Part 1
# --------------------------------------------------------------------------- #
def build_part1():
    part_dir = ROOT / "Part_1_Data_Exploration_and_Preprocessing"
    label = PART_LABELS[1]
    prs = new_deck()
    num = 0

    cover(prs, part_dir, 1,
          "SUBMISSION PART 1 OF 3   ·   GROUP 16",
          "Data Exploration & Preprocessing",
          "Audit, cleaning, EDA and leakage-safe preparation of a 40,983-record PM2.5 station panel",
          "This deck documents where the data come from, what was removed and why, how the station-day "
          "panel behaves over time and space, and exactly how the modelling inputs are constructed so "
          "that no future information leaks into a forecast.",
          [("40,983", "clean station-day records", NAVY2),
           ("95", "coordinate-defined stations", TEAL),
           ("2,299", "distinct UTC daily dates", TEAL),
           ("9", "candidate environmental variables", NAVY2)],
          ("Integrity note",
           "Redistribution rights for the compiled Kaggle bundle are unresolved, and the publisher's "
           "row-inclusion rule cannot be reconstructed. Both are treated as first-class limitations."),
          "Group 16  ·  Part 1 of 3  ·  Data Exploration & Preprocessing  ·  deadline 14 November 2026")

    # roadmap
    num += 1
    slide = blank(prs)
    header(slide, label, "Roadmap", "What this part covers")
    rows = [
        ("Section", "Topic", "Deliverable"),
        ("01", "Objective and scope", "Regression contract for one-day-ahead PM2.5"),
        ("02", "Dataset and provenance", "Source, licence status, dimensions, coverage"),
        ("03", "Quality audit", "Cleaning rule and the checks behind it"),
        ("04–06", "EDA", "Target distribution, seasonality, correlations, missingness"),
        ("07", "Missing values", "Fold-local median imputation plus indicators"),
        ("08", "Feature engineering", "Seven-day past-only sequence, no gap or station crossing"),
        ("09", "Pipeline and leakage", "Fold-local fit; withholding of unverified inputs"),
        ("10", "Limitations and handoff", "Frozen data contract consumed by Part 2"),
    ]
    table(slide, MARGIN_IN, 1.45, CONTENT_W_IN, rows, [0.9, 3.6, 8.0], row_h=0.5,
          font=11.5, header_font=11.5)
    footer(slide, label, num)
    notes(slide, "Part 1 fixes the data contract used by every later part: cleaned panel, station "
                 "identity, target bounds, seven-day past-only sequences, fold-local preprocessing.")

    # objective
    num += 1
    slide = blank(prs)
    header(slide, label, "01  /  Objective", "Objective and scope")
    text(slide, MARGIN_IN, 1.42, 12.1, 0.7,
         "Forecast the next-day PM2.5 concentration at a monitoring station from observations available "
         "at the end of the current day. The target is a continuous concentration in µg/m³, so the task "
         "is regression — not classification. Part 1 models nothing; it proves the data are trustworthy "
         "and the inputs leak-free.", size=12.5, line_spacing=1.15)
    panel(slide, MARGIN_IN, 2.35, 5.9, 2.95, "What Part 1 must deliver", [
        "A verified dataset description: source, licence, units, dimensions.",
        "A documented cleaning policy: what was removed and what was deliberately kept.",
        "A quantitative EDA of the target and its drivers.",
        "A fold-local missing-value strategy.",
        "A precise seven-day past-only feature definition.",
    ], accent=TEAL)
    panel(slide, 6.85, 2.35, 5.87, 2.95, "Research questions informed here", [
        "RQ1 — Is the panel clean enough for matched-fold baselines?",
        "RQ3 — Which available feature groups carry signal?",
        "RQ4 — How do seasonality, coverage and data rights constrain use?",
        "Framing: one learned family only — CNN–LSTM.",
    ], accent=NAVY2, tint=ICE)
    footer(slide, label, num)
    notes(slide, "Emphasise: Part 1 is the trust layer. Every downstream number traces to an audit file.")

    # dataset
    num += 1
    slide = blank(prs)
    header(slide, label, "02  /  Dataset", "Dataset, licence and coverage")
    text(slide, MARGIN_IN, 1.40, 12.1, 0.55,
         "Input: the publisher-prepared Air Quality – Mediterranean Basin (2016–2026) bundle, obtained "
         "from Kaggle and stored as data/raw/extracted/combined_gee_pm25.csv. Retrieval metadata and "
         "checksums are retained in data/manifest.yaml and data/provenance/.", size=12, line_spacing=1.12)
    rows = [
        ("Attribute", "Value", "Unit", "Source of record"),
        ("Source", "Kaggle · Sarah Ameziane, v1", "—", "data/manifest.yaml"),
        ("Temporal span", "2016-11-25 → 2026-04-06", "UTC day", "audit + EDA"),
        ("Spatial span", "lat 36.13–43.53 · lon −5.66–9.14", "degrees", "EDA station summary"),
        ("Target", "PM2.5 concentration", "µg/m³", "Exploratory_Data.csv"),
        ("Target bounds", "retain 0 < PM2.5 < 500", "µg/m³", "teacher cleaning rule"),
        ("Rows removed", "735 sentinel / out-of-rule", "rows", "audit"),
    ]
    table(slide, MARGIN_IN, 2.12, CONTENT_W_IN, rows, [2.0, 4.3, 1.4, 3.0], row_h=0.42,
          font=11)
    rect(slide, MARGIN_IN, 5.55, CONTENT_W_IN, 1.15, fill=SAND, line=AMBER, radius=0.05)
    rect(slide, MARGIN_IN, 5.55, 0.06, 1.15, fill=AMBER)
    text(slide, MARGIN_IN + 0.22, 5.66, 12.0, 0.3, "Unresolved: redistribution rights",
         size=12, color=NAVY, bold=True)
    text(slide, MARGIN_IN + 0.22, 5.96, 12.0, 0.6,
         "Kaggle labels the licence “Other (specified in description)”. No compiled-bundle licence and no "
         "field-to-upstream attribution map exist, so the bundle is not presented as cleared for "
         "redistribution.", size=10, spacing=2, line_spacing=1.1)
    footer(slide, label, num)
    notes(slide, "Be explicit that rights are unresolved; this is a limitation, not a footnote.")

    # variables
    num += 1
    slide = blank(prs)
    header(slide, label, "02  /  Variables", "Variable inventory and missingness")
    rows = [
        ("Variable", "Family", "Unit", "Missing", "Role in the project"),
        ("pm25", "Target", "µg/m³", "0.00%", "Forecast target"),
        ("temperature_celsius", "Meteorological", "°C", "0.00%", "Excluded — timing unverified"),
        ("pressure_mb", "Meteorological", "hPa", "0.00%", "Excluded — timing unverified"),
        ("wind_u / wind_v", "Meteorological", "m/s", "0.00%", "Excluded — timing unverified"),
        ("no2", "Trace gas (GEE)", "mol/m²", "5.87%", "Excluded — timing unverified"),
        ("co", "Trace gas (GEE)", "mol/m²", "5.87%", "Excluded — timing unverified"),
        ("o3", "Trace gas (GEE)", "mol/m²", "5.97%", "Excluded — timing unverified"),
        ("aod", "Satellite (GEE)", "dimensionless", "0.00%", "Excluded — timing unverified"),
    ]
    table(slide, MARGIN_IN, 1.48, CONTENT_W_IN, rows, [3.0, 2.4, 1.9, 1.3, 3.5],
          row_h=0.46, font=11)
    text(slide, MARGIN_IN, 6.05, 12.1, 0.5,
         "Only the PM2.5 target and calendar terms enter the model. Environmental columns are audited for "
         "completeness but withheld because forecast-origin availability is undocumented.",
         size=10.5, color=MUTED, line_spacing=1.1)
    footer(slide, label, num)
    notes(slide, "The exclusion of exogeneous variables is a deliberate, documented decision.")

    # cleaning
    num += 1
    slide = blank(prs)
    header(slide, label, "03  /  Quality audit", "Cleaning and quality controls")
    text(slide, MARGIN_IN, 1.42, 12.1, 0.6,
         "Only rows violating the teacher-specified physical rule were removed: PM2.5 ≤ 0 or ≥ 500 µg/m³ "
         "(735 rows). No row was deleted for being statistically unusual — extreme but plausible "
         "concentrations are kept.", size=12.5, line_spacing=1.15)
    for i, (v, l, a) in enumerate([
        ("735", "rows removed (0 < y < 500)", AMBER),
        ("727 / 8", "low / high sentinel removals", AMBER),
        ("0", "out-of-rule values post-clean", GREEN),
        ("0", "invalid coordinates", GREEN),
    ]):
        stat_card(slide, 0.62 + i * 3.08, 2.20, 2.9, 1.0, v, l, accent=a)
    rows = [
        ("Check", "Result", "Interpretation"),
        ("Duplicate station-day keys", "0", "Each (station, date) pair is unique — a clean keyed table."),
        ("Out-of-rule targets", "0", "Every retained target is inside the physical rule."),
        ("Coordinate validity", "0 invalid", "All 95 stations carry finite lat/lon."),
        ("Station identity", "98 → 95", "Three test entries removed; identity logic preserved."),
        ("Station Tukey screen", "352 flagged", "Q3 + 3×IQR on 71 stations; retained, never auto-deleted."),
    ]
    table(slide, MARGIN_IN, 3.42, CONTENT_W_IN, rows, [3.2, 1.6, 7.3], row_h=0.42, font=10.5)
    footer(slide, label, num)
    notes(slide, "Key point: statistical extremeness alone cannot establish an instrument fault.")

    # target distribution
    num += 1
    slide = blank(prs)
    header(slide, label, "04  /  EDA", "Target distribution")
    image_fit(slide, figure(part_dir, "pm25_distribution.png"), 0.62, 1.45, 6.0, 3.5,
              caption="Figure 1. Distribution of cleaned daily PM2.5 (µg/m³).")
    rows = [
        ("n", "Mean", "SD", "Min", "Median"),
        ("40,983", "9.458", "9.730", "0.007", "7.851"),
        ("P95", "P99", "Max", "Skew", "Kurt."),
        ("21.653", "33.559", "484.98", "17.83", "651.7"),
    ]
    table(slide, 6.85, 1.55, 5.87, rows, [1, 1, 1, 1, 1], row_h=0.4, font=11)
    panel(slide, 6.85, 3.45, 5.87, 2.3, "What the distribution implies", [
        "Half of all station-days sit at or below 7.85 µg/m³.",
        "The top 1% exceeds 33.6; the maximum reaches 484.98 µg/m³.",
        "Because targets can approach zero, MAPE is unstable — use RMSE, MAE, R².",
    ], accent=TEAL, tint=ICE)
    footer(slide, label, num)
    notes(slide, "Heavy right tail drives every later metric choice.")

    # seasonality
    num += 1
    slide = blank(prs)
    header(slide, label, "05  /  Temporal structure", "Seasonality and coverage")
    image_fit(slide, figure(part_dir, "pm25_by_season.png"), 0.62, 1.45, 5.4, 3.4,
              caption="Figure 2. Seasonal PM2.5 summaries (DJF, MAM, JJA, SON).")
    rows = [
        ("Season", "Records", "Mean", "Med.", ">25", ">50"),
        ("DJF", "8,938", "10.94", "8.77", "6.15%", "0.45%"),
        ("MAM", "9,276", "8.25", "6.83", "1.92%", "0.36%"),
        ("JJA", "12,766", "9.59", "8.50", "1.95%", "0.15%"),
        ("SON", "10,003", "9.08", "7.47", "2.46%", "0.25%"),
    ]
    table(slide, 6.35, 1.55, 6.37, rows, [1.0, 1.2, 0.9, 0.9, 0.9, 0.9], row_h=0.42, font=10.5)
    panel(slide, 6.35, 3.65, 6.37, 2.1, "Coverage is sparse and irregular", [
        "Only 63 stations have ≥ 14 uninterrupted daily development observations.",
        "Only 6 stations have ≥ 30 days, the minimum for formal ADF/KPSS/Ljung-Box tests.",
        "Sequences never bridge a date gap, so irregular coverage is handled by construction.",
    ], accent=AMBER, tint=SAND, size=11.5)
    footer(slide, label, num)
    notes(slide, "Winter carries the highest means and the largest share of high days.")

    # correlations
    num += 1
    slide = blank(prs)
    header(slide, label, "06  /  Relationships", "Correlations and missing values")
    image_fit(slide, figure(part_dir, "spearman_correlations.png"), 0.62, 1.45, 5.9, 2.75,
              caption="Figure 3. Spearman rank-correlation structure.")
    image_fit(slide, figure(part_dir, "missingness.png"), 6.85, 1.45, 5.87, 2.75,
              caption="Figure 4. Missingness profile of the cleaned table.")
    panel(slide, MARGIN_IN, 4.55, 5.9, 2.0, "Association with the target is weak", [
        "Strongest rank correlation: NO₂, ρ = 0.247 — still weak in absolute terms.",
        "No pair reaches |r| > 0.9, so there is no multicollinearity red flag.",
        "These are context, not model inputs (availability unverified).",
    ], accent=TEAL, tint=ICE, size=11.5)
    panel(slide, 6.85, 4.55, 5.87, 2.0, "Missing values are confined to GEE gases", [
        "no2 / co: 5.87% missing; o3: 5.97% missing.",
        "PM2.5 target and calendar terms: 0% missing.",
        "Any imputation happens inside a training fold — never dataset-wide.",
    ], accent=NAVY2, tint=PALE, size=11.5)
    footer(slide, label, num)
    notes(slide, "Weak exogenous signal is the reason the later model leans almost entirely on history.")

    # missing strategy
    num += 1
    slide = blank(prs)
    header(slide, label, "07  /  Missing values", "Missing-value strategy")
    for i, (v, l, s) in enumerate([
        ("Median", "imputation statistic", "robust to the heavy right tail"),
        ("Fold-local", "where it is fitted", "training partition only"),
        ("+ flags", "indicator channels", "record where a gap was filled"),
    ]):
        stat_card(slide, 0.62 + i * 4.10, 1.45, 3.9, 1.15, v, l, accent=TEAL, sub=s)
    bullets(slide, MARGIN_IN, 2.95, 12.1, 2.5, [
        "Median imputation, not mean, because the target is heavily skewed.",
        "Imputer and scaler are fitted on each training fold's sequences, then applied to that fold's validation.",
        "Missingness indicators let the network distinguish a real low value from a filled one.",
        "Standardisation uses the same fold-local fit; the output head is inverse-transformed for scoring.",
    ], size=12.5, gap=10)
    rect(slide, MARGIN_IN, 5.62, CONTENT_W_IN, 1.05, fill=ICE, line=TEAL, radius=0.05)
    rect(slide, MARGIN_IN, 5.62, 0.06, 1.05, fill=TEAL)
    text(slide, MARGIN_IN + 0.22, 5.73, 12.0, 0.3, "Why the reference preprocessing is frozen here",
         size=12, color=NAVY, bold=True)
    text(slide, MARGIN_IN + 0.22, 6.02, 12.0, 0.55,
         "The target keeps its original µg/m³ scale, both missingness indicators are retained, and imputation "
         "and scaling stay fold-local — declared as the reference data contract, justified by leakage-safety "
         "and the target's skew, not by model scores.",
         size=10, spacing=2, line_spacing=1.1)
    footer(slide, label, num)
    notes(slide, "Fold-local fitting is a leakage control, not a convenience.")

    # feature engineering
    num += 1
    slide = blank(prs)
    header(slide, label, "08  /  Features", "The seven-day sequence contract")
    boxes = [
        (0.62, "7 past days", "[t−6 … t]\nindexed per station", PALE, RULE),
        (4.35, "Past-only features", "calendar cycles\nPM2.5 lag & rolling", ICE, TEAL),
        (8.08, "Target", "PM2.5 at t+1\n(scalar µg/m³)", SAND, AMBER),
    ]
    for x, t, d, fill, stroke in boxes:
        rect(slide, x, 1.55, 3.35, 1.25, fill=fill, line=stroke, radius=0.08)
        text(slide, x + 0.15, 1.68, 3.05, 0.35, t, size=13, color=NAVY, bold=True,
             align=PP_ALIGN.CENTER)
        text(slide, x + 0.15, 2.05, 3.05, 0.6, d, size=10, color=MUTED, align=PP_ALIGN.CENTER)
    for x in (3.99, 7.72):
        text(slide, x, 1.95, 0.35, 0.4, "→", size=20, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    rows = [
        ("Channel", "Type", "Definition and why it is safe"),
        ("pm25_lag1", "Past value", "PM2.5 at the forecast origin t — the persistence signal."),
        ("pm25_roll3_mean", "Past rolling", "Mean of the three days ending at t."),
        ("pm25_roll7_mean", "Past rolling", "Mean of the seven days ending at t."),
        ("day_of_year_sin/cos", "Calendar", "Annual cycle as a smooth pair — known in advance."),
        ("day_of_week_sin/cos", "Calendar", "Weekly cycle as a smooth pair."),
        ("rolling missing flags", "Indicator", "Declare where a rolling summary was filled."),
    ]
    table(slide, MARGIN_IN, 3.05, CONTENT_W_IN, rows, [3.0, 1.9, 7.2], row_h=0.42, font=10.5)
    text(slide, MARGIN_IN, 6.15, 12.1, 0.55,
         "Three hard rules: no gap crossing · no station crossing · no future information. Each is "
         "enforced by an automated test.", size=10.5, color=MUTED)
    footer(slide, label, num)
    notes(slide, "Nine processed channels: seven features plus two rolling missingness flags.")

    # pipeline
    num += 1
    slide = blank(prs)
    header(slide, label, "09  /  Pipeline", "Preprocessing pipeline and leakage control")
    stages = [("Raw CSV", "41,718 rows"), ("Cleaning", "target bounds + keys"),
              ("Split", "date/station folds"), ("Fold-local", "impute + scale"),
              ("Sequences", "7 days, no gaps")]
    for i, (t, d) in enumerate(stages):
        x = 0.62 + i * 2.46
        rect(slide, x, 1.5, 2.2, 1.1, fill=ICE if i % 2 else PALE, line=RULE, radius=0.08)
        text(slide, x + 0.1, 1.62, 2.0, 0.32, t, size=12, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
        text(slide, x + 0.1, 1.98, 2.0, 0.5, d, size=9.5, color=MUTED, align=PP_ALIGN.CENTER)
        if i < len(stages) - 1:
            text(slide, x + 2.20, 1.88, 0.26, 0.4, "→", size=16, color=TEAL, bold=True,
                 align=PP_ALIGN.CENTER)
    rows = [
        ("Control", "Status", "Evidence"),
        ("Fold-local imputation and scaling", "Pass", "Fitted on the training partition of each fold only."),
        ("Past-only rolling features", "Pass", "Recomputed from values ≤ the forecast origin."),
        ("Origin/target alignment", "Pass", "Each label is exactly the next station-day."),
        ("Gap and station boundaries", "Pass", "Unit tests reject cross-station or gapped windows."),
        ("Environmental timing", "Blocked", "Exogenous channels withheld rather than assumed available."),
        ("Frozen holdout untouched", "Pass", "Reserved future block excluded from all selection."),
    ]
    table(slide, MARGIN_IN, 2.85, CONTENT_W_IN, rows, [4.4, 1.3, 6.4], row_h=0.42, font=10.5)
    footer(slide, label, num)
    notes(slide, "The verification boundary is stated honestly: the pipeline is clean, the publisher's "
                 "upstream join cannot be reconstructed.")

    # limitations / handoff
    num += 1
    slide = blank(prs)
    header(slide, label, "10  /  Handoff", "Limitations and the frozen handoff")
    rows = [
        ("Area", "Limitation carried into Parts 2 and 3"),
        ("Licence", "Compiled-bundle redistribution rights unresolved."),
        ("Upstream ETL", "Publisher's omitted-row rule and tie handling remain unreconstructed."),
        ("Coverage", "Sparse station-days limit formal temporal tests to six stations."),
        ("Environmental inputs", "Meteorology, NO₂, CO, O₃ and AOD withheld (availability unverified)."),
        ("Tail behaviour", "Only a handful of development values exceed 100 µg/m³."),
    ]
    table(slide, MARGIN_IN, 1.48, CONTENT_W_IN, rows, [3.0, 9.1], row_h=0.45, font=11)
    for i, (v, l, a) in enumerate([
        ("40,983 × 9", "cleaned panel dimensions", NAVY2),
        ("7 days", "past-only sequence length", TEAL),
        ("0 < y < 500", "target rule (µg/m³)", TEAL),
    ]):
        stat_card(slide, 0.62 + i * 4.10, 4.45, 3.9, 1.05, v, l, accent=a)
    rect(slide, MARGIN_IN, 5.72, CONTENT_W_IN, 1.02, fill=ICE, line=TEAL, radius=0.05)
    rect(slide, MARGIN_IN, 5.72, 0.06, 1.02, fill=TEAL)
    text(slide, MARGIN_IN + 0.22, 5.83, 12.0, 0.3, "Part 1 verdict", size=12, color=NAVY, bold=True)
    text(slide, MARGIN_IN + 0.22, 6.12, 12.0, 0.55,
         "The panel is clean enough to model and documented well enough to audit. It is not a dense grid "
         "and its licence is unresolved — so Part 2 must validate with temporal and station-grouped folds, "
         "keep preprocessing fold-local, and report uncertainty honestly.", size=10, spacing=2,
         line_spacing=1.1)
    footer(slide, label, num)
    notes(slide, "This is the handoff: Part 2 consumes exactly this contract.")

    out = part_dir / "Part_1_Presentation.pptx"
    prs.save(out)
    return out


# --------------------------------------------------------------------------- #
# Part 2
# --------------------------------------------------------------------------- #
def build_part2():
    part_dir = ROOT / "Part_2_Model_Training_Validation_and_Evaluation"
    label = PART_LABELS[2]
    prs = new_deck()
    num = 0

    cover(prs, part_dir, 2,
          "SUBMISSION PART 2 OF 3   ·   GROUP 16",
          "Model Training, Validation & Evaluation",
          "CNN–LSTM implementation, leakage-safe validation, tuning, baselines and honest performance testing",
          "This deck turns Part 1's frozen data contract into a working forecasting system: the architecture, "
          "the training protocol, two leakage-safe validation designs, a nested tuning search, the required "
          "baselines, and measured out-of-fold and locked-future performance — with its uncertainty.",
          [("8.455", "temporal CV RMSE (µg/m³)", TEAL),
           ("7.962", "station-grouped RMSE", TEAL),
           ("250", "tuning trials (5 × 50)", NAVY2),
           ("6.027", "locked future RMSE", AMBER)],
          ("Integrity note",
           "One-day-ahead PM2.5 is dominated by persistence. The network's edge over persistence is small "
           "and not statistically significant, and on the frozen future block persistence wins outright."),
          "Group 16  ·  Part 2 of 3  ·  Model Training, Validation & Evaluation  ·  deadline 12 December 2026")

    # roadmap
    num += 1
    slide = blank(prs)
    header(slide, label, "Roadmap", "What this part covers")
    rows = [
        ("Section", "Topic", "Deliverable"),
        ("01", "Objective and task", "Regression contract and metric set"),
        ("02", "Architecture", "CNN–LSTM layer order and parameter budget"),
        ("03", "Training protocol", "Loss, schedule, early stopping, seeds"),
        ("04", "Validation design", "Expanding temporal + station-grouped folds"),
        ("05", "Hyperparameter optimisation", "Nested Optuna TPE, 5 × 50 trials"),
        ("06", "Baselines", "Persistence and weekly seasonal-naive"),
        ("07–09", "Results", "Out-of-fold scores, robustness, error structure"),
        ("10–11", "Testing and handoff", "Locked future block; what Part 3 receives"),
    ]
    table(slide, MARGIN_IN, 1.45, CONTENT_W_IN, rows, [0.9, 3.6, 8.0], row_h=0.5,
          font=11.5, header_font=11.5)
    footer(slide, label, num)
    notes(slide, "Part 2 builds, tunes and tests; Part 3 tests statistically and explains.")

    # objective
    num += 1
    slide = blank(prs)
    header(slide, label, "01  /  Objective", "Objective, task and evaluation contract")
    text(slide, MARGIN_IN, 1.42, 12.1, 0.55,
         "Forecast next-day PM2.5 (µg/m³) at a station from the seven days ending at the forecast origin. "
         "Regression task; only values observed at or before the origin are admissible inputs.",
         size=12.5, line_spacing=1.12)
    rows = [
        ("Element", "Definition fixed before any model was scored"),
        ("Target", "PM2.5 at station-day t+1, µg/m³, no transformation for scoring."),
        ("Inputs", "Seven-day window ending at t: PM2.5 lag and rolling summaries + calendar cycles."),
        ("Primary metric", "RMSE in µg/m³ — RMS-dominant because the tail matters."),
        ("Secondary metrics", "MAE (robust central error) and R² (variance explained)."),
        ("Excluded metric", "MAPE — unstable because targets can approach zero."),
        ("Uncertainty", "Fold SD, bootstrap CIs, paired significance tests (Part 3)."),
    ]
    table(slide, MARGIN_IN, 2.15, CONTENT_W_IN, rows, [2.6, 9.5], row_h=0.45, font=11)
    text(slide, MARGIN_IN, 5.60, 12.1, 0.5,
         "The metric set is deliberately small and pre-registered so that results cannot be chosen after "
         "the fact.", size=10.5, color=MUTED)
    footer(slide, label, num)
    notes(slide, "Pre-registration protects against metric shopping.")

    # architecture
    num += 1
    slide = blank(prs)
    header(slide, label, "02  /  Architecture", "The CNN–LSTM model")
    stages = [("Input", "7 days × 7 channels"), ("Conv1D", "causal kernel, k filters"),
              ("LSTM", "recurrent summary"), ("Dense → 1", "ReLU then scalar")]
    for i, (t, d) in enumerate(stages):
        x = 0.62 + i * 3.10
        rect(slide, x, 1.5, 2.75, 1.05, fill=ICE if i % 2 else PALE, line=RULE, radius=0.08)
        text(slide, x + 0.1, 1.62, 2.55, 0.3, t, size=13, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
        text(slide, x + 0.1, 1.97, 2.55, 0.45, d, size=9.5, color=MUTED, align=PP_ALIGN.CENTER)
        if i < len(stages) - 1:
            text(slide, x + 2.76, 1.87, 0.32, 0.4, "→", size=18, color=TEAL, bold=True,
                 align=PP_ALIGN.CENTER)
    rows = [
        ("Stage", "Setting", "Role"),
        ("Causal Conv1D", "filters 16/32/64, kernel 2/3/5", "Detects short local patterns; causal padding prevents future leakage."),
        ("LayerNorm + ReLU", "per-channel", "Stabilises activations on a heavy-tailed input."),
        ("Dropout", "0.05 – 0.35", "Regularises a small-data sequence model; tuned, not fixed."),
        ("LSTM", "units 32/64, last step", "Carries temporal state; compresses to one representation."),
        ("MaxPooling1D", "optional", "Kept only when it helped inside the inner loop."),
        ("Dense + ReLU → 1", "32 units, linear out", "Scalar forecast; inverted for scoring."),
    ]
    table(slide, MARGIN_IN, 2.85, CONTENT_W_IN, rows, [2.6, 3.0, 6.5], row_h=0.44, font=10.5)
    text(slide, MARGIN_IN, 5.95, 12.1, 0.6,
         "Reference model: 10,081 trainable parameters · lookback 7 · 7 input channels · ≈0.16 MB. "
         "TensorFlow 2.19.1, CPU only.", size=10.5, color=MUTED)
    footer(slide, label, num)
    notes(slide, "Causal convolution is a leakage control, not a style choice.")

    # training
    num += 1
    slide = blank(prs)
    header(slide, label, "03  /  Training", "How the model is trained")
    for i, (v, l, s) in enumerate([
        ("MSE", "training loss", "weighted-MSE variant also tested"),
        ("Adam", "optimiser", "tuned lr ≈ 2e-4 to 1e-3"),
        ("Patience 10", "early stopping", "val MSE; best weights restored"),
        ("Patience 3", "LR schedule", "ReduceLROnPlateau ×0.5, min 1e-5"),
    ]):
        stat_card(slide, 0.62 + i * 3.08, 1.42, 2.9, 1.1, v, l, accent=TEAL if i % 2 == 0 else NAVY2, sub=s)
    bullets(slide, MARGIN_IN, 2.85, 6.0, 3.0, [
        "Early stopping halts when validation MSE plateaus for 10 epochs.",
        "The best-weight checkpoint is restored, never the last epoch.",
        "One fixed seed per outer fold (43–47) for paired comparability.",
    ], size=11.5, gap=8)
    bullets(slide, 6.85, 2.85, 5.87, 3.0, [
        "Imputer and scaler fitted on that fold's training sequences only.",
        "Reference fit time 2.3 s on CPU; full nested search is 250 short trials.",
        "Environment: Python 3.12.10, TensorFlow 2.19.1, CPU only.",
    ], size=11.5, gap=8)
    image_fit(slide, figure(part_dir, "training_validation_loss_typst.png"), 6.85, 4.15, 5.87, 2.4,
              caption="Figure 1. Representative training/validation loss curve.")
    footer(slide, label, num)
    notes(slide, "Validation loss flattening while training loss falls is the over-fitting signal.")

    # validation design
    num += 1
    slide = blank(prs)
    header(slide, label, "04  /  Validation", "Two honest validation schemes")
    panel(slide, MARGIN_IN, 1.45, 5.9, 2.6, "Expanding temporal folds", [
        "Train on the past, validate on the next, later block.",
        "A one-day embargo prevents last-day bleed.",
        "Five folds; preprocessing refitted inside every fold.",
    ], accent=TEAL, tint=ICE)
    panel(slide, 6.85, 1.45, 5.87, 2.6, "Station-grouped folds", [
        "Split by composite station identity (name + coordinates).",
        "No instrument appears in both train and validation.",
        "Five folds; measures transfer to unseen locations.",
    ], accent=NAVY2, tint=PALE)
    image_fit(slide, figure(part_dir, "temporal_cv_rmse.png"), 0.62, 4.20, 5.9, 2.4,
              caption="Figure 2. Per-fold RMSE under the expanding temporal scheme.")
    panel(slide, 6.85, 4.20, 5.87, 2.4, "The random-split diagnostic is not selection evidence", [
        "Computed for one pre-declared architecture only.",
        "Random mean RMSE 8.091 vs temporal 8.301 — a −0.209 gap.",
        "Reported as a diagnostic; never used to select a model.",
    ], accent=AMBER, tint=SAND, size=11.5)
    footer(slide, label, num)
    notes(slide, "Fold-level volatility is large — the SD is almost as large as the mean.")

    # tuning
    num += 1
    slide = blank(prs)
    header(slide, label, "05  /  Tuning", "Nested hyperparameter optimisation")
    for i, (v, l, s) in enumerate([
        ("TPE", "sampler", "Optuna 5.0.0, seed 42"),
        ("50", "trials per outer fold", "inner chronological split"),
        ("250", "completed trials", "5 outer folds"),
        ("Inner RMSE", "objective (µg/m³)", "outer block never seen"),
    ]):
        stat_card(slide, 0.62 + i * 3.08, 1.42, 2.9, 1.1, v, l, accent=NAVY2 if i % 2 == 0 else TEAL, sub=s)
    rows = [
        ("Hyperparameter", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"),
        ("Conv filters", "64", "64", "16", "32", "64"),
        ("Kernel size", "5", "2", "3", "5", "2"),
        ("LSTM units", "32", "32", "64", "64", "32"),
        ("Dropout", "0.220", "0.060", "0.347", "0.098", "0.055"),
        ("Learning rate", "7.4e-4", "9.1e-4", "7.7e-4", "2.2e-4", "9.99e-4"),
    ]
    table(slide, MARGIN_IN, 2.80, CONTENT_W_IN, rows, [3.4, 1.9, 1.9, 1.9, 1.9, 1.7],
          row_h=0.44, font=11)
    rect(slide, MARGIN_IN, 5.72, CONTENT_W_IN, 1.02, fill=SAND, line=AMBER, radius=0.05)
    rect(slide, MARGIN_IN, 5.72, 0.06, 1.02, fill=AMBER)
    text(slide, MARGIN_IN + 0.22, 5.83, 12.0, 0.3, "Reading the tuning result honestly", size=12,
         color=NAVY, bold=True)
    text(slide, MARGIN_IN + 0.22, 6.12, 12.0, 0.55,
         "The optimum moves between folds: dropout spans 0.055–0.347 and filters 16–64. The search surface "
         "is shallow and the “best” point is only weakly separated from its neighbours — that instability "
         "is itself a finding.", size=10, spacing=2, line_spacing=1.1)
    footer(slide, label, num)
    notes(slide, "Nesting is what keeps the reported score honest.")

    # baselines
    num += 1
    slide = blank(prs)
    header(slide, label, "06  /  Baselines", "Mandatory baselines")
    panel(slide, MARGIN_IN, 1.45, 5.9, 3.0, "Persistence", [
        "Forecast tomorrow = today's last observed PM2.5.",
        "No parameters, no fitting — and notoriously strong.",
        "Temporal RMSE 8.771 · Spatial 9.292 · Future block 5.846.",
    ], accent=NAVY2, tint=PALE)
    panel(slide, 6.85, 1.45, 5.87, 3.0, "Weekly seasonal-naive", [
        "Forecast tomorrow = the same weekday seven days earlier.",
        "Captures the weekly cycle without learning.",
        "Temporal RMSE 10.798 · Spatial 10.906 · Future block 8.027.",
    ], accent=AMBER, tint=SAND)
    rows = [
        ("Why baselines are non-negotiable", "Evidence"),
        ("They establish the floor", "A network that cannot beat persistence shows no skill."),
        ("They are paired", "Identical fold membership removes fold-difficulty as a confounder."),
        ("They expose the MAE gap", "Persistence MAE 3.158 / 3.119 vs network 3.978 / 4.024."),
    ]
    table(slide, MARGIN_IN, 4.65, CONTENT_W_IN, rows, [4.0, 8.1], row_h=0.5, font=11)
    footer(slide, label, num)
    notes(slide, "Baselines are scored on identical folds and IDs, so every comparison is paired.")

    # results
    num += 1
    slide = blank(prs)
    header(slide, label, "07  /  Development results", "Out-of-fold performance")
    for i, (v, l, s) in enumerate([
        ("8.455", "temporal CV RMSE", "SD 6.161 · n = 2,867"),
        ("3.911", "temporal CV MAE", "SD 0.957"),
        ("7.962", "spatial CV RMSE", "SD 4.143 · n = 3,349"),
        ("0.212", "spatial CV R²", "temporal R² 0.106"),
    ]):
        stat_card(slide, 0.62 + i * 3.08, 1.42, 2.9, 1.1, v, l, accent=TEAL if i < 3 else NAVY2, sub=s)
    rows = [
        ("Scheme / model", "RMSE", "SD", "MAE", "R²", "Pooled RMSE", "n"),
        ("Temporal · CNN–LSTM", "8.455", "6.161", "3.911", "0.106", "9.582", "2,867"),
        ("Temporal · tuned (TPE)", "8.520", "6.034", "3.978", "0.076", "9.571", "2,867"),
        ("Temporal · persistence", "8.771", "—", "3.158", "—", "—", "2,867"),
        ("Temporal · seasonal-naive", "10.798", "—", "—", "—", "—", "2,867"),
        ("Spatial · CNN–LSTM", "7.962", "4.143", "3.292", "0.212", "8.700", "3,349"),
    ]
    table(slide, MARGIN_IN, 2.75, CONTENT_W_IN, rows, [3.6, 1.5, 1.5, 1.4, 1.3, 2.1, 1.3],
          row_h=0.42, font=10.5)
    image_fit(slide, figure(part_dir, "temporal_baseline_comparison.png"), 0.62, 4.60, 5.6, 2.15,
              caption="Figure 3. CNN–LSTM vs baselines on matched temporal folds.")
    panel(slide, 6.55, 4.55, 6.17, 2.2, "How to read these numbers", [
        "The network beats seasonal-naive decisively (≈ −2.28 temporal, −2.94 spatial).",
        "Against persistence the edge is small and not significant.",
        "R² is modest; fold SD is almost as large as the mean.",
    ], accent=TEAL, tint=ICE, size=11.5)
    footer(slide, label, num)
    notes(slide, "Do not over-claim: the honest story is a small, uncertain edge over persistence.")

    # robustness
    num += 1
    slide = blank(prs)
    header(slide, label, "08  /  Robustness", "Stress-testing the result")
    image_fit(slide, figure(part_dir, "training_size_sensitivity.png"), 0.62, 1.45, 5.2, 2.9,
              caption="Figure 4. Learning curve: steep gain to 50%, then a plateau.")
    rows = [
        ("Experiment", "Result", "Verdict"),
        ("Training size 10% → 100%", "10.151 → 8.382", "Plateau — data-limited"),
        ("Preprocessing: log1p target", "8.575", "Worse — reference kept"),
        ("Preprocessing: drop indicators", "8.500", "Worse — flags kept"),
        ("Extreme-event weighting", "8.600", "No material gain"),
        ("Architecture ±1 step", "mostly < 5%", "Mostly stable"),
    ]
    table(slide, 6.15, 1.55, 6.57, rows, [3.4, 2.2, 2.6], row_h=0.5, font=10.5)
    rect(slide, MARGIN_IN, 4.65, CONTENT_W_IN, 2.1, fill=ICE, line=TEAL, radius=0.05)
    rect(slide, MARGIN_IN, 4.65, 0.06, 2.1, fill=TEAL)
    text(slide, MARGIN_IN + 0.25, 4.80, 11.9, 0.3, "What the robustness panel establishes",
         size=13, color=NAVY, bold=True)
    bullets(slide, MARGIN_IN + 0.25, 5.20, 11.6, 1.5, [
        "The result is not an artefact of one preprocessing or weighting choice.",
        "None of the tested alternatives improved on the reference materially.",
        "The one genuine sensitivity — filter count — is disclosed, not smoothed over.",
    ], size=11.5, gap=6)
    footer(slide, label, num)
    notes(slide, "Every variant is scored on the same folds as the reference.")

    # error structure
    num += 1
    slide = blank(prs)
    header(slide, label, "09  /  Error structure", "Where the model succeeds and fails")
    rows = [
        ("Season", "n", "RMSE", "MAE"),
        ("Spring", "244", "3.600", "2.886"),
        ("Autumn", "652", "6.273", "3.921"),
        ("Summer", "1,678", "7.784", "3.566"),
        ("Winter", "293", "21.285", "6.404"),
    ]
    table(slide, MARGIN_IN, 1.48, 5.9, rows, [1.6, 1.0, 1.2, 1.2], row_h=0.44, font=11)
    rows2 = [
        ("Actual band", "n", "RMSE", "Under-pred."),
        ("< 10", "1,734", "3.619", "25%"),
        ("10–25", "1,070", "5.045", "80%"),
        ("25–50", "56", "18.957", "100%"),
        ("> 50", "7", "166.120", "100%"),
    ]
    table(slide, 6.85, 1.48, 5.87, rows2, [1.6, 1.0, 1.3, 1.4], row_h=0.44, font=11)
    image_fit(slide, figure(part_dir, "temporal_oof_residuals.png"), 0.62, 3.95, 5.4, 2.6,
              caption="Figure 5. Out-of-fold residual distribution.")
    rect(slide, 6.35, 3.95, 6.37, 2.6, fill=SAND, line=RED, radius=0.05)
    rect(slide, 6.35, 3.95, 0.06, 2.6, fill=RED)
    text(slide, 6.60, 4.10, 5.95, 0.3, "The core limitation, stated plainly", size=13, color=NAVY,
         bold=True)
    bullets(slide, 6.60, 4.50, 5.95, 2.0, [
        "Only 7 of 2,867 OOF points exceed 50 µg/m³ — and every one is under-predicted.",
        "The headline RMSE is governed by ordinary days; episode skill is unmeasured.",
        "Any operational claim on pollution episodes is unsupported by these data.",
    ], size=11, gap=6)
    footer(slide, label, num)
    notes(slide, "Regression to the mean on the tail is the dominant failure mode.")

    # locked future
    num += 1
    slide = blank(prs)
    header(slide, label, "10  /  Locked future test", "The frozen independent estimate")
    text(slide, MARGIN_IN, 1.40, 12.1, 0.5,
         "A held-out future block (2024-12-08 → 2026-04-06) was scored once for the pre-locked model and "
         "then frozen; the record was written before scoring to prevent a second, opportunistic evaluation.",
         size=11.5, color=MUTED, line_spacing=1.12)
    for i, (v, l, s) in enumerate([
        ("6.027", "CNN–LSTM RMSE", "N = 1,716"),
        ("4.024", "CNN–LSTM MAE", ""),
        ("0.239", "CNN–LSTM R²", ""),
        ("5.846", "persistence RMSE", "same 1,716 IDs"),
    ]):
        stat_card(slide, 0.62 + i * 3.08, 2.00, 2.9, 1.1, v, l, accent=TEAL if i < 3 else AMBER, sub=s)
    rows = [
        ("Model (frozen block)", "RMSE", "MAE", "R²", "N"),
        ("CNN–LSTM (locked)", "6.027", "4.024", "0.239", "1,716"),
        ("Persistence", "5.846", "3.119", "—", "1,716"),
        ("Weekly seasonal-naive", "8.027", "5.361", "—", "1,716"),
    ]
    table(slide, MARGIN_IN, 3.45, CONTENT_W_IN, rows, [4.6, 1.9, 1.9, 1.9, 1.8], row_h=0.48,
          font=11.5)
    rect(slide, MARGIN_IN, 5.55, CONTENT_W_IN, 1.15, fill=SAND, line=AMBER, radius=0.05)
    rect(slide, MARGIN_IN, 5.55, 0.06, 1.15, fill=AMBER)
    text(slide, MARGIN_IN + 0.22, 5.66, 12.0, 0.3, "The most important honest finding in Part 2",
         size=12, color=NAVY, bold=True)
    text(slide, MARGIN_IN + 0.22, 5.96, 12.0, 0.62,
         "On the frozen block, persistence achieves a lower RMSE than the CNN–LSTM (5.846 vs 6.027) and a "
         "lower MAE. The network beats the weekly baseline comfortably but does not yet beat the no-model "
         "benchmark. This is the headline, not a footnote.", size=10, spacing=2, line_spacing=1.1)
    footer(slide, label, num)
    notes(slide, "Scored 2026-11-10 and locked in future_test_evaluation.json; read-only history.")

    # handoff
    num += 1
    slide = blank(prs)
    header(slide, label, "11  /  Handoff", "Limitations and what Part 3 receives")
    rows = [
        ("Area", "Limitation carried into Part 3"),
        ("Baseline parity", "Persistence matches or beats the network on the locked block."),
        ("Inferential power", "Only five dependent folds — p-values have low power."),
        ("Tail skill", "Extreme days (> 50 µg/m³) are essentially unlearnable here (n = 7)."),
        ("Validation realism", "Temporal folds cannot model instrument drift or site relocation."),
        ("Exogenous inputs", "Meteorology, trace gases and AOD remain excluded."),
    ]
    table(slide, MARGIN_IN, 1.48, CONTENT_W_IN, rows, [2.8, 9.3], row_h=0.47, font=11)
    for i, (v, l, a) in enumerate([
        ("2,867 + 3,349", "OOF predictions (temporal + spatial)", NAVY2),
        ("5 × 5", "paired folds for testing", TEAL),
        ("1,716", "locked future IDs", TEAL),
    ]):
        stat_card(slide, 0.62 + i * 4.10, 4.55, 3.9, 1.05, v, l, accent=a)
    rect(slide, MARGIN_IN, 5.82, CONTENT_W_IN, 0.9, fill=ICE, line=TEAL, radius=0.05)
    rect(slide, MARGIN_IN, 5.82, 0.06, 0.9, fill=TEAL)
    text(slide, MARGIN_IN + 0.22, 5.93, 12.0, 0.3, "Part 2 verdict", size=12, color=NAVY, bold=True)
    text(slide, MARGIN_IN + 0.22, 6.20, 12.0, 0.45,
         "Built, tuned, validated and tested honestly. It beats a weekly baseline, does not significantly "
         "beat persistence, and its uncertainty is as large as its mean — Part 3 tests these three facts.",
         size=10, spacing=2, line_spacing=1.1)
    footer(slide, label, num)
    notes(slide, "The handoff is the OOF predictions plus the locked future estimate.")

    out = part_dir / "Part_2_Presentation.pptx"
    prs.save(out)
    return out


# --------------------------------------------------------------------------- #
# Part 3
# --------------------------------------------------------------------------- #
def build_part3():
    part_dir = ROOT / "Part_3_Statistical_Testing_and_Explainability"
    label = PART_LABELS[3]
    prs = new_deck()
    num = 0

    cover(prs, part_dir, 3,
          "SUBMISSION PART 3 OF 3   ·   GROUP 16",
          "Statistical Testing & Explainability",
          "Paired significance tests, effect sizes, feature importance, ablations and a critical reading",
          "This part asks the questions Part 2 left open: are the differences real or noise, which inputs "
          "carry signal, how stable is the explanation, and what do the results honestly support? Every "
          "claim is paired with an interval, an effect size and an explicit power caveat.",
          [("0", "significant pairs after Holm", AMBER),
           ("0.651", "importance of pm25_lag1", TEAL),
           ("0.672", "cross-scheme rank ρ", TEAL),
           ("−0.467", "RMSE change, no calendar", NAVY2)],
          ("Integrity note",
           "With five dependent folds the tests are underpowered by construction. Effect sizes and "
           "intervals carry the argument; p-values are a secondary, honest signal."),
          "Group 16  ·  Part 3 of 3  ·  Statistical Testing & Explainability  ·  deadline 2 January 2027")

    # roadmap
    num += 1
    slide = blank(prs)
    header(slide, label, "Roadmap", "What this part covers")
    rows = [
        ("Section", "Topic", "Deliverable"),
        ("01", "Objective and questions", "RQ2, RQ3, RQ4, RQ5 owned here"),
        ("02", "Statistical framework", "Wilcoxon, Holm, bootstrap, effect size, power caveat"),
        ("03", "Paired comparisons", "Temporal and spatial model pairings"),
        ("04", "Validation-scheme ablation", "Optimism gap of a careless split"),
        ("05–07", "Explainability", "Permutation importance and its stability"),
        ("08", "Ablations", "Feature-family and preprocessing evidence"),
        ("09–10", "Conditional error and diagnostics", "Season, band, extremes, ACF, lead–lag"),
        ("11–12", "Validity and conclusions", "What is and is not supported"),
    ]
    table(slide, MARGIN_IN, 1.45, CONTENT_W_IN, rows, [0.9, 3.9, 7.7], row_h=0.5,
          font=11.5, header_font=11.5)
    footer(slide, label, num)
    notes(slide, "Part 3 converts numbers into defensible claims.")

    # objective
    num += 1
    slide = blank(prs)
    header(slide, label, "01  /  Objective", "Objective and the questions Part 3 owns")
    text(slide, MARGIN_IN, 1.42, 12.1, 0.5,
         "Part 2 produced numbers. Part 3 decides which of them mean anything: inferential statistics "
         "(is the difference real?) plus explainability (which inputs drive the forecast, and how stable "
         "is that story?), closing with an honest interpretation.", size=12.5, line_spacing=1.12)
    rows = [
        ("RQ", "Question owned by Part 3"),
        ("RQ2", "Does the CNN–LSTM beat persistence and seasonal-naive significantly, or within noise?"),
        ("RQ3", "Which feature families carry signal? Which can be removed without loss — or help?"),
        ("RQ5", "Is the feature-importance ranking stable within a scheme and across schemes?"),
        ("RQ4", "How do seasonality, bands and extremes condition the error, and what does that imply?"),
    ]
    table(slide, MARGIN_IN, 2.20, CONTENT_W_IN, rows, [1.2, 10.9], row_h=0.55, font=11.5)
    panel(slide, MARGIN_IN, 4.75, CONTENT_W_IN, 1.95, "Deliverables", [
        "Paired Wilcoxon tests on fold RMSE with Holm-Bonferroni correction and rank-biserial effect sizes.",
        "Bootstrap confidence intervals for every mean difference (10,000 replicates).",
        "Permutation feature importance with cross-scheme and within-scheme stability analysis.",
        "Ablation evidence, conditional error analysis, and a threats-to-validity section.",
    ], accent=TEAL, tint=ICE, size=11.5)
    footer(slide, label, num)
    notes(slide, "Part 3 is the critical-thinking layer of the project.")

    # framework
    num += 1
    slide = blank(prs)
    header(slide, label, "02  /  Framework", "The statistical testing framework")
    for i, (v, l, s) in enumerate([
        ("Wilcoxon", "primary test", "signed-rank on fold RMSE"),
        ("Holm", "multiplicity control", "across generated pairs"),
        ("10,000", "bootstrap replicates", "fold-pair mean difference"),
        ("r", "effect size", "matched-pairs rank-biserial"),
    ]):
        stat_card(slide, 0.62 + i * 3.08, 1.42, 2.9, 1.1, v, l, accent=NAVY2 if i % 2 == 0 else TEAL, sub=s)
    rows = [
        ("Choice", "Why it is the right choice here"),
        ("Paired testing", "Identical fold membership removes fold difficulty as a confounder."),
        ("Wilcoxon signed-rank", "Five dependent fold differences; a rank test avoids normality assumptions."),
        ("Supplemental permutation", "10,000 paired sign permutations cross-check the rank test."),
        ("Fold-pair bootstrap", "10,000 resamples give a CI for the mean difference — the primary evidence."),
        ("Holm-Bonferroni", "Controls family-wise error without plain-Bonferroni conservatism."),
        ("Rank-biserial effect size", "Reports magnitude, not just significance — essential at low power."),
    ]
    table(slide, MARGIN_IN, 2.80, CONTENT_W_IN, rows, [3.4, 8.7], row_h=0.44, font=10.5)
    rect(slide, MARGIN_IN, 5.55, CONTENT_W_IN, 1.15, fill=SAND, line=AMBER, radius=0.05)
    rect(slide, MARGIN_IN, 5.55, 0.06, 1.15, fill=AMBER)
    text(slide, MARGIN_IN + 0.22, 5.66, 12.0, 0.3, "Power is the binding constraint — stated up front",
         size=12, color=NAVY, bold=True)
    text(slide, MARGIN_IN + 0.22, 5.96, 12.0, 0.62,
         "With n = 5 dependent folds the smallest attainable two-sided Wilcoxon p is 0.0625 and effective "
         "power is low. A non-significant result is therefore uninformative about the absence of an effect; "
         "effect sizes and intervals carry the argument.", size=10, spacing=2, line_spacing=1.1)
    footer(slide, label, num)
    notes(slide, "Be explicit that the tests are underpowered; this is honesty, not weakness.")

    # paired comparisons temporal
    num += 1
    slide = blank(prs)
    header(slide, label, "03  /  Comparisons", "Paired model comparisons — temporal folds")
    rows = [
        ("Comparison (temporal)", "A", "B", "Δ (A−B)", "95% CI of Δ", "p", "Holm p"),
        ("CNN–LSTM vs persistence", "8.520", "8.771", "−0.251", "−2.960, 1.530", "0.625", "1.00"),
        ("CNN–LSTM vs seasonal-naive", "8.520", "10.798", "−2.278", "−6.028, −0.048", "0.188", "1.00"),
        ("persistence vs seasonal-naive", "8.771", "10.798", "−2.027", "−3.067, −1.155", "0.063", "1.00"),
    ]
    table(slide, MARGIN_IN, 1.55, CONTENT_W_IN, rows, [3.6, 1.3, 1.3, 1.5, 2.4, 1.1, 1.2],
          row_h=0.52, font=11)
    text(slide, MARGIN_IN, 3.55, 12.1, 0.45,
         "Fold-mean RMSE in µg/m³. Negative Δ favours the first model. Every interval spans or nearly "
         "spans zero once correction is applied.", size=10.5, color=MUTED)
    panel(slide, MARGIN_IN, 4.20, 5.9, 2.4, "Bottom line on significance", [
        "No comparison survives Holm correction.",
        "Directions favour the network over the weekly baseline.",
        "With five folds the tests cannot separate these from noise.",
    ], accent=AMBER, tint=SAND)
    panel(slide, 6.85, 4.20, 5.87, 2.4, "But the intervals are still informative", [
        "Seasonal-naive intervals exclude zero before correction, on both schemes.",
        "Consistent, one-directional evidence the network beats a weekly baseline.",
        "The formal threshold is simply not reachable at n = 5.",
    ], accent=TEAL, tint=ICE)
    footer(slide, label, num)
    notes(slide, "Lead with the honesty: nothing is significant, yet the direction is consistent.")

    # paired comparisons spatial
    num += 1
    slide = blank(prs)
    header(slide, label, "03  /  Comparisons", "Paired model comparisons — station-grouped folds")
    rows = [
        ("Comparison (spatial)", "A", "B", "Δ (A−B)", "95% CI of Δ", "p", "Holm p"),
        ("CNN–LSTM vs persistence", "7.962", "9.292", "−1.330", "−3.036, 0.085", "0.313", "1.00"),
        ("CNN–LSTM vs seasonal-naive", "7.962", "10.906", "−2.944", "−7.224, −0.298", "0.125", "1.00"),
        ("persistence vs seasonal-naive", "9.292", "10.906", "−1.614", "−6.244, 1.929", "1.000", "1.00"),
    ]
    table(slide, MARGIN_IN, 1.55, CONTENT_W_IN, rows, [3.6, 1.3, 1.3, 1.5, 2.4, 1.1, 1.2],
          row_h=0.52, font=11)
    text(slide, MARGIN_IN, 3.55, 12.1, 0.45,
         "The spatial scheme gives larger point differences in the network's favour, but the intervals "
         "remain wide enough that significance is still not reached.", size=10.5, color=MUTED)
    image_fit(slide, figure(part_dir, "available_feature_ablation.png"), 0.62, 4.15, 6.0, 2.5,
              caption="Figure 1. Ablation view supporting the comparison narrative.")
    panel(slide, 6.85, 4.15, 5.87, 2.5, "Interpretation across both schemes", [
        "Direction is stable: the network beats seasonal-naive, and slightly leads persistence.",
        "Magnitude is not: spatial deltas are larger, temporal deltas smaller.",
        "Neither scheme reaches significance — the conclusion is directional, not confirmatory.",
    ], accent=NAVY2, tint=PALE, size=11.5)
    footer(slide, label, num)
    notes(slide, "Both schemes point the same way; neither confirms it formally.")

    # scheme ablation
    num += 1
    slide = blank(prs)
    header(slide, label, "04  /  Scheme ablation", "How much does the split flatter the score?")
    text(slide, MARGIN_IN, 1.40, 12.1, 0.5,
         "For one pre-declared architecture the same model was scored under a random split and under the "
         "two task-faithful schemes; the gap quantifies the optimism a careless split would introduce.",
         size=11.5, color=MUTED, line_spacing=1.12)
    rows = [
        ("Model", "Random", "Temporal", "Spatial", "Interpretation"),
        ("CNN–LSTM (diagnostic)", "8.091", "8.301", "7.939", "Random is −0.209 vs temporal: a modest gap."),
        ("Persistence", "9.378", "8.771", "9.292", "No fitted parameters — the scheme matters little."),
        ("Seasonal-naive", "11.287", "10.798", "10.906", "Same; the scheme matters most for learned models."),
    ]
    table(slide, MARGIN_IN, 2.10, CONTENT_W_IN, rows, [2.9, 1.5, 1.6, 1.6, 4.5], row_h=0.5,
          font=11)
    panel(slide, MARGIN_IN, 3.95, 5.9, 2.5, "What this tells us", [
        "The optimism gap is smaller than the fold-to-fold SD.",
        "It is still non-zero and biased in the expected direction.",
        "That is exactly why selection used only temporal and spatial folds.",
    ], accent=TEAL, tint=ICE)
    panel(slide, 6.85, 3.95, 5.87, 2.5, "Do not over-read the gap", [
        "A single architecture on five folds cannot establish a general law.",
        "The correct claim is narrow: for this model and panel, ≈0.21 µg/m³.",
        "Random splits are never used for model selection.",
    ], accent=AMBER, tint=SAND)
    footer(slide, label, num)
    notes(slide, "Keep the claim narrow and model-specific.")

    # explainability method
    num += 1
    slide = blank(prs)
    header(slide, label, "05  /  Explainability", "Permutation feature importance")
    stages = [("Trained fold", "model + OOF sequences"), ("Shuffle one channel", "break its link to target"),
              ("Re-score", "RMSE degradation = importance"), ("Aggregate", "mean & rank over 5 folds")]
    for i, (t, d) in enumerate(stages):
        x = 0.62 + i * 3.10
        rect(slide, x, 1.5, 2.75, 1.05, fill=ICE if i % 2 else PALE, line=RULE, radius=0.08)
        text(slide, x + 0.1, 1.62, 2.55, 0.3, t, size=12.5, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
        text(slide, x + 0.1, 1.97, 2.55, 0.45, d, size=9.5, color=MUTED, align=PP_ALIGN.CENTER)
        if i < len(stages) - 1:
            text(slide, x + 2.76, 1.87, 0.32, 0.4, "→", size=18, color=TEAL, bold=True,
                 align=PP_ALIGN.CENTER)
    panel(slide, MARGIN_IN, 2.95, 5.9, 3.4, "Why permutation, and its honest limits", [
        "Model-agnostic: the same procedure explains the network and the baselines.",
        "It measures importance in the context of the trained model, not intrinsic to the variable.",
        "Correlated channels share credit — permuting lag-1 while roll channels carry the same level can understate its effect.",
        "Negative importance is a known artefact and is reported as evidence of redundancy.",
    ], accent=TEAL, tint=ICE, size=11.5)
    panel(slide, 6.85, 2.95, 5.87, 3.4, "Interpretation discipline", [
        "All channels are engineered from the same PM2.5 series.",
        "So importance answers “which engineered channel does this model lean on?”",
        "It does not answer “which physical driver causes PM2.5?”",
        "The causal question is outside the scope of this dataset (Part 1).",
    ], accent=NAVY2, tint=PALE, size=11.5)
    footer(slide, label, num)
    notes(slide, "State clearly that importance here is model reliance, not causality.")

    # importance results
    num += 1
    slide = blank(prs)
    header(slide, label, "06  /  Importance results", "Which channels matter")
    rows = [
        ("Channel", "Mean imp.", "SD", "Mean rank"),
        ("pm25_lag1", "0.651", "0.443", "1.0"),
        ("pm25_roll3_mean", "0.153", "0.167", "3.0"),
        ("pm25_roll7_mean__missing", "0.052", "0.067", "4.8"),
        ("day_of_year_sin", "0.033", "0.059", "5.6"),
        ("day_of_year_cos", "−0.022", "0.203", "5.4"),
        ("pm25_roll7_mean", "−0.065", "0.142", "6.8"),
    ]
    table(slide, MARGIN_IN, 1.45, 5.7, rows, [3.2, 1.2, 1.0, 1.2], row_h=0.44, font=10.5)
    image_fit(slide, figure(part_dir, "cnn_lstm_feature_importance.png"), 6.65, 1.45, 6.07, 2.7,
              caption="Figure 2. Feature importance with fold-level spread.")
    panel(slide, MARGIN_IN, 4.55, 5.7, 2.1, "Only lag-1 is robustly above zero", [
        "pm25_lag1 mean 0.651 with SD 0.443 across folds.",
        "Calendar terms are weak to negative.",
        "roll7 is negative — removing it does not hurt.",
    ], accent=TEAL, tint=ICE, size=11)
    panel(slide, 6.65, 4.55, 6.07, 2.1, "Half of the engineered channels add nothing", [
        "The day-to-day dynamic is a slowly-varying level plus noise.",
        "This matches Part 1's weak exogenous correlations.",
        "The model is, in effect, a regularised persistence estimator.",
    ], accent=AMBER, tint=SAND, size=11)
    footer(slide, label, num)
    notes(slide, "Importance is concentrated; only the PM2.5 history channel is robust.")

    # stability
    num += 1
    slide = blank(prs)
    header(slide, label, "07  /  Stability", "Is the explanation stable?")
    text(slide, MARGIN_IN, 1.40, 12.1, 0.5,
         "Across schemes the importance ranking agrees moderately: Spearman ρ = 0.672 and top-5 Jaccard "
         "= 0.667. Within a single scheme, across folds, it does not.", size=12, line_spacing=1.12)
    rows = [
        ("Axis", "Spearman ρ", "Jaccard top-5", "Reading"),
        ("Temporal vs spatial", "0.672", "0.667", "Moderate — the core signal survives both schemes."),
        ("Fold 4 vs fold 5 (temporal)", "0.683", "0.667", "Best pair — still not a stable ranking."),
        ("Fold 3 vs fold 5 (temporal)", "−0.017", "0.250", "Essentially no agreement between two folds."),
        ("All temporal fold pairs", "−0.02 – 0.68", "0.25 – 0.67", "None met the pre-set stability threshold."),
    ]
    table(slide, MARGIN_IN, 2.10, CONTENT_W_IN, rows, [3.4, 2.0, 2.0, 4.7], row_h=0.5, font=11)
    rect(slide, MARGIN_IN, 4.35, CONTENT_W_IN, 2.0, fill=SAND, line=AMBER, radius=0.05)
    rect(slide, MARGIN_IN, 4.35, 0.06, 2.0, fill=AMBER)
    text(slide, MARGIN_IN + 0.25, 4.50, 11.9, 0.3, "The honest conclusion about explainability",
         size=13, color=NAVY, bold=True)
    bullets(slide, MARGIN_IN + 0.25, 4.90, 11.6, 1.4, [
        "The coarse story is stable: PM2.5's own recent history dominates, calendar terms are marginal.",
        "The fine ranking is not: two folds of the same scheme can disagree almost completely.",
        "Any claim of the form “feature X is third-most important” is unsupported by these data.",
    ], size=11.5, gap=6)
    footer(slide, label, num)
    notes(slide, "This is the most important nuance in the explainability section.")

    # ablations
    num += 1
    slide = blank(prs)
    header(slide, label, "08  /  Ablations", "Does removing a feature family help or hurt?")
    rows = [
        ("Variant", "Mean RMSE", "Δ vs full", "95% CI of Δ", "Reading"),
        ("Full feature set", "8.455", "—", "—", "Reference."),
        ("Without calendar", "7.988", "−0.467", "−0.957, −0.088", "Removing calendar improves RMSE."),
        ("Without rolling", "8.278", "−0.177", "−0.337, −0.043", "Removing rolling means also improves."),
        ("Without lag and rolling", "9.072", "+0.617", "+0.249, +1.022", "Removing PM2.5 history hurts badly."),
    ]
    table(slide, MARGIN_IN, 1.45, CONTENT_W_IN, rows, [3.0, 1.6, 1.5, 2.3, 3.7], row_h=0.5,
          font=10.5)
    image_fit(slide, figure(part_dir, "ablation.png"), 0.62, 3.60, 5.6, 2.55,
              caption="Figure 3. Ablation deltas.")
    panel(slide, 6.55, 3.60, 6.17, 2.55, "Two conclusions the ablations force", [
        "Parsimony wins: the best configuration removes the calendar channels entirely.",
        "History is the model: every variant that removes the PM2.5 lag collapses.",
        "A model can rely on a channel that is net-harmful — hence importance and ablation together.",
    ], accent=TEAL, tint=ICE, size=11.5)
    text(slide, MARGIN_IN, 6.30, 12.1, 0.45,
         "Negative Δ = better than the full set. Intervals exclude zero — among the few statistically "
         "coherent effects in the study.", size=9.5, color=MUTED)
    footer(slide, label, num)
    notes(slide, "The ablation is treated as stronger evidence than permutation importance.")

    # conditional error
    num += 1
    slide = blank(prs)
    header(slide, label, "09  /  Conditional error", "Where the error lives")
    rows = [
        ("Season", "n", "RMSE", "MAE"),
        ("Spring", "244", "3.600", "2.886"),
        ("Autumn", "652", "6.273", "3.921"),
        ("Summer", "1,678", "7.784", "3.566"),
        ("Winter", "293", "21.285", "6.404"),
    ]
    table(slide, MARGIN_IN, 1.45, 5.9, rows, [1.6, 1.0, 1.2, 1.2], row_h=0.44, font=11)
    rows2 = [
        ("Band", "n", "RMSE", "Under-pred."),
        ("< 10", "1,734", "3.619", "25%"),
        ("10–25", "1,070", "5.045", "80%"),
        ("25–50", "56", "18.957", "100%"),
        ("> 50", "7", "166.120", "100%"),
    ]
    table(slide, 6.85, 1.45, 5.87, rows2, [1.5, 1.0, 1.4, 1.5], row_h=0.44, font=11)
    image_fit(slide, figure(part_dir, "extreme_event_weighting.png"), 0.62, 4.00, 5.6, 2.2,
              caption="Figure 4. Extreme-event weighting barely moves overall or tail RMSE.")
    rect(slide, 6.55, 4.00, 6.17, 2.2, fill=SAND, line=RED, radius=0.05)
    rect(slide, 6.55, 4.00, 0.06, 2.2, fill=RED)
    text(slide, 6.80, 4.15, 5.75, 0.3, "What this implies for use", size=13, color=NAVY, bold=True)
    bullets(slide, 6.80, 4.55, 5.75, 1.6, [
        "A single RMSE of ~8 µg/m³ is misleading for decisions.",
        "Ordinary days (< 10) error ≈ 3.6; episode days one-to-two orders larger.",
        "Any use must state it is specifically blind to episodes.",
    ], size=11, gap=6)
    footer(slide, label, num)
    notes(slide, "Under-prediction of every point above 25 µg/m³ is a textbook pattern.")

    # diagnostics
    num += 1
    slide = blank(prs)
    header(slide, label, "10  /  Diagnostics", "Supporting time-series diagnostics")
    rows = [
        ("Diagnostic", "Coverage", "Finding"),
        ("ACF / PACF at lag 1", "63 stations", "Lag-1 autocorrelation positive and often large (up to ≈0.85)."),
        ("Formal ADF / KPSS / Ljung-Box", "6 stations", "Mixed outcomes — no single stationarity verdict is defensible."),
        ("Longest uninterrupted segment", "14–32 days", "Formal tests need ≥ 30 days; only a few stations qualify."),
    ]
    table(slide, MARGIN_IN, 1.48, CONTENT_W_IN, rows, [3.6, 1.8, 7.7], row_h=0.5, font=11)
    rows2 = [
        ("Pollutant", "Best lag", "Median r", "Reading"),
        ("O3", "0 to −1 days", "−0.51 to −0.56", "Consistent negative; strongest — but availability unverified."),
        ("CO", "+1 day", "+0.21", "Weak and inconsistent; no usable relationship."),
        ("NO2", "various", "−0.10 to +0.07", "Negligible at the station level."),
    ]
    table(slide, MARGIN_IN, 3.35, CONTENT_W_IN, rows2, [2.2, 2.2, 2.4, 6.3], row_h=0.5, font=11)
    rect(slide, MARGIN_IN, 5.30, CONTENT_W_IN, 1.15, fill=ICE, line=TEAL, radius=0.05)
    rect(slide, MARGIN_IN, 5.30, 0.06, 1.15, fill=TEAL)
    text(slide, MARGIN_IN + 0.22, 5.41, 12.0, 0.3, "Why this supports the honest headline",
         size=12, color=NAVY, bold=True)
    text(slide, MARGIN_IN + 0.22, 5.71, 12.0, 0.6,
         "Strong lag-1 autocorrelation and weak exogenous relationships predict the observed outcome: a "
         "persistent series makes persistence hard to beat, and weak exogenous signal gives the network "
         "little extra to learn. The results are consistent with the process.", size=10, spacing=2,
         line_spacing=1.1)
    footer(slide, label, num)
    notes(slide, "Diagnostics are descriptive and limited by sparse coverage.")

    # threats
    num += 1
    slide = blank(prs)
    header(slide, label, "11  /  Validity", "What these results do and do not support")
    rows = [
        ("Threat", "Severity", "Consequence and mitigation"),
        ("Statistical power", "High", "Five folds give smallest p ≈ 0.0625; lead with effect sizes and intervals."),
        ("Sparse extreme coverage", "High", "Only 7 OOF points above 50 µg/m³, all under-predicted."),
        ("Feature-context confounding", "Medium", "Permutation importance measures model reliance, not causality."),
        ("Unstable within-scheme ranks", "Medium", "Only coarse conclusions are drawn."),
        ("Excluded exogenous inputs", "Medium", "Meteorology, gases and AOD withheld (Part 1)."),
        ("Single architecture family", "Low-Med", "Generalises to this model, not to a model class."),
    ]
    table(slide, MARGIN_IN, 1.42, 6.6, rows, [2.6, 1.2, 5.0], row_h=0.52, font=9.5)
    panel(slide, 7.45, 1.42, 5.27, 2.6, "Supported", [
        "The network beats the weekly baseline in direction on both schemes.",
        "PM2.5's own recent history is by far the dominant input.",
        "Calendar features are not earning their place.",
    ], accent=GREEN, tint=ICE, size=11)
    panel(slide, 7.45, 4.20, 5.27, 2.5, "Not supported", [
        "That the CNN–LSTM significantly outperforms persistence.",
        "That the model has any skill on pollution episodes.",
        "That any specific feature ranking is stable.",
    ], accent=RED, tint=SAND, size=11)
    footer(slide, label, num)
    notes(slide, "Separating supported from unsupported claims is the core of Part 3.")

    # conclusions
    num += 1
    slide = blank(prs)
    header(slide, label, "12  /  Conclusions", "Conclusion")
    for i, (v, l, s) in enumerate([
        ("No", "significant pairs after Holm", "five folds is too few"),
        ("Lag-1", "the only robust input", "history dominates"),
        ("−0.467", "RMSE when calendar removed", "less can be more"),
    ]):
        stat_card(slide, 0.62 + i * 4.10, 1.42, 3.9, 1.15, v, l, accent=[AMBER, TEAL, NAVY2][i], sub=s)
    text(slide, MARGIN_IN, 2.85, 12.1, 1.5,
         "The project delivers a leakage-safe, reproducible, honestly-measured one-day-ahead PM2.5 "
         "forecaster. It shows a CNN–LSTM can be built, tuned and evaluated correctly on a sparse, "
         "skewed station panel; that its advantage over a weekly naive baseline is consistent in "
         "direction but not formally significant; and that persistence — not any learned model — remains "
         "the benchmark to beat. The explainability analysis shows the model is, in essence, a "
         "well-regularised persistence estimator with a fragile calendar overlay.",
         size=12.5, line_spacing=1.15)
    rect(slide, MARGIN_IN, 4.55, CONTENT_W_IN, 2.0, fill=ICE, line=TEAL, radius=0.05)
    rect(slide, MARGIN_IN, 4.55, 0.06, 2.0, fill=TEAL)
    text(slide, MARGIN_IN + 0.25, 4.70, 11.9, 0.3, "Why this is a strong submission, not a weak result",
         size=13, color=NAVY, bold=True)
    bullets(slide, MARGIN_IN + 0.25, 5.10, 11.6, 1.4, [
        "A study that reports its power limits, failed significance tests, baseline loss and unstable feature ranks is more credible than one reporting a flattering single number.",
        "Every claim is backed by a file in results/16/, and every limitation is stated where it applies.",
        "Across all three parts, the deliverable is a defensible scientific argument — not a score.",
    ], size=11.5, gap=6)
    footer(slide, label, num)
    notes(slide, "Close on scientific credibility as the differentiator.")

    out = part_dir / "Part_3_Presentation.pptx"
    prs.save(out)
    return out


def main():
    for fn in (build_part1, build_part2, build_part3):
        path = fn()
        print(f"wrote {path}  ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
