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

ROOT = Path(__file__).resolve().parents[1]

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
    text(slide, x + 0.16, y + 0.10, w - 0.28, 0.42, value, size=19 if sub else 21,
         color=accent, bold=True)
    text(slide, x + 0.16, y + 0.53, w - 0.28, 0.26, label, size=9.2, color=MUTED, spacing=2)
    if sub:
        text(slide, x + 0.16, y + 0.76, w - 0.28, 0.26, sub, size=8.2, color=MUTED, spacing=1)


def panel(slide, x, y, w, h, title, lines, accent=TEAL, tint=PALE, size=12.5, title_size=13,
          gap=6):
    rect(slide, x, y, w, h, fill=tint, line=RULE, radius=0.06)
    rect(slide, x, y, 0.06, h, fill=accent)
    text(slide, x + 0.22, y + 0.14, w - 0.4, 0.3, title, size=title_size, color=NAVY, bold=True)
    bullets(slide, x + 0.22, y + 0.52, w - 0.42, h - 0.62, lines, size=size, gap=gap)


def table(slide, x, y, w, rows, col_ratios, font=9.5, header_font=9.5, row_h=0.34,
          header_fill=NAVY, band=PALE):
    n_rows = len(rows)
    n_cols = len(rows[0])
    total_h = row_h * n_rows
    shape = slide.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y), Inches(w),
                                   Inches(total_h))
    tbl = shape.table
    tbl.first_row = True
    ratio_sum = sum(col_ratios)
    for j, ratio in enumerate(col_ratios):
        tbl.columns[j].width = Emu(int(Inches(w) * ratio / ratio_sum))
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
    return ROOT / "figures" / name


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


# =========================================================================== #
# Part 1 - 12-slide submission deck (2026-11-14 redesign)
# =========================================================================== #
P1_ACCENTS = ["0E7C86", "E2604F", "D99A2B", "6C5CE7", "1FA97A", "D9458C", "2E6FA7", "3B4C8C"]
P1_TOTAL = 12


def _rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


def _tint(h: str, f: float) -> RGBColor:
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return RGBColor(int(r + (255 - r) * f), int(g + (255 - g) * f), int(b + (255 - b) * f))


def _alpha(shape, pct: float) -> None:
    from pptx.oxml.ns import qn

    solid = shape.fill.fore_color._xFill
    clr = solid.find(qn("a:srgbClr"))
    clr.append(clr.makeelement(qn("a:alpha"), {"val": str(int(pct * 1000))}))


def _grad(slide, x, y, w, h, c1, c2, angle=0, radius=None):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    shp.line.fill.background()
    shp.shadow.inherit = False
    try:
        shp.fill.gradient()
        shp.fill.gradient_angle = angle
        st = shp.fill.gradient_stops
        st[0].position = 0.0
        st[0].color.rgb = _rgb(c1)
        st[1].position = 1.0
        st[1].color.rgb = _rgb(c2)
    except Exception:
        shp.fill.solid()
        shp.fill.fore_color.rgb = _rgb(c1)
    if radius:
        try:
            shp.adjustments[0] = radius
        except (IndexError, KeyError):
            pass
    return shp


def _bg_grad(slide, c1, c2, angle=45):
    fill = slide.background.fill
    try:
        fill.gradient()
        fill.gradient_angle = angle
        st = fill.gradient_stops
        st[0].position = 0.0
        st[0].color.rgb = _rgb(c1)
        st[1].position = 1.0
        st[1].color.rgb = _rgb(c2)
    except Exception:
        fill.solid()
        fill.fore_color.rgb = _rgb(c1)


def _blob(slide, x, y, w, h, color, alpha=20):
    shp = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.line.fill.background()
    shp.shadow.inherit = False
    shp.fill.solid()
    shp.fill.fore_color.rgb = _rgb(color)
    _alpha(shp, alpha)


def _card(slide, x, y, w, h, accent, fill_f=0.93, line_f=0.72, radius=0.05):
    rect(slide, x, y, w, h, fill=_tint(accent, fill_f), line=_tint(accent, line_f), radius=radius)
    rect(slide, x, y, 0.065, h, fill=_rgb(accent))


def _pill(slide, x, y, w, h, label, fill, fg=WHITE, size=9, bold=True):
    rect(slide, x, y, w, h, fill=_rgb(fill), radius=0.5)
    text(slide, x, y, w, h, label, size=size, color=fg, bold=bold,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def _p1_header(slide, num, kicker, title, accent, ghost):
    _grad(slide, 0, 0, SW_IN, 0.085, "0B1F3A", "6C5CE7", angle=0)
    _pill(slide, 0.62, 0.30, 0.62, 0.28, f"{num:02d}", accent, size=9.5)
    text(slide, 1.38, 0.325, 8.6, 0.26, kicker.upper(), size=9.5, color=_rgb(accent), bold=True)
    text(slide, 0.62, 0.66, 10.05, 0.60, title, size=26, color=NAVY, bold=True)
    text(slide, 10.70, 0.18, 2.0, 0.85, ghost, size=54, color=RGBColor(233, 239, 247),
         bold=True, align=PP_ALIGN.RIGHT)
    _grad(slide, 0.62, 1.27, 1.75, 0.055, accent, "1FA97A", angle=0)


def _p1_footer(slide, idx, accent):
    rect(slide, 0.62, 7.02, CONTENT_W_IN, 0.01, fill=RULE)
    text(slide, 0.62, 7.08, 7.4, 0.22, PART_LABELS[1], size=7.8, color=MUTED)
    spacing = 0.155
    start = 11.25 - (P1_TOTAL - 1) * spacing
    for i in range(P1_TOTAL):
        current = i == idx - 1
        rect(slide, start + i * spacing, 7.13, 0.10 if current else 0.055, 0.055,
             fill=_rgb(accent) if current else RGBColor(214, 222, 232), radius=0.5)
    text(slide, 11.45, 7.05, 0.65, 0.24, f"{idx:02d}", size=9, color=_rgb(accent),
         bold=True, align=PP_ALIGN.RIGHT)


def _tile(slide, x, y, w, h, value, label, accent, sub=None, val_size=18):
    _card(slide, x, y, w, h, accent, fill_f=0.92, line_f=0.76, radius=0.08)
    text(slide, x + 0.17, y + 0.08, w - 0.26, 0.34, value, size=val_size,
         color=_rgb(accent), bold=True)
    text(slide, x + 0.17, y + 0.42, w - 0.26, 0.22, label, size=8.2, color=MUTED)
    if sub:
        text(slide, x + 0.17, y + 0.63, w - 0.26, 0.20, sub, size=6.9, color=MUTED)


def _fig(slide, part_dir, name, x, y, w, h, caption=None, accent=None):
    path = ROOT / "figures" / name
    if not path.exists():
        path = part_dir / "figures" / name
    rect(slide, x, y, w, h, fill=WHITE, line=RULE, radius=0.04)
    image_fit(slide, path, x + 0.06, y + 0.06, w - 0.12, h - 0.12, border=False)
    if caption:
        text(slide, x, y + h + 0.04, w, 0.22, caption, size=7.6, color=MUTED)


def build_part1():
    part_dir = ROOT / "reports" / "part_1"
    prs = new_deck()

    # ------------------------------------------------------------ 1. cover --
    slide = blank(prs)
    _bg_grad(slide, "081A33", "252C63", angle=40)
    _blob(slide, 9.3, -1.7, 6.0, 6.0, "0FA6A6", 26)
    _blob(slide, 10.9, 3.2, 5.2, 5.2, "E2604F", 24)
    _blob(slide, 7.6, 5.0, 3.9, 3.9, "E0A33B", 20)

    text(slide, 0.9, 0.80, 8.0, 0.28, "SUBMISSION PART 1 OF 3   ·   GROUP 16", size=11,
         color=RGBColor(126, 227, 214), bold=True)
    text(slide, 0.9, 1.22, 8.6, 1.75, ["Data Exploration &", "Preprocessing"], size=42,
         color=WHITE, bold=True, spacing=4, line_spacing=1.0)
    text(slide, 0.9, 3.02, 7.5, 0.75,
         "Audit, cleaning, EDA and leakage-safe preparation of a 40,983-record PM2.5 station panel",
         size=15, color=RGBColor(178, 231, 230), line_spacing=1.1)
    _grad(slide, 0.9, 3.92, 2.0, 0.06, "34D3C0", "6C5CE7", angle=0)
    text(slide, 0.9, 4.16, 7.1, 0.85,
         "This deck documents where the data come from, what was removed and why, how the station-day "
         "panel behaves over time and space, and how the modelling inputs are built so that no future "
         "information leaks into a forecast.",
         size=10.8, color=RGBColor(214, 226, 240), line_spacing=1.18)

    stats = [("40,983", "clean station-day records"), ("95", "coordinate-defined stations"),
             ("2,299", "distinct UTC days"), ("8 + 1", "environmental variables + target")]
    for i, (val, lab) in enumerate(stats):
        x = 0.9 + i * 2.92
        chip = rect(slide, x, 5.16, 2.72, 0.98, fill=RGBColor(255, 255, 255),
                    line=RGBColor(120, 150, 190), radius=0.08)
        _alpha(chip, 9)
        text(slide, x + 0.16, 5.26, 2.4, 0.40, val, size=21,
             color=RGBColor(140, 236, 224), bold=True)
        text(slide, x + 0.16, 5.69, 2.45, 0.24, lab, size=8.2,
             color=RGBColor(206, 219, 235))

    box = rect(slide, 0.9, 6.36, 11.48, 0.62, fill=RGBColor(43, 48, 78), radius=0.05)
    rect(slide, 0.9, 6.36, 0.07, 0.62, fill=_rgb("E0A33B"))
    text(slide, 1.14, 6.42, 11.1, 0.22, "Integrity note", size=9.2,
         color=_rgb("F2C274"), bold=True)
    text(slide, 1.14, 6.63, 11.1, 0.28,
         "Redistribution rights are unresolved and the publisher's upstream row-inclusion rule cannot "
         "be reconstructed. Both are treated as first-class limitations.",
         size=8.2, color=RGBColor(214, 224, 238))
    text(slide, 0.9, 7.10, 9.0, 0.22,
         "Group 16  ·  Part 1 of 3  ·  Data Exploration & Preprocessing  ·  deadline 14 November 2026",
         size=8.2, color=RGBColor(150, 170, 200))

    # ------------------------------------------------------------ 2. agenda --
    slide = blank(prs)
    _p1_header(slide, 2, "Roadmap", "What this part covers", "6C5CE7", "02")
    text(slide, 0.62, 1.42, 12.05, 0.34,
         "Ten sections: two set the data contract, eight carry the analysis and the handoff.",
         size=11, color=MUTED)
    agenda = [
        ("03", "Objective and scope", "Regression contract · RQs 1, 3 and 4"),
        ("04", "Dataset, licence and coverage", "Source, dimensions, provenance audit"),
        ("05", "Target distribution", "Histogram, boxplot, extreme concentrations"),
        ("06", "Exploratory data analysis", "AOD, trace gases and meteorology"),
        ("07", "Correlation analysis", "Matrix and predictor relationships"),
        ("08", "Temporal analysis", "Seasons, months and temporal dependence"),
        ("09", "Data cleaning", "Invalid observations, missing values, duplicates, outliers"),
        ("10", "Preprocessing pipeline", "Imputation, scaling, feature engineering"),
        ("11", "Leakage prevention and validation", "Train-only transforms, temporal and spatial splits"),
        ("12", "Conclusion and next steps", "Findings, limitations, next development stage"),
    ]
    for i, (num, title, sub) in enumerate(agenda):
        col, row = i % 2, i // 2
        x = 0.62 + col * 6.16
        y = 1.88 + row * 0.93
        accent = P1_ACCENTS[i % len(P1_ACCENTS)]
        _card(slide, x, y, 5.93, 0.82, accent, fill_f=0.94, line_f=0.78, radius=0.08)
        rect(slide, x + 0.13, y + 0.13, 0.56, 0.56, fill=_rgb(accent), radius=0.18)
        text(slide, x + 0.13, y + 0.20, 0.56, 0.38, num, size=12.5, color=WHITE,
             bold=True, align=PP_ALIGN.CENTER)
        text(slide, x + 0.82, y + 0.12, 4.95, 0.28, title, size=11.2, color=NAVY, bold=True)
        text(slide, x + 0.82, y + 0.44, 4.95, 0.24, sub, size=8.4, color=MUTED)
    _p1_footer(slide, 2, "6C5CE7")

    # --------------------------------------------------------- 3. objective --
    slide = blank(prs)
    _p1_header(slide, 3, "Objective", "Objective and scope", "0E7C86", "03")
    tiles = [("Regression", "continuous µg/m³ target"), ("Horizon", "one day ahead"),
             ("Lookback", "seven daily steps"), ("Fit scope", "fold-local only")]
    for i, (val, lab) in enumerate(tiles):
        x = 0.62 + i * 3.05
        _tile(slide, x, 1.48, 2.92, 0.72, val, lab, P1_ACCENTS[i], val_size=14)

    _card(slide, 0.62, 2.44, 5.9, 2.95, "0E7C86", fill_f=0.945, radius=0.06)
    text(slide, 0.84, 2.58, 5.4, 0.28, "What Part 1 must deliver", size=12.5, color=NAVY, bold=True)
    bullets(slide, 0.84, 2.95, 5.45, 2.3, [
        "A verified dataset description: source, licence, units and coverage.",
        "A documented cleaning policy: removals and deliberate retentions.",
        "A quantitative EDA of the target and its available drivers.",
        "A fold-local missing-value strategy and a seven-day, past-only feature definition.",
        "A fixed nine-channel feature contract for Parts 2 and 3.",
        "An explicit provenance position: redistribution rights are unresolved.",
    ], size=10.4, gap=13)

    _card(slide, 6.78, 2.44, 5.93, 2.95, "6C5CE7", fill_f=0.945, radius=0.06)
    text(slide, 7.0, 2.58, 5.4, 0.28, "Research questions informed here", size=12.5,
         color=NAVY, bold=True)
    rqs = [("RQ1", "Is the cleaned panel complete enough for matched-fold baselines?"),
           ("RQ3", "Which available feature groups carry signal, and which are excluded?"),
           ("RQ4", "How do seasonality, coverage and data rights constrain use?")]
    for i, (tag, q) in enumerate(rqs):
        y = 3.02 + i * 0.86
        _pill(slide, 7.0, y + 0.06, 0.55, 0.28, tag, P1_ACCENTS[3])
        text(slide, 7.68, y, 4.85, 0.62, q, size=9.6, color=INK, line_spacing=1.08)

    band = _grad(slide, 0.62, 5.52, 12.09, 1.10, "10243A", "2A2F66", angle=0, radius=0.05)
    text(slide, 0.95, 5.66, 11.5, 0.30, "Part 1 models nothing.", size=12.5,
         color=RGBColor(126, 227, 214), bold=True)
    text(slide, 0.95, 5.99, 11.5, 0.32,
         "It establishes that the data are trustworthy enough to model and that the inputs are built "
         "without leakage.", size=10.6, color=RGBColor(216, 228, 242))
    _p1_footer(slide, 3, "0E7C86")

    # ----------------------------------------------------------- 4. dataset --
    slide = blank(prs)
    _p1_header(slide, 4, "Dataset", "Dataset, licence and coverage", "2E6FA7", "04")
    tiles = [("41,718", "raw combined rows", "0E7C86"), ("40,983", "rows after cleaning", "1FA97A"),
             ("95", "coordinate-defined stations", "2E6FA7"), ("0", "duplicate station-day keys", "6C5CE7")]
    for i, (val, lab, acc) in enumerate(tiles):
        _tile(slide, 0.62 + i * 3.05, 1.45, 2.92, 0.80, val, lab, acc)

    rows = [
        ("Attribute", "Value", "Unit"),
        ("Source", "Kaggle · Air Quality – Mediterranean Basin (2016–2026), v1", "—"),
        ("Temporal span", "2016-11-25 → 2026-04-06", "UTC day"),
        ("Spatial span", "36.13–43.53 N · −5.66–9.14 E", "degrees"),
        ("Target", "PM2.5 concentration", "µg/m³"),
        ("Retain rule", "0 < PM2.5 < 500", "µg/m³"),
        ("Rows removed", "735  (727 low · 8 high)", "rows"),
    ]
    table(slide, 0.62, 2.48, 6.55, rows, [1.15, 3.6, 0.85], font=8.6, header_font=9.0,
          row_h=0.44, header_fill=_rgb("2E6FA7"), band=_tint("2E6FA7", 0.93))

    _card(slide, 7.42, 2.48, 5.29, 3.18, "1FA97A", fill_f=0.945, radius=0.06)
    text(slide, 7.64, 2.60, 4.9, 0.26, "Source-to-bundle matching audit", size=11.6,
         color=NAVY, bold=True)
    bullets(slide, 7.64, 2.95, 4.85, 2.6, [
        "PM2.5 source rows: 88,041 · GEE source rows: 164,208.",
        "All 41,718 bundle rows match both sources exactly.",
        "Maximum target difference: 3.6×10⁻¹⁵ µg/m³.",
        "46,323 PM2.5 source rows are absent; the upstream inclusion rule is not recoverable.",
        "The original ETL is not recoverable, so only the published rows can be audited.",
    ], size=9.4, gap=11)

    box = rect(slide, 0.62, 6.02, 12.09, 0.80, fill=_tint("D99A2B", 0.90),
               line=_tint("D99A2B", 0.55), radius=0.05)
    rect(slide, 0.62, 6.02, 0.07, 0.80, fill=_rgb("D99A2B"))
    text(slide, 0.88, 6.10, 11.6, 0.24, "Licence: redistribution rights unresolved",
         size=10.4, color=NAVY, bold=True)
    text(slide, 0.88, 6.36, 11.6, 0.36,
         "Kaggle labels the licence “Other (specified in description)”. No compiled-bundle licence or "
         "field-to-upstream attribution map exists, so the bundle is not presented as cleared for "
         "redistribution.", size=8.6, color=INK)
    _p1_footer(slide, 4, "2E6FA7")

    # ------------------------------------------------------------ 5. target --
    slide = blank(prs)
    _p1_header(slide, 5, "Target distribution", "The target is heavy-tailed", "E2604F", "05")
    _fig(slide, part_dir, "pm25_distribution.png", 0.62, 1.48, 5.95, 2.95,
         caption="Cleaned daily PM2.5 across 40,983 station-day records.", accent="E2604F")
    _fig(slide, part_dir, "pm25_boxplot.png", 0.62, 4.72, 5.95, 1.60,
         caption="Same target on linear and log scales.", accent="E2604F")

    stats = [("40,983", "records", "2E6FA7"), ("9.458", "mean", "0E7C86"), ("7.851", "median", "0E7C86"),
             ("9.730", "SD", "2E6FA7"), ("21.653", "P95", "D99A2B"), ("33.559", "P99", "D99A2B"),
             ("484.98", "maximum", "E2604F"), ("17.83", "skewness", "6C5CE7"),
             ("651.67", "excess kurtosis", "6C5CE7")]
    for i, (val, lab, acc) in enumerate(stats):
        col, row = i % 3, i // 3
        _tile(slide, 6.82 + col * 1.98, 1.48 + row * 0.72, 1.88, 0.64, val, lab, acc,
              val_size=14.5)

    _card(slide, 6.82, 3.78, 5.89, 1.62, "D99A2B", fill_f=0.93, radius=0.06)
    text(slide, 7.02, 3.88, 5.4, 0.24, "How often the target exceeds the cutoffs", size=10.8,
         color=NAVY, bold=True)
    cuts = [("> 25 µg/m³", "1,223", "2.98%", "E2604F"),
            ("> 50 µg/m³", "117", "0.29%", "D99A2B"),
            ("> 100 µg/m³", "37", "0.09%", "6C5CE7")]
    for i, (lab, n, share, acc) in enumerate(cuts):
        x = 7.02 + i * 1.88
        rect(slide, x, 4.20, 1.74, 0.98, fill=WHITE, line=_tint(acc, 0.6), radius=0.07)
        text(slide, x + 0.10, 4.26, 1.54, 0.20, lab, size=8.0, color=MUTED)
        text(slide, x + 0.10, 4.46, 1.54, 0.30, share, size=15, color=_rgb(acc), bold=True)
        text(slide, x + 0.10, 4.80, 1.54, 0.22, f"{n} records", size=8.0, color=INK)

    _card(slide, 6.82, 5.55, 5.89, 1.30, "0E7C86", fill_f=0.945, radius=0.06)
    bullets(slide, 7.02, 5.64, 5.5, 1.15, [
        "Half of all station-days sit at or below 7.85 µg/m³.",
        "P99 is 33.56 µg/m³ and the maximum 484.98 µg/m³ — roughly two orders of magnitude above the median.",
        "MAPE is unstable near zero, so RMSE, MAE and R² are the primary metrics.",
    ], size=9.4, gap=8)
    _p1_footer(slide, 5, "E2604F")

    # --------------------------------------------------------------- 6. EDA --
    slide = blank(prs)
    _p1_header(slide, 6, "Exploratory data analysis", "Every audited environmental variable", "1FA97A", "06")
    text(slide, 0.62, 1.40, 12.05, 0.30,
         "Eight distributions, each as a histogram with a boxplot. All are audited for completeness; "
         "none is used as a model input.", size=10.2, color=MUTED)
    vars_ = [("Temperature", "temperature_celsius", "Meteorology", "0E7C86"),
             ("Pressure", "pressure_mb", "Meteorology", "0E7C86"),
             ("Wind U", "wind_u", "Meteorology", "0E7C86"),
             ("Wind V", "wind_v", "Meteorology", "0E7C86"),
             ("NO₂", "no2", "Trace gas", "E2604F"),
             ("CO", "co", "Trace gas", "E2604F"),
             ("O₃", "o3", "Trace gas", "E2604F"),
             ("AOD", "aod", "Satellite", "6C5CE7")]
    for i, (label, file, family, acc) in enumerate(vars_):
        col, row = i % 4, i // 4
        x = 0.62 + col * 3.05
        y = 1.80 + row * 2.02
        text(slide, x, y, 2.9, 0.22, label, size=9.6, color=_rgb(acc), bold=True)
        _fig(slide, part_dir, f"distribution_{file}.png", x, y + 0.24, 2.9, 1.42,
             accent=acc)
    # family bands
    bands = [("Meteorology", "0% missing · 4 variables", "0E7C86"),
             ("Trace gases", "5.87–5.97% missing · 3 variables", "E2604F"),
             ("Satellite AOD", "0% missing · unitless", "6C5CE7")]
    for i, (title, sub, acc) in enumerate(bands):
        x = 0.62 + i * 4.08
        _card(slide, x, 6.02, 3.93, 0.80, acc, fill_f=0.94, radius=0.06)
        text(slide, x + 0.20, 6.10, 3.6, 0.24, title, size=10.2, color=NAVY, bold=True)
        text(slide, x + 0.20, 6.36, 3.6, 0.24, sub, size=8.0, color=MUTED)
    _p1_footer(slide, 6, "1FA97A")

    # ------------------------------------------------------- 7. correlations --
    slide = blank(prs)
    _p1_header(slide, 7, "Correlation analysis", "Associations are weak", "D99A2B", "07")
    _fig(slide, part_dir, "spearman_correlations.png", 0.62, 1.47, 5.55, 4.15,
         caption="Spearman rank-correlation structure of the cleaned panel.", accent="D99A2B")

    rows = [
        ("Variable", "Pearson r", "Spearman ρ", "Reading"),
        ("no2", "0.159", "0.247", "Strongest, still weak"),
        ("aod", "0.094", "0.180", "Weak positive"),
        ("pressure_mb", "0.113", "0.167", "Weak positive"),
        ("wind_u", "−0.079", "−0.164", "Weak negative"),
        ("o3", "−0.087", "−0.135", "Weak negative"),
        ("co", "0.041", "0.036", "Essentially none"),
        ("temperature_celsius", "−0.006", "0.064", "Essentially none"),
    ]
    table(slide, 6.42, 1.47, 6.29, rows, [2.0, 1.1, 1.15, 1.9], font=8.6, header_font=8.8,
          row_h=0.40, header_fill=_rgb("D99A2B"), band=_tint("D99A2B", 0.93))

    _card(slide, 6.42, 4.78, 6.29, 1.00, "0E7C86", fill_f=0.945, radius=0.06)
    bullets(slide, 6.62, 4.86, 5.9, 0.85, [
        "No pair exceeds |r| = 0.9, so no linear redundancy is flagged.",
        "These are context, not model inputs: availability at the forecast origin is undocumented.",
    ], size=8.6, gap=4)

    band = _grad(slide, 0.62, 6.02, 12.09, 0.82, "10243A", "2A2F66", angle=0, radius=0.05)
    text(slide, 0.90, 6.14, 11.5, 0.24,
         "Weak exogenous signal — and a pairwise screen cannot rule out nonlinear or multivariate dependence.",
         size=10.2, color=RGBColor(126, 227, 214), bold=True)
    text(slide, 0.90, 6.42, 11.5, 0.24,
         "Computed on the cleaned panel: results/correlations.csv", size=8.0,
         color=RGBColor(190, 205, 225))
    _p1_footer(slide, 7, "D99A2B")

    # ---------------------------------------------------------- 8. temporal --
    slide = blank(prs)
    _p1_header(slide, 8, "Temporal analysis", "Seasonality, drift and dependence", "0E7C86", "08")
    _fig(slide, part_dir, "pm25_by_season.png", 0.62, 1.44, 5.90, 1.99,
         caption="Seasonal summaries (DJF, MAM, JJA, SON).", accent="0E7C86")
    _fig(slide, part_dir, "pm25_by_month.png", 6.78, 1.44, 5.93, 2.15,
         caption="Monthly mean and median pattern.", accent="6C5CE7")

    rows = [
        ("Season", "Records", "Mean", "> 25"),
        ("DJF", "8,938", "10.94", "6.15%"),
        ("MAM", "9,276", "8.25", "1.92%"),
        ("JJA", "12,766", "9.59", "1.95%"),
        ("SON", "10,003", "9.08", "2.46%"),
    ]
    table(slide, 0.62, 3.98, 5.90, rows, [1.1, 1.3, 1.0, 1.0], font=8.8, header_font=8.8,
          row_h=0.315, header_fill=_rgb("0E7C86"), band=_tint("0E7C86", 0.93))

    # annual drift card
    _card(slide, 6.78, 3.96, 5.93, 1.62, "D99A2B", fill_f=0.94, radius=0.06)
    text(slide, 6.98, 4.04, 5.4, 0.24, "Annual mean — long-run drift", size=10.4,
         color=NAVY, bold=True)
    years = [("2020", 7.56), ("2021", 8.86), ("2022", 9.62), ("2023", 10.22),
             ("2024", 9.61), ("2025", 9.92)]
    base_y = 5.16
    for i, (yr, val) in enumerate(years):
        x = 7.02 + i * 0.93
        h = (val / 10.6) * 0.52
        acc = P1_ACCENTS[i % len(P1_ACCENTS)]
        rect(slide, x, base_y - h, 0.52, h, fill=_rgb(acc), radius=0.12)
        text(slide, x - 0.06, base_y - h - 0.20, 0.64, 0.18, f"{val:.2f}", size=7.0,
             color=_rgb(acc), bold=True, align=PP_ALIGN.CENTER)
        text(slide, x - 0.03, base_y + 0.02, 0.58, 0.18, yr, size=7.0, color=MUTED,
             align=PP_ALIGN.CENTER)
    text(slide, 6.98, 5.38, 5.6, 0.20,
         "Weighted by monthly record counts; 2023 covers 10 months. Recomputed from results/eda/monthly_trend.csv.",
         size=6.6, color=MUTED)

    # dependence strip
    _card(slide, 0.62, 5.70, 12.09, 1.05, "6C5CE7", fill_f=0.94, radius=0.06)
    facts = [("17 d", "median longest run"), ("32 d", "maximum run"),
             ("63", "stations with ≥ 14 days"), ("6", "stations testable by ADF/KPSS")]
    for i, (val, lab) in enumerate(facts):
        x = 0.92 + i * 2.95
        text(slide, x, 5.80, 0.95, 0.32, val, size=14, color=_rgb(P1_ACCENTS[i]),
             bold=True)
        text(slide, x + 0.92, 5.84, 1.95, 0.42, lab, size=8.0, color=MUTED)
    text(slide, 0.92, 6.34, 11.5, 0.30,
         "Formal stationarity tests are only defensible on long uninterrupted runs; they are therefore "
         "descriptive, not a global verdict.", size=8.2, color=INK)
    _p1_footer(slide, 8, "0E7C86")

    # ----------------------------------------------------------- 9. cleaning --
    slide = blank(prs)
    _p1_header(slide, 9, "Data cleaning", "One rule, six checks, no silent deletions", "E2604F", "09")
    flow = [("Raw combined", "41,718 rows", "2E6FA7"), ("Removed", "735 rows · 727 low · 8 high", "E2604F"),
            ("Clean panel", "40,983 rows · 95 stations", "1FA97A")]
    xs = [0.62, 4.55, 8.45]
    widths = [3.55, 3.45, 4.26]
    for (title, sub, acc), x, w in zip(flow, xs, widths):
        _card(slide, x, 1.45, w, 1.05, acc, fill_f=0.91, radius=0.08)
        text(slide, x + 0.20, 1.55, w - 0.3, 0.28, title, size=12.0, color=NAVY, bold=True)
        text(slide, x + 0.20, 1.88, w - 0.3, 0.26, sub, size=9.0, color=_rgb(acc), bold=True)
    for ax, aw in ((4.17, 0.38), (8.00, 0.45)):
        text(slide, ax, 1.80, aw, 0.34, "→", size=15, color=_rgb("93A3B8"), bold=True,
             align=PP_ALIGN.CENTER)

    rows = [
        ("Check", "Result", "Reading"),
        ("Duplicate station-day keys", "0", "Keyed table is unique."),
        ("Out-of-rule / non-finite targets", "0", "All retained targets inside the rule."),
        ("Coordinate validity", "0 invalid", "95 stations with finite lat/lon."),
        ("Station identity reconciliation", "98 → 95", "Three zero-target identities removed."),
        ("Station-relative Tukey screen", "352 flagged", "71 stations; retained for review."),
        ("Development high values", "90 / 36 / 11", "> 50 / > 100 / > 250 µg/m³ kept."),
    ]
    table(slide, 0.62, 2.70, 6.55, rows, [2.1, 0.9, 2.4], font=8.2, header_font=8.6,
          row_h=0.50, header_fill=_rgb("E2604F"), band=_tint("E2604F", 0.94))

    _card(slide, 7.42, 2.70, 5.29, 1.55, "D99A2B", fill_f=0.93, radius=0.06)
    text(slide, 7.62, 2.80, 4.9, 0.24, "Extremes are kept, not deleted", size=10.6,
         color=NAVY, bold=True)
    bullets(slide, 7.62, 3.12, 4.85, 1.05, [
        "Statistical extremeness alone cannot establish an instrument fault.",
        "Deleting extremes would bias the target and remove the episodes the model must learn.",
    ], size=9.8, gap=10)

    _card(slide, 7.42, 4.39, 5.29, 1.81, "2E6FA7", fill_f=0.945, radius=0.06)
    text(slide, 7.62, 4.49, 4.9, 0.24, "Station identity is composite", size=10.6,
         color=NAVY, bold=True)
    text(slide, 7.62, 4.78, 4.9, 0.28, "A name is not a station.", size=12.5,
         color=_rgb("2E6FA7"), bold=True)
    text(slide, 7.62, 5.16, 4.85, 0.95,
         "A station ID is the sensor name plus latitude and longitude to six decimals. Cleaning removed "
         "three zero-target identities, leaving 95 stations. Four sensor names map to two stations each, "
         "so Part 2 must split on the composite ID — splitting on the name alone would leak one "
         "instrument across folds.",
         size=8.8, color=INK, line_spacing=1.12)

    text(slide, 0.62, 6.36, 6.55, 0.26,
         "Development-scope audit: 30,651 rows and 73 stations to 2024-12-07.",
         size=8.2, color=MUTED)
    _p1_footer(slide, 9, "E2604F")

    # --------------------------------------------------------- 10. pipeline --
    slide = blank(prs)
    _p1_header(slide, 10, "Preprocessing pipeline", "From raw rows to seven-day sequences", "6C5CE7", "10")
    stages = [("Raw CSV", "41,718 rows", "2E6FA7"), ("Clean", "target rule + checks", "E2604F"),
              ("Split", "date / station folds", "D99A2B"), ("Fold-local", "impute + scale", "1FA97A"),
              ("Sequences", "7 days, no gaps", "6C5CE7")]
    for i, (title, sub, acc) in enumerate(stages):
        x = 0.62 + i * 2.44
        _card(slide, x, 1.44, 2.28, 1.02, acc, fill_f=0.91, radius=0.09)
        text(slide, x + 0.14, 1.54, 2.0, 0.26, title, size=11.0, color=NAVY, bold=True)
        text(slide, x + 0.14, 1.86, 2.0, 0.36, sub, size=8.2, color=_rgb(acc), bold=True)
        if i < 4:
            text(slide, x + 2.27, 1.80, 0.18, 0.3, "→", size=13, color=_rgb("93A3B8"), bold=True)

    rows = [
        ("Input channel", "Type", "Why it is safe"),
        ("pm25_lag1", "Past value", "PM2.5 at the origin t — persistence signal."),
        ("pm25_roll3_mean", "Past rolling", "Mean of the three days ending at t."),
        ("pm25_roll7_mean", "Past rolling", "Mean of the seven days ending at t."),
        ("day_of_year_sin/cos", "Calendar", "Annual cycle known in advance."),
        ("day_of_week_sin/cos", "Calendar", "Weekly cycle known in advance."),
        ("rolling missing flags", "Indicator", "Mark where a rolling summary was filled."),
    ]
    table(slide, 0.62, 2.78, 6.55, rows, [1.9, 1.0, 2.9], font=8.2, header_font=8.6,
          row_h=0.50, header_fill=_rgb("6C5CE7"), band=_tint("6C5CE7", 0.94))
    text(slide, 0.62, 6.38, 6.55, 0.36,
         "Nine input channels: seven engineered features plus two rolling-missingness indicators.",
         size=8.4, color=MUTED, line_spacing=1.1)

    _tile(slide, 7.42, 2.78, 2.55, 0.96, "9", "model input channels", "6C5CE7",
          sub="7 features + 2 flags", val_size=20)
    _tile(slide, 10.14, 2.78, 2.57, 0.96, "8 + 1", "audited variables + target", "0E7C86",
          sub="target in µg/m³", val_size=17)

    _card(slide, 7.42, 3.92, 5.29, 1.62, "1FA97A", fill_f=0.94, radius=0.06)
    text(slide, 7.62, 4.02, 4.9, 0.24, "Missing-value strategy", size=10.6, color=NAVY, bold=True)
    bullets(slide, 7.62, 4.32, 4.85, 1.15, [
        "Median imputation, fitted inside each training fold only.",
        "Two rolling-missingness flags keep filled and observed values distinguishable.",
        "Gas gaps: 5.87% (NO₂, CO) and 5.97% (O₃); target and calendar terms are complete.",
    ], size=8.8, gap=6)

    _card(slide, 7.42, 5.70, 5.29, 0.58, "2E6FA7", fill_f=0.945, radius=0.06)
    text(slide, 7.66, 5.81, 4.9, 0.30,
         "No target scaling — PM2.5 stays on its original µg/m³ scale.",
         size=9.0, color=INK)
    _p1_footer(slide, 10, "6C5CE7")

    # --------------------------------------------------------- 11. leakage --
    slide = blank(prs)
    _p1_header(slide, 11, "Leakage prevention and validation", "Every transform is train-only", "1FA97A", "11")
    rows = [
        ("Control", "Status", "Automated check"),
        ("Fold-local imputation and scaling", "Pass", "Statistics fitted on the training fold only."),
        ("Past-only rolling features", "Pass", "Future-value mutation leaves features unchanged."),
        ("Origin / target alignment", "Pass", "Windows end at t; label is t + 1."),
        ("Gap and station boundaries", "Pass", "No window crosses a gap or a station."),
        ("Environmental timing", "Blocked", "Exogenous channels withheld, not assumed available."),
        ("Frozen holdout untouched", "Pass", "Latest chronological block excluded from folds."),
    ]
    table(slide, 0.62, 1.48, 6.55, rows, [2.2, 0.75, 3.1], font=8.2, header_font=8.6,
          row_h=0.46, header_fill=_rgb("1FA97A"), band=_tint("1FA97A", 0.94))

    _card(slide, 7.42, 1.48, 5.29, 2.30, "2E6FA7", fill_f=0.945, radius=0.06)
    text(slide, 7.62, 1.58, 4.9, 0.24, "Fold strategy", size=10.6, color=NAVY, bold=True)
    text(slide, 7.62, 1.86, 4.9, 0.26, "Development window 2016-11 → 2024-12", size=8.2,
         color=MUTED)
    fold_cols = ["2E6FA7", "0E7C86", "1FA97A", "D99A2B", "6C5CE7"]
    for i in range(5):
        x = 7.62 + i * 0.62
        rect(slide, x, 2.22, 0.54, 0.52, fill=_rgb(fold_cols[i]), radius=0.12)
        text(slide, x, 2.34, 0.54, 0.24, f"F{i+1}", size=8.0, color=WHITE, bold=True,
             align=PP_ALIGN.CENTER)
    rect(slide, 10.82, 2.22, 0.14, 0.52, fill=_rgb("E2604F"), radius=0.2)
    text(slide, 10.76, 2.80, 1.0, 0.20, "1-day embargo", size=6.8, color=MUTED)
    rect(slide, 11.06, 2.22, 1.42, 0.52, fill=_rgb("10243A"), radius=0.1)
    text(slide, 11.06, 2.30, 1.42, 0.36, "Frozen\nfuture", size=7.6, color=WHITE, bold=True,
         align=PP_ALIGN.CENTER, line_spacing=0.95)
    text(slide, 7.62, 3.36, 4.9, 0.30,
         "5 temporal folds · 5 spatial folds · 1 diagnostic standard split",
         size=8.0, color=INK)

    _card(slide, 7.42, 3.92, 5.29, 0.90, "D99A2B", fill_f=0.94, radius=0.06)
    text(slide, 7.62, 4.02, 4.9, 0.24, "Fold artifacts", size=10.2, color=NAVY, bold=True)
    text(slide, 7.62, 4.28, 4.9, 0.44,
         "folds/folds_temporal.json · folds/folds_spatial.json · folds/folds_standard.json",
         size=7.6, color=INK, line_spacing=1.1)

    band = _grad(slide, 0.62, 5.10, 12.09, 1.42, "10243A", "2A2F66", angle=0, radius=0.05)
    text(slide, 0.92, 5.22, 11.5, 0.26, "Executed evidence — 63 passed, 0 failed in 22.9 s",
         size=12.0, color=RGBColor(126, 227, 214), bold=True)
    text(slide, 0.92, 5.54, 11.5, 0.28,
         "17 test files · 2026-11-14 · log: reports/reproducibility/pytest_2026-11-14_part1.txt",
         size=8.6, color=RGBColor(200, 214, 232))
    text(slide, 0.92, 5.92, 11.5, 0.55,
         "Boundary: the tests assess the executed preprocessing pipeline. They cannot establish that the "
         "publisher's original upstream join was leakage-free or reconstruct its row-selection rule.",
         size=8.6, color=RGBColor(190, 205, 226), line_spacing=1.12)
    _p1_footer(slide, 11, "1FA97A")

    # ------------------------------------------------------- 12. conclusion --
    slide = blank(prs)
    _p1_header(slide, 12, "Conclusion and next steps", "What Part 1 delivers", "3B4C8C", "12")
    cols = [
        ("What Part 1 establishes", "1FA97A", [
            "A verified panel: 40,983 rows, 95 stations, no duplicate keys.",
            "A documented rule: 735 invalid targets removed, extremes kept.",
            "A weak but quantified relationship between PM2.5 and every available predictor.",
            "A leak-free input contract: seven past-only days, fold-local fitting.",
            "A coverage profile: 2,299 UTC days, median uninterrupted run of 17 days.",
        ]),
        ("Limitations carried forward", "E2604F", [
            "Redistribution rights remain unresolved.",
            "The publisher's upstream inclusion rule cannot be reconstructed.",
            "Sparse, irregular coverage limits formal time-series tests to six stations.",
            "Only a handful of development values exceed 100 µg/m³.",
            "Exogenous variables are excluded because origin availability is unverified.",
        ]),
        ("Next development stage", "6C5CE7", [
            "Part 2: temporal and station-grouped validation of the CNN–LSTM.",
            "Keep imputation, scaling and feature construction fold-local.",
            "Report uncertainty on every aggregate and every comparison.",
            "Treat the frozen future block as a one-shot test.",
            "Freeze the feature contract so Parts 2 and 3 consume identical inputs.",
        ]),
    ]
    for i, (title, acc, items) in enumerate(cols):
        x = 0.62 + i * 4.08
        _card(slide, x, 1.46, 3.93, 3.05, acc, fill_f=0.945, radius=0.06)
        text(slide, x + 0.20, 1.58, 3.55, 0.50, title, size=11.2, color=_rgb(acc), bold=True,
             line_spacing=1.0)
        bullets(slide, x + 0.20, 2.20, 3.55, 2.12, items, size=9.6, gap=8)

    band = _grad(slide, 0.62, 4.88, 12.09, 1.62, "10243A", "2A2F66", angle=0, radius=0.05)
    text(slide, 0.95, 5.02, 11.4, 0.30, "Part 1 verdict", size=11.8,
         color=RGBColor(126, 227, 214), bold=True)
    text(slide, 0.95, 5.38, 11.4, 0.60,
         "The panel is clean enough to model and documented well enough to audit. It is not a dense grid "
         "and its licence is unresolved — so Part 2 must validate with temporal and station-grouped "
         "folds, keep preprocessing fold-local, and report uncertainty honestly.",
         size=10.4, color=RGBColor(214, 226, 240), line_spacing=1.18)
    text(slide, 0.95, 6.12, 11.4, 0.22,
         "Part 1 of 3 — the data contract is frozen for Part 2 and Part 3.",
         size=8.2, color=RGBColor(150, 170, 200))
    _p1_footer(slide, 12, "3B4C8C")

    out = ROOT / "presentation" / "Part_1_Presentation.pptx"
    prs.save(out)
    return out


# --------------------------------------------------------------------------- #
# Part 2
# --------------------------------------------------------------------------- #
def build_part2():
    part_dir = ROOT / "reports" / "part_2"
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

    out = ROOT / "presentation" / "Part_2_Presentation.pptx"
    prs.save(out)
    return out


# --------------------------------------------------------------------------- #
# Part 3
# --------------------------------------------------------------------------- #
def build_part3():
    part_dir = ROOT / "reports" / "part_3"
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
        "Every claim is backed by a file in results/, and every limitation is stated where it applies.",
        "Across all three parts, the deliverable is a defensible scientific argument — not a score.",
    ], size=11.5, gap=6)
    footer(slide, label, num)
    notes(slide, "Close on scientific credibility as the differentiator.")

    out = ROOT / "presentation" / "Part_3_Presentation.pptx"
    prs.save(out)
    return out


def main():
    for fn in (build_part1, build_part2, build_part3):
        path = fn()
        print(f"wrote {path}  ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
