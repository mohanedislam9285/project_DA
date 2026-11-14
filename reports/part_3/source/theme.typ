// ============================================================================
//  Group 16 · PM2.5 forecasting · submission part design system
//  Shared Typst theme (palette, components, helpers) for all three parts.
//  Build: typst compile   ·   Lint: tinymist lint
// ============================================================================

#let navy = rgb("#10243A")
#let navy2 = rgb("#1B3A5C")
#let teal = rgb("#087F80")
#let amber = rgb("#D9822B")
#let green = rgb("#2E8B57")
#let red = rgb("#C0392B")
#let ink = rgb("#243447")
#let muted = rgb("#607184")
#let pale = rgb("#EEF4F6")
#let ice = rgb("#E8F3F3")
#let sand = rgb("#F8F1E8")
#let rule = rgb("#D5DEE7")

// ---- small labels -----------------------------------------------------------
#let eyebrow(label) = text(size: 8pt, weight: "bold", fill: teal, tracking: 1.4pt)[#label]
#let kicker(label) = text(size: 8.2pt, weight: "bold", fill: muted, tracking: 0.8pt)[#label]

// ---- caption ---------------------------------------------------------------
#let caption(body) = block(width: 100%)[
  #text(size: 8pt, fill: muted, style: "italic")[#body]
]

// ---- callout ----------------------------------------------------------------
#let callout(title, body, accent: teal, tint: pale) = block(
  width: 100%,
  fill: tint,
  stroke: (left: 3pt + accent),
  inset: (x: 11pt, y: 9pt),
  radius: 3pt,
)[
  #text(weight: "bold", fill: navy, size: 9.6pt)[#title]
  #v(3pt)
  #body
]

// ---- flow node --------------------------------------------------------------
#let node(title, detail, tone: pale, border: rule) = block(
  width: 100%,
  fill: tone,
  stroke: 0.7pt + border,
  inset: (x: 6pt, y: 7pt),
  radius: 4pt,
)[
  #align(center)[
    #text(size: 8.5pt, weight: "bold", fill: navy)[#title]
    #linebreak()
    #text(size: 7.1pt, fill: muted)[#detail]
  ]
]

// ---- KPI tile ---------------------------------------------------------------
#let stat(value, label, accent: teal, sub: none) = block(
  width: 100%,
  fill: white,
  stroke: 0.8pt + rule,
  inset: (x: 9pt, y: 8pt),
  radius: 4pt,
)[
  #text(size: 16pt, weight: "bold", fill: accent)[#value]
  #linebreak()
  #text(size: 7.5pt, fill: muted)[#label]
  #if sub != none [
    #linebreak()
    #text(size: 6.8pt, fill: rule.darken(20%))[#sub]
  ]
]

// ---- tables -----------------------------------------------------------------
#let th(s) = text(size: 8pt, weight: "bold", fill: white)[#s]

#let tbl(cols, ..cells) = table(
  columns: cols,
  inset: (x: 6.5pt, y: 5pt),
  stroke: (x, y) => if y == 0 { 0pt } else { 0.45pt + rule },
  fill: (x, y) => if y == 0 { navy } else if calc.rem(y, 2) == 0 { pale } else { white },
  align: left,
  ..cells,
)

// ---- bullet list with square markers ----------------------------------------
#let ticks(size: 9.4pt, ..items) = {
  set list(marker: text(fill: teal)[▸], indent: 0.6em)
  set text(size: size, fill: ink)
  list(..items)
}

// ---- metric chip (inline) ---------------------------------------------------
#let chip(text-body, accent: teal) = box(
  fill: ice,
  stroke: 0.6pt + accent,
  inset: (x: 6pt, y: 2.5pt),
  radius: 3pt,
)[#text(size: 8pt, weight: "bold", fill: accent)[#text-body]]
