"""Run the exploratory analysis and write the EDA tables and figures.

Produces ``results/Exploratory_Data.csv``, ``results/Exploratory_temporal.csv``,
``results/correlations.csv``, the per-variable figures under
``figures/`` and the two summary figures named in the project template:
``figures/eda_histogrammes.png`` and ``figures/correlation_matrix.png``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pm25_project.cli import main  # noqa: E402

CONFIG = str(ROOT / "src" / "configs" / "project.yaml")


def promote_figures() -> None:
    """Move CLI-written charts from results/figures into the template-level figures/."""
    source = ROOT / "results" / "figures"
    if not source.is_dir():
        return
    destination = ROOT / "figures"
    destination.mkdir(exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, destination / path.name)
            path.unlink()
    source.rmdir()


def summary_figures() -> None:
    """Compose the template summary figures from the cleaned panel and saved correlations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    navy, teal, amber, ink = "#10243A", "#087F80", "#D9822B", "#243447"
    frame = pd.read_csv(ROOT / "data" / "processed" / "cleaned_daily.csv", parse_dates=["timestamp"])

    panels = [
        ("pm25", "PM2.5 (µg/m³)", teal),
        ("aod", "AOD (dimensionless)", amber),
        ("temperature_celsius", "Temperature (°C)", "#2E6FA7"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (column, label, colour) in zip(axes, panels):
        values = frame[column].dropna()
        ax.hist(values, bins=45, color=colour, edgecolor="white", linewidth=0.6)
        ax.set_title(f"{label} — n = {len(values):,}", color=navy, fontsize=11, weight="bold")
        ax.set_xlabel(label, color=ink)
        ax.set_ylabel("Station-days", color=ink)
        ax.grid(axis="y", color="#D5DEE7", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = ROOT / "figures" / "eda_histogrammes.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)

    pairs = pd.read_csv(ROOT / "results" / "correlations.csv")
    variables = sorted(set(pairs.variable_a) | set(pairs.variable_b))
    index = {name: i for i, name in enumerate(variables)}
    matrix = np.eye(len(variables))
    for row in pairs.itertuples():
        matrix[index[row.variable_a], index[row.variable_b]] = row.spearman_rho
        matrix[index[row.variable_b], index[row.variable_a]] = row.spearman_rho
    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(variables)), variables, rotation=45, ha="right")
    ax.set_yticks(range(len(variables)), variables)
    for i in range(len(variables)):
        for j in range(len(variables)):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > 0.55 else ink)
    ax.set_title("Spearman correlation of target and predictors", color=navy, fontsize=12, weight="bold")
    fig.colorbar(image, ax=ax, label="Spearman ρ", shrink=0.85)
    fig.tight_layout()
    out = ROOT / "figures" / "correlation_matrix.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)


def run(*commands: str) -> int:
    for command in commands:
        code = main([command, "--config", CONFIG])
        if code:
            return code
    return 0


if __name__ == "__main__":
    code = run("audit", "eda")
    if code:
        raise SystemExit(code)
    promote_figures()
    summary_figures()
