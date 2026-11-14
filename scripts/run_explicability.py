"""Explain the fitted model: permutation importance, local cases and rank stability.

Writes ``results/feature_importance.csv``, the per-case explanations and
``results/ranking_stability.csv`` (Spearman ρ and Jaccard overlap of the top
features across validation schemes).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pm25_project.cli import main  # noqa: E402

CONFIG = str(ROOT / "src" / "configs" / "project.yaml")


def run(*commands: str) -> int:
    for command in commands:
        code = main([command, "--config", CONFIG])
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(run("local-explanations", "ranking-stability"))
