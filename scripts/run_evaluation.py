"""Evaluate the model suite on every saved fold scheme.

Runs the temporal, station-grouped and random diagnostic benchmarks, writing
``results/results.csv`` and ``results/comparisons.csv`` alongside the
hyperparameter, importance and experiment tables. The frozen future block is
not touched.
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
    raise SystemExit(run("benchmark"))
