"""Run the feature-family ablation and write ``results/ablation.csv``.

Each variant drops one predictor family and is scored on the same samples as
the full model, so the difference reflects the dropped family rather than a
change in the evaluated cohort.
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
    raise SystemExit(run("ablate"))
