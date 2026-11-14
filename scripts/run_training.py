"""Train the CNN-LSTM and select hyperparameters inside each temporal fold.

Writes fold-level metrics, the selected configurations
(``results/hyperparameters.csv``), permutation importance and the paired
comparison table. Configuration search is chronological and restricted to the
training partition of each outer fold.
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
    raise SystemExit(run("cnn-lstm"))
