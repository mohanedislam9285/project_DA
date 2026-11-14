"""Fold generation and JSON persistence.

Three fold families are written to ``folds/``: expanding temporal folds with a
one-day embargo (``folds_temporal.json``), station-grouped folds
(``folds_spatial.json``) and random diagnostic folds (``folds_standard.json``).
"""

from __future__ import annotations

from pm25_project.splits import (
    development_future_split,
    random_diagnostic_folds,
    save_folds,
    spatial_folds,
    temporal_panel_folds,
)

__all__ = [
    "development_future_split",
    "temporal_panel_folds",
    "spatial_folds",
    "random_diagnostic_folds",
    "save_folds",
]
