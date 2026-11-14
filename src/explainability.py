"""Permutation importance and ranking stability.

Importance is measured by permuting one input channel at a time on held-out
data. Stability across folds and validation schemes is summarized with
Spearman correlation and Jaccard overlap of the top-ranked features.
"""

from __future__ import annotations

from pm25_project.explainability import (
    fold_ranking_stability,
    sequence_channel_permutation,
    tabular_permutation_importance,
)

__all__ = [
    "tabular_permutation_importance",
    "sequence_channel_permutation",
    "fold_ranking_stability",
]
