"""Feature-family ablation.

Variants drop one predictor family at a time and are scored on identical
samples, so the difference isolates the family's contribution rather than a
change in the evaluated cohort.
"""

from __future__ import annotations

from pm25_project.ablations import (
    feature_family_variants,
    require_same_samples,
    run_feature_ablation,
)

__all__ = [
    "feature_family_variants",
    "require_same_samples",
    "run_feature_ablation",
]
