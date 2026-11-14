"""Metrics and paired statistical tests.

``regression_metrics`` returns RMSE, MAE and R² on the original µg/m³ scale.
``paired_bootstrap_ci`` resamples fold-level or station-level differences to
report an uncertainty interval around a paired comparison.
"""

from __future__ import annotations

from pm25_project.metrics import paired_bootstrap_ci, regression_metrics

__all__ = ["regression_metrics", "paired_bootstrap_ci"]
