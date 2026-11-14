"""Origin-safe feature construction: lags, rolling means and cyclic calendar terms.

Every feature is computed from observations available at the forecast origin.
Windows never bridge a station boundary or a missing-day gap, and exogenous
variables are admitted only when their value is certified available at origin
in ``src/configs/project.yaml``.
"""

from __future__ import annotations

from pm25_project.features import build_origin_features

__all__ = ["build_origin_features"]
