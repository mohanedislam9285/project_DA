"""Project paths and global constants.

Configuration loading and validation are implemented in
:mod:`pm25_project.config`; this module is the stable entry point used by the
scripts in ``scripts/`` and by the notebooks in ``notebooks/``.
"""

from __future__ import annotations

from pathlib import Path

from pm25_project.config import load_config, resolve_path, validate_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "src" / "configs" / "project.yaml"
SEED = 42

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
FOLDS_DIR = PROJECT_ROOT / "folds"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
PRESENTATION_DIR = PROJECT_ROOT / "presentation"


def load_project_config(path: str | Path | None = None):
    """Load ``project.yaml``, validate it, and return ``(config, project_root)``."""
    cfg, root = load_config(path or CONFIG_PATH)
    validate_config(cfg)
    return cfg, root


__all__ = [
    "PROJECT_ROOT",
    "CONFIG_PATH",
    "SEED",
    "DATA_DIR",
    "DATA_RAW",
    "DATA_PROCESSED",
    "FOLDS_DIR",
    "RESULTS_DIR",
    "FIGURES_DIR",
    "REPORTS_DIR",
    "PRESENTATION_DIR",
    "load_config",
    "resolve_path",
    "validate_config",
    "load_project_config",
]
