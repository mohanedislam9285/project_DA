"""Reproducible, leakage-aware PM2.5 forecasting pipeline."""
import os

__version__ = "0.1.0"

# Pin Python's string-hash randomisation. PYTHONHASHSEED only binds when the interpreter
# starts, so export it before launching (see docs/REPRODUCIBILITY.md); defaulting it here
# keeps the same value for any child process spawned from the package.
os.environ.setdefault("PYTHONHASHSEED", "42")
