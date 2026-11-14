"""Reproducible, leakage-aware PM2.5 forecasting pipeline."""
import os

__version__ = "0.1.0"

# Pin Python's string-hash randomisation. PYTHONHASHSEED only binds when the interpreter
# starts, so export it before launching. The default here keeps child processes
# launched from the package consistent as well.
os.environ.setdefault("PYTHONHASHSEED", "42")
