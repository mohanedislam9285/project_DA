"""Inventory the assigned CNN-LSTM and supplementary model-family references."""
from __future__ import annotations
import importlib.util
import pandas as pd

INVENTORY = [
    ("CNN-LSTM", "primary_hybrid", "implemented", "Core model, TensorFlow required for execution")
]


def normalize_model_id(value: object) -> str:
    """Coerce the project's CNN-LSTM naming variations to one canonical identifier."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return text
    return text.replace("-", "_")


def inventory_table():
    rows = []
    for name, family, status, notes in INVENTORY:
        dependency = "tensorflow" if normalize_model_id(name) == "CNN_LSTM" else None
        available = bool(importlib.util.find_spec(dependency)) if dependency else True
        assignment_scope = "required_primary"
        rows.append({
            "model_id": name,
            "model_group": family,
            "assignment_scope": assignment_scope,
            "implementation_status": status,
            "dependency": dependency,
            "dependency_available": available if dependency else None,
            "notes": notes,
        })
    return pd.DataFrame(rows)
