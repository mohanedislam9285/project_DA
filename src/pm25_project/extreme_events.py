from __future__ import annotations

import numpy as np


def extreme_event_sample_weights(target, moderate_weight:float=2.0, extreme_weight:float=4.0)->np.ndarray:
    """Return fixed sample weights for the >25 and >50 µg/m³ target bands."""
    if moderate_weight<=0 or extreme_weight<=0:
        raise ValueError("sample weights must be positive")
    values=np.asarray(target,dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("targets must be finite")
    return np.where(values>50,extreme_weight,np.where(values>25,moderate_weight,1.0)).astype(np.float32)
