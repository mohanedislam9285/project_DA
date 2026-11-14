"""Model registry.

The registered predictor is the CNN-LSTM used throughout the study: a causal
convolution stack followed by an LSTM encoder and a dense regression head.
Naive baselines live in :mod:`pm25_project.baselines` and are evaluated through
the same fold protocol.
"""

from __future__ import annotations

from pm25_project.models.model_registry import CNNLSTMRegistryWrapper, model_registry

__all__ = ["CNNLSTMRegistryWrapper", "model_registry"]
