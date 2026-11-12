"""Pure CNN-LSTM model registry."""
from __future__ import annotations
from .cnn_lstm import build_cnn_lstm

class CNNLSTMRegistryWrapper:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def __call__(self, cfg: dict | None = None):
        cfg = dict(cfg or {})
        cfg.setdefault('seed', self.seed)
        lookback = cfg.get('lookback_days', 7)
        n_features = cfg.get('n_features', 0)
        return build_cnn_lstm(lookback, n_features, cfg)

def model_registry(seed: int = 42) -> dict:
    return {'CNN-LSTM': CNNLSTMRegistryWrapper(seed), 'CNN_LSTM': CNNLSTMRegistryWrapper(seed)}
