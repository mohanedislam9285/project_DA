from pm25_project.config import load_config
from pm25_project.cnn_lstm_benchmark import cnn_lstm_candidates

def test_declared_search_space_drives_config_candidates():
    cfg, _ = load_config("configs/project.yaml")
    assert len(cnn_lstm_candidates(cfg)) == 3
    cnn_candidates = cnn_lstm_candidates(cfg)
    assert [c["lookback_days"] for c in cnn_candidates] == [7, 7, 7]
    assert len({(c["filters"], c["kernel_size"], c["lstm_units"], c["dropout"], c["learning_rate"]) for c in cnn_candidates}) == 3
    assert cfg["_search_spaces"]["budget"]["maximum_configurations_per_model_fold"] == 50

