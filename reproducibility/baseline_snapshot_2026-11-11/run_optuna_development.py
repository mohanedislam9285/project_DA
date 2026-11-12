"""Run the nested TPE search using development rows only."""
from pathlib import Path
import pandas as pd

from pm25_project.config import load_config
from pm25_project.optuna_development import run_optuna_temporal


config, _ = load_config("configs/project.yaml")
source = pd.read_csv("data/processed/cleaned_daily.csv", parse_dates=["timestamp"])
source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True)
print(run_optuna_temporal(source, config, trial_budget=50))
