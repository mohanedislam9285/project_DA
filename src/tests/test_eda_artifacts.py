import pandas as pd
from pm25_project.eda import exploratory_table,run_eda
from pm25_project.temporal_diagnostics import temporal_table

def test_required_summary_fields_are_computed(panel,config):
    table=exploratory_table(panel,config)
    target=table[table.variable=="pm25"].iloc[0]
    assert target.n_total==len(panel)
    assert target.n_missing==0
    temporal=temporal_table(panel)
    assert len(temporal)==2
    assert temporal.acf_lag1.notna().all()
    assert temporal.trend_slope_per_day.notna().all()
    assert temporal.notes.str.contains("KPSS p-value is at least 0.1").all()

def test_eda_generates_actual_summary_and_figures(panel,config):
    panel=panel.assign(clean_index=range(len(panel)))
    run_eda(panel,config)
    root=__import__("pathlib").Path(config["_config_dir"])/"results/TEST-00"
    assert pd.read_csv(root/"station_summary.csv").shape[0]==2
    assert (root/"correlations.csv").exists()
    assert (root/"figures/pm25_over_time.png").exists()
