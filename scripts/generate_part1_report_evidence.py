"""Generate the additional Part 1 report evidence from saved artifacts.

Outputs:
  results/part1_evidence_summary.json
  figures/pm25_boxplot.png
  figures/predictor_missingness.png
  figures/station_coverage_development.png
  figures/pollutant_missingness_heatmap_development.png

Every value is computed from data/processed/cleaned_daily.csv, the saved
development partition (results/future_test.json), the pollutant
station-month missingness table, and the upstream join audit. No value is
typed in by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INK = "#243447"
NAVY = "#10243A"
TEAL = "#087F80"
AMBER = "#D9822B"
RULE = "#D5DEE7"


def longest_run(dates: pd.Series) -> int:
    days = sorted(pd.to_datetime(dates, utc=True).dt.normalize().unique())
    best = current = 1
    for earlier, later in zip(days, days[1:]):
        current = current + 1 if (later - earlier).days == 1 else 1
        best = max(best, current)
    return best


def main() -> None:
    frame = pd.read_csv(
        ROOT / "data" / "processed" / "cleaned_daily.csv",
        parse_dates=["timestamp"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    audit = json.loads((ROOT / "data" / "processed" / "audit.json").read_text(encoding="utf-8"))
    provenance = json.loads(
        (OUT / "provenance_audit" / "upstream_join_audit_summary.json").read_text(
            encoding="utf-8"
        )
    )
    partition = json.loads((OUT / "future_test.json").read_text(encoding="utf-8"))
    development = frame.iloc[np.asarray(partition["development_indices"], dtype=int)].copy()

    target = frame["pm25"].dropna()
    summary_stats = {
        "n": int(target.size),
        "mean": float(target.mean()),
        "sd": float(target.std(ddof=1)),
        "min": float(target.min()),
        "median": float(target.median()),
        "p95": float(target.quantile(0.95)),
        "p99": float(target.quantile(0.99)),
        "max": float(target.max()),
        "skewness": float(target.skew()),
        "excess_kurtosis": float(target.kurt()),
        "n_above_25": int((target > 25).sum()),
        "pct_above_25": float((target > 25).mean() * 100),
        "n_above_50": int((target > 50).sum()),
        "pct_above_50": float((target > 50).mean() * 100),
        "n_above_100": int((target > 100).sum()),
        "pct_above_100": float((target > 100).mean() * 100),
    }

    # --- target boxplot -----------------------------------------------------
    fig, axes = plt.subplots(
        1, 2, figsize=(9.6, 2.7), gridspec_kw={"width_ratios": [2.0, 1.0]}
    )
    for axis, log_scale in zip(axes, (False, True)):
        axis.boxplot(
            target,
            vert=False,
            widths=0.42,
            whis=1.5,
            showfliers=True,
            patch_artist=True,
            boxprops={"facecolor": "#E8F3F3", "edgecolor": TEAL, "linewidth": 1.1},
            medianprops={"color": AMBER, "linewidth": 1.4},
            whiskerprops={"color": TEAL},
            capprops={"color": TEAL},
            flierprops={
                "marker": "o",
                "markersize": 2.6,
                "markerfacecolor": AMBER,
                "markeredgecolor": "none",
                "alpha": 0.55,
            },
        )
        axis.set(
            xlabel="PM2.5 (µg/m³)",
            yticks=[],
            title="Linear scale" if not log_scale else "Log scale",
        )
        if log_scale:
            axis.set_xscale("log")
        axis.grid(axis="x", alpha=0.2)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
    axes[0].text(
        0.985,
        0.72,
        f"median {summary_stats['median']:.2f} · P95 {summary_stats['p95']:.2f} · "
        f"max {summary_stats['max']:.2f}",
        transform=axes[0].transAxes,
        ha="right",
        va="center",
        fontsize=8,
        color=INK,
    )
    fig.suptitle("Boxplot of cleaned daily PM2.5 (whiskers = 1.5×IQR)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "pm25_boxplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- predictor missingness overview ------------------------------------
    columns = [
        "pm25",
        "temperature_celsius",
        "pressure_mb",
        "wind_u",
        "wind_v",
        "no2",
        "co",
        "o3",
        "aod",
    ]
    missing_pct = frame[columns].isna().mean().mul(100)
    colors = [
        NAVY if column == "pm25" else (AMBER if column in {"no2", "co", "o3"} else TEAL)
        for column in columns
    ]
    fig, axis = plt.subplots(figsize=(6.4, 2.7))
    axis.barh(columns[::-1], missing_pct.loc[columns[::-1]], color=colors[::-1], height=0.62)
    for position, column in enumerate(columns[::-1]):
        value = missing_pct.loc[column]
        axis.text(
            value + 0.08,
            position,
            f"{value:.2f}%",
            va="center",
            fontsize=8,
            color=INK,
        )
    axis.set(
        xlabel="Missing values (% of cleaned rows)",
        xlim=(0, 7.2),
        title="Predictor missingness after target cleaning",
    )
    axis.grid(axis="x", alpha=0.18)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "predictor_missingness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- development station coverage --------------------------------------
    coverage_rows = []
    for station, group in development.groupby("station_id"):
        coverage_rows.append(
            {
                "station_id": station,
                "n_days": int(group["timestamp"].dt.normalize().nunique()),
                "longest_run": longest_run(group["timestamp"]),
            }
        )
    coverage = pd.DataFrame(coverage_rows).sort_values("n_days", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})
    axes[0].bar(
        np.arange(len(coverage)),
        coverage["n_days"],
        color=TEAL,
        width=0.82,
        alpha=0.88,
    )
    axes[0].plot(
        np.arange(len(coverage)),
        coverage["longest_run"],
        color=AMBER,
        lw=1.1,
        label="longest uninterrupted run",
    )
    axes[0].set(
        xlabel="Stations sorted by observation count",
        ylabel="Consecutive daily observations",
        title="Development coverage by station",
        xticks=[],
    )
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1].hist(coverage["longest_run"], bins=range(1, 35), color=NAVY, alpha=0.82)
    axes[1].axvline(14, color=AMBER, ls="--", lw=1.1, label="ACF/PACF ≥14")
    axes[1].axvline(30, color="#C0392B", ls="--", lw=1.1, label="formal tests ≥30")
    axes[1].set(
        xlabel="Longest uninterrupted daily run",
        ylabel="Stations",
        title="Distribution of longest daily runs",
    )
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.18)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        FIG / "station_coverage_development.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

    # --- pollutant missingness heatmap -------------------------------------
    missingness = pd.read_csv(OUT / "pollutant_missingness_station_month_development.csv")
    months = sorted(missingness["month"].unique())
    stations = (
        missingness.groupby("station_id")["pct_missing"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    variables = ["NO2", "CO", "O3"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    image = None
    for axis, variable in zip(axes, variables):
        table = (
            missingness.loc[missingness["variable"].eq(variable)]
            .pivot_table(
                index="station_id",
                columns="month",
                values="pct_missing",
                aggfunc="mean",
            )
            .reindex(index=stations, columns=months)
        )
        image = axis.imshow(
            table.to_numpy(dtype=float),
            aspect="auto",
            interpolation="nearest",
            cmap="Reds",
            vmin=0,
            vmax=100,
        )
        axis.set_facecolor("#E6E6E6")
        axis.set(title=variable, xlabel="Month")
        axis.set_yticks([])
        axis.set_xticks(np.arange(0, len(months), 24))
        axis.set_xticklabels([months[i] for i in range(0, len(months), 24)], rotation=0)
    axes[0].set_ylabel("Stations")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.012)
    colorbar.set_label("% missing")
    fig.suptitle(
        "Station-days missing by station and month (development partition)",
        fontsize=10.5,
    )
    fig.savefig(
        FIG / "pollutant_missingness_heatmap_development.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    missingness_summary = {
        variable: {
            "station_months": int(len(group)),
            "station_months_with_missing": int((group["n_missing"] > 0).sum()),
            "stations_with_missing": int(
                group.loc[group["n_missing"] > 0, "station_id"].nunique()
            ),
            "max_station_month_pct_missing": float(group["pct_missing"].max()),
            "station_days": int(group["n_station_days"].sum()),
        }
        for variable, group in missingness.groupby("variable")
    }

    folds = {}
    for name in ("temporal", "spatial", "standard"):
        payload = json.loads((ROOT / "folds" / f"folds_{name}.json").read_text(encoding="utf-8"))
        folds[name] = {
            "scheme": payload.get("scheme"),
            "status": payload.get("status", payload.get("cv_purpose", "defined")),
            "n_splits": payload.get("n_splits"),
        }
    future = json.loads((OUT / "future_test.json").read_text(encoding="utf-8"))

    output = {
        "source": "derived only from saved project artifacts",
        "cleaned_panel": {
            "rows": int(len(frame)),
            "stations": int(frame["station_id"].nunique()),
            "dates": int(frame["timestamp"].dt.normalize().nunique()),
            "date_min": frame["timestamp"].min().isoformat(),
            "date_max": frame["timestamp"].max().isoformat(),
        },
        "cleaning": audit["cleaning"],
        "target": summary_stats,
        "development_partition": {
            "rows": int(len(development)),
            "stations": int(development["station_id"].nunique()),
            "end": str(partition["development_end"]),
            "stations_longest_run_ge_14": int((coverage["longest_run"] >= 14).sum()),
            "stations_longest_run_ge_30": int((coverage["longest_run"] >= 30).sum()),
            "median_longest_run": float(coverage["longest_run"].median()),
            "max_longest_run": int(coverage["longest_run"].max()),
        },
        "station_identity": audit["station_identity_reconciliation"]
        | {
            "cleaned_sensor_names": int(frame["sensor_name"].nunique()),
        },
        "provenance": provenance["combined_bundle_correspondence"],
        "pollutant_missingness": missingness_summary,
        "folds": folds,
        "future_holdout": {
            "scheme": future.get("scheme"),
            "status": future.get("status"),
        },
    }
    (OUT / "part1_evidence_summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
