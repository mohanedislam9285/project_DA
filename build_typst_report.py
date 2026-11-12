import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "16"
REPORT_SOURCE = ROOT / "reports" / "source" / "final_report.typ"
OUTPUT = ROOT / "reports" / "final_report.pdf"


def locate_typst() -> str | None:
    installed = shutil.which("typst")
    if installed:
        return installed
    local_app = Path.home() / "AppData" / "Local" / "Programs" / "typst-tinymist"
    candidates = (
        local_app / "typst.exe",
        local_app / "typst-x86_64-pc-windows-msvc" / "typst.exe",
    )
    return next((str(path) for path in candidates if path.is_file()), None)


def locate_tinymist() -> str | None:
    installed = shutil.which("tinymist")
    if installed:
        return installed
    local_app = Path.home() / "AppData" / "Local" / "Programs" / "typst-tinymist"
    candidate = local_app / "tinymist.exe"
    return str(candidate) if candidate.is_file() else None


def build_training_history_figure() -> None:
    history = pd.read_csv(
        RESULTS / "histories" / "cnn_lstm_optuna_final_inner_history.csv"
    )
    required = {"epoch", "loss", "val_loss"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(
            f"Training history is missing required columns: {sorted(missing)}"
        )
    if history.empty or history[["epoch", "loss", "val_loss"]].isna().any().any():
        raise ValueError("Training history has no usable loss observations.")

    fig, axis = plt.subplots(figsize=(8, 3.2))
    axis.plot(
        history["epoch"], history["loss"], label="Training MSE", color="#087F80", lw=2
    )
    axis.plot(
        history["epoch"],
        history["val_loss"],
        label="Inner-validation MSE",
        color="#D9822B",
        lw=2,
    )
    best = history.loc[history["val_loss"].idxmin()]
    axis.scatter([best["epoch"]], [best["val_loss"]], color="#D9822B", zorder=3)
    axis.annotate(
        f"best inner validation: epoch {int(best['epoch'])}",
        (best["epoch"], best["val_loss"]),
        xytext=(8, 9),
        textcoords="offset points",
        fontsize=8,
    )
    axis.set(xlabel="Epoch", ylabel="MSE (µg/m³)²")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    destination = RESULTS / "figures" / "training_validation_loss_typst.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not REPORT_SOURCE.is_file():
        raise FileNotFoundError(f"Typst source not found: {REPORT_SOURCE}")
    typst = locate_typst()
    if typst is None:
        raise FileNotFoundError(
            "Typst CLI is required. Install Typst, then run this script again."
        )
    tinymist = locate_tinymist()
    if tinymist is None:
        raise FileNotFoundError(
            "Tinymist is required. Install Tinymist, then run this script again."
        )
    build_training_history_figure()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [tinymist, "lint", "--root", str(ROOT), str(REPORT_SOURCE)],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [typst, "compile", "--root", str(ROOT), str(REPORT_SOURCE), str(OUTPUT)],
        cwd=ROOT,
        check=True,
    )
    print(f"Built {OUTPUT}")


if __name__ == "__main__":
    main()
