"""
Logging and paper artifact utilities.

Saves metrics to CSV, generates paper-ready tables and plots.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_metrics_table(
    metrics: Dict,
    save_path: str,
    exclude_keys: List[str] = None,
) -> None:
    """
    Save metrics dict to CSV, excluding non-scalar values.
    """
    if exclude_keys is None:
        exclude_keys = ["confusion_matrix", "classification_report"]

    scalar_metrics = {
        k: v for k, v in metrics.items()
        if k not in exclude_keys and not isinstance(v, (np.ndarray, list, dict))
    }

    df = pd.DataFrame([scalar_metrics])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Metrics saved to {save_path}")


def save_loso_summary(
    per_sensor_metrics: Dict[str, Dict],
    save_path: str,
) -> None:
    """
    Save LOSO per-sensor results + aggregate stats.
    """
    rows = []
    for sensor_id, metrics in per_sensor_metrics.items():
        row = {"sensor_id": sensor_id}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add summary row
    summary = {"sensor_id": "MEAN"}
    for col in df.select_dtypes(include=[np.number]).columns:
        summary[col] = df[col].mean()
    summary_std = {"sensor_id": "STD"}
    for col in df.select_dtypes(include=[np.number]).columns:
        summary_std[col] = df[col].std()

    df = pd.concat([df, pd.DataFrame([summary, summary_std])], ignore_index=True)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"LOSO summary saved to {save_path}")


def save_fewshot_curve(
    k_values: List[int],
    accuracies: Dict[str, List[float]],
    save_path: str,
    title: str = "Few-Shot LOSO Accuracy vs K",
    dpi: int = 300,
) -> None:
    """
    Save few-shot accuracy vs K plot.

    Args:
        k_values: list of K values.
        accuracies: dict mapping method name -> list of acc values (one per K).
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for method_name, accs in accuracies.items():
        ax.plot(k_values, accs, "o-", label=method_name, linewidth=2, markersize=6)

    ax.set_xlabel("K (labels per class)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Few-shot curve saved to {save_path}")


def generate_paper_tables(
    results_dir: str = "results/",
    output_dir: str = "results/paper_tables/",
) -> None:
    """
    Generate paper-ready tables from results CSVs.
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Main comparison table (pooled)
    pooled_csvs = list(results_path.glob("pooled_*.csv"))
    if pooled_csvs:
        dfs = []
        for csv_path in pooled_csvs:
            df = pd.read_csv(csv_path)
            df["method"] = csv_path.stem.replace("pooled_", "")
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(output_path / "main_comparison.csv", index=False)

        # LaTeX table
        latex = combined.to_latex(index=False, float_format="%.4f")
        with open(output_path / "main_comparison.tex", "w") as f:
            f.write(latex)
        print("Paper comparison table generated.")

    # LOSO summary
    loso_csvs = list(results_path.glob("loso_*.csv"))
    if loso_csvs:
        for csv_path in loso_csvs:
            df = pd.read_csv(csv_path)
            latex = df.to_latex(index=False, float_format="%.4f")
            tex_path = output_path / f"{csv_path.stem}.tex"
            with open(tex_path, "w") as f:
                f.write(latex)
        print("LOSO tables generated.")

    print(f"All paper tables saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--output_dir", type=str, default="results/paper_tables/")
    args = parser.parse_args()
    generate_paper_tables(args.results_dir, args.output_dir)
