"""
Evaluation metrics for PSD classification.

Includes: accuracy, macro-F1, per-class F1, confusion matrix, ECE.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: (N,) ground truth labels.
        y_pred: (N,) predicted labels.
        y_prob: (N, C) predicted probabilities (for ECE).
        class_names: list of class name strings.

    Returns:
        Dict with all metrics.
    """
    metrics = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["macro_precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["macro_recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    if class_names is not None:
        for i, name in enumerate(class_names):
            if i < len(per_class_f1):
                metrics[f"f1_{name}"] = float(per_class_f1[i])
    else:
        for i, f in enumerate(per_class_f1):
            metrics[f"f1_class_{i}"] = float(f)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm

    # Classification report (string)
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    metrics["classification_report"] = report

    # ECE
    if y_prob is not None:
        metrics["ece"] = float(expected_calibration_error(y_prob, y_true))

    return metrics


def expected_calibration_error(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = Σ (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args:
        y_prob: (N, C) predicted class probabilities.
        y_true: (N,) ground truth labels.
        n_bins: Number of confidence bins.

    Returns:
        ECE value (lower is better).
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

    return ece


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 300,
) -> None:
    """Save a publication-quality confusion matrix plot."""
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")


def save_reliability_diagram(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    save_path: str,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    dpi: int = 300,
) -> None:
    """Save a reliability diagram (calibration plot)."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_accs.append(accuracies[in_bin].mean())
            bin_counts.append(in_bin.sum())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.bar(bin_centers, bin_accs, width=1.0 / n_bins, alpha=0.6, edgecolor="black", label="Model")
    ax1.set_xlabel("Confidence", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.bar(bin_centers, bin_counts, width=1.0 / n_bins, alpha=0.6, color="gray", edgecolor="black")
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Reliability diagram saved to {save_path}")
