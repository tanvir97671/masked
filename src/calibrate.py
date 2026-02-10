"""
Post-hoc calibration entry point.

Fits temperature scaling on validation set, then evaluates on test set.

Usage:
    python src/calibrate.py --ckpt results/checkpoints/finetune/classifier-last.ckpt
"""

import argparse
import json
import sys
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datamodule import ElectroSenseDataModule
from src.models.classifier import PSDClassifier
from src.utils.calibration import TemperatureScaling, SNRAwareTemperature
from src.utils.metrics import compute_all_metrics, expected_calibration_error, save_reliability_diagram
from src.utils.logging_utils import save_metrics_table
from src.evaluate import collect_predictions, CLASS_NAMES


def main():
    parser = argparse.ArgumentParser(description="Post-hoc Calibration")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/finetune_classify.yaml")
    parser.add_argument("--method", type=str, default="both",
                        choices=["temperature", "snr_aware", "both"])
    parser.add_argument("--data.fraction", type=float, default=None, dest="fraction")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if "defaults" in cfg and "base" in cfg["defaults"]:
        with open(cfg["defaults"]["base"]) as f:
            base = yaml.safe_load(f)
        from src.train_pretrain import deep_merge
        cfg = deep_merge(base, cfg)

    seed = args.seed or cfg.get("seed", 42)
    fraction = args.fraction or cfg.get("data", {}).get("fraction", 1.0)
    psd_length = cfg.get("dataset", {}).get("psd_length", 200)
    results_dir = cfg.get("results_dir", "results/")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = PSDClassifier.load_from_checkpoint(args.ckpt)
    model.eval()
    model.to(device)

    # DataModule
    splits_dir = cfg.get("dataset", {}).get("splits_dir", "data/splits/")
    split_file = Path(splits_dir) / f"pooled_seed{seed}_frac{fraction}.json"

    dm = ElectroSenseDataModule(
        manifest_path=cfg.get("dataset", {}).get("manifest_path", "data/manifest.csv"),
        split_path=str(split_file),
        psd_length=psd_length,
        batch_size=cfg.get("training", {}).get("batch_size", 128),
    )
    dm.setup()

    # Collect val predictions for fitting
    val_results = collect_predictions(model, dm.val_dataloader(), device)
    val_logits = torch.tensor(val_results["logits"])
    val_labels = torch.tensor(val_results["labels"])
    val_snr = torch.tensor(val_results["snr"])

    # Collect test predictions for evaluation
    test_results = collect_predictions(model, dm.test_dataloader(), device)
    test_logits = torch.tensor(test_results["logits"])
    test_labels = torch.tensor(test_results["labels"])
    test_snr = torch.tensor(test_results["snr"])

    # Before calibration
    before_ece = expected_calibration_error(
        torch.softmax(test_logits, dim=-1).numpy(), test_labels.numpy()
    )
    print(f"\nBefore calibration — ECE: {before_ece:.4f}")

    save_dir = Path(results_dir) / "calibration"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) Temperature scaling
    if args.method in ("temperature", "both"):
        print("\n--- Temperature Scaling ---")
        ts = TemperatureScaling()
        ts.fit(val_logits, val_labels)

        calibrated_logits = ts(test_logits)
        calibrated_probs = torch.softmax(calibrated_logits, dim=-1).numpy()
        after_ece = expected_calibration_error(calibrated_probs, test_labels.numpy())
        print(f"After temperature scaling — ECE: {after_ece:.4f}")

        metrics = compute_all_metrics(
            test_labels.numpy(),
            calibrated_logits.argmax(dim=-1).numpy(),
            calibrated_probs,
            CLASS_NAMES,
        )
        metrics["ece_before"] = before_ece
        metrics["temperature"] = ts.temperature.item()
        save_metrics_table(metrics, str(save_dir / "temp_scaling_results.csv"))

        save_reliability_diagram(
            calibrated_probs, test_labels.numpy(),
            str(save_dir / "reliability_temp_scaling.png"),
            title="After Temperature Scaling",
        )

    # 2) SNR-aware temperature
    if args.method in ("snr_aware", "both"):
        print("\n--- SNR-Aware Temperature ---")
        snr_ts = SNRAwareTemperature(hidden_dim=32)
        snr_ts.fit(val_logits, val_labels, val_snr, lr=1e-3, epochs=200)

        snr_calibrated_logits = snr_ts(test_logits, test_snr)
        snr_calibrated_probs = torch.softmax(snr_calibrated_logits, dim=-1).detach().numpy()
        snr_after_ece = expected_calibration_error(snr_calibrated_probs, test_labels.numpy())
        print(f"After SNR-aware scaling — ECE: {snr_after_ece:.4f}")

        metrics = compute_all_metrics(
            test_labels.numpy(),
            snr_calibrated_logits.argmax(dim=-1).detach().numpy(),
            snr_calibrated_probs,
            CLASS_NAMES,
        )
        metrics["ece_before"] = before_ece
        save_metrics_table(metrics, str(save_dir / "snr_aware_results.csv"))

        save_reliability_diagram(
            snr_calibrated_probs, test_labels.numpy(),
            str(save_dir / "reliability_snr_aware.png"),
            title="After SNR-Aware Calibration",
        )

    print("\nCalibration complete. Results saved to", save_dir)


if __name__ == "__main__":
    main()
