"""
Evaluation entry point.

Supports: pooled, LOSO, and few-shot LOSO evaluation protocols.

Usage:
    python src/evaluate.py --protocol pooled --ckpt results/checkpoints/finetune/classifier-last.ckpt
    python src/evaluate.py --protocol loso --ckpt results/checkpoints/finetune/classifier-last.ckpt --num_sensors 5
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

from src.data.datamodule import ElectroSenseDataModule, PSDDataset
from src.models.classifier import PSDClassifier
from src.utils.metrics import (
    compute_all_metrics,
    save_confusion_matrix_plot,
    save_reliability_diagram,
)
from src.utils.logging_utils import save_metrics_table, save_loso_summary


CLASS_NAMES = ["dab", "dvbt", "fm", "gsm", "lte", "tetra"]


def collect_predictions(
    model: PSDClassifier,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Run model on dataloader and collect all predictions."""
    model.eval()
    model.to(device)

    all_logits = []
    all_labels = []
    all_snr = []

    with torch.no_grad():
        for batch in dataloader:
            psd, labels, sensor_ids, snr = batch
            psd = psd.to(device)
            logits = model(psd)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_snr.append(snr)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    snr = torch.cat(all_snr, dim=0)

    probs = torch.softmax(logits, dim=-1).numpy()
    preds = logits.argmax(dim=-1).numpy()
    labels = labels.numpy()
    snr = snr.numpy()

    return {
        "logits": logits.numpy(),
        "probs": probs,
        "preds": preds,
        "labels": labels,
        "snr": snr,
    }


def evaluate_pooled(
    model: PSDClassifier,
    dm: ElectroSenseDataModule,
    results_dir: str,
    tag: str = "pretrained",
    device: torch.device = torch.device("cpu"),
):
    """Standard pooled evaluation."""
    dm.setup()
    results = collect_predictions(model, dm.test_dataloader(), device)

    metrics = compute_all_metrics(
        results["labels"], results["preds"], results["probs"], CLASS_NAMES
    )

    print("\n=== Pooled Evaluation ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    if "ece" in metrics:
        print(f"ECE: {metrics['ece']:.4f}")
    print(metrics["classification_report"])

    save_dir = Path(results_dir)
    save_metrics_table(metrics, str(save_dir / f"pooled_{tag}.csv"))
    save_confusion_matrix_plot(
        metrics["confusion_matrix"], CLASS_NAMES,
        str(save_dir / f"confusion_matrix_pooled_{tag}.png"),
        title=f"Pooled ({tag})",
    )

    if "ece" in metrics:
        save_reliability_diagram(
            results["probs"], results["labels"],
            str(save_dir / f"reliability_pooled_{tag}.png"),
        )

    return metrics


def evaluate_loso(
    model_cls,
    model_kwargs: dict,
    ckpt_path: str,
    manifest_path: str,
    splits_dir: str,
    fraction: float,
    seed: int,
    num_sensors: int,
    results_dir: str,
    device: torch.device,
    psd_length: int = 200,
    batch_size: int = 128,
):
    """Leave-One-Sensor-Out evaluation."""
    split_file = Path(splits_dir) / f"loso_seed{seed}_frac{fraction}.json"
    if not split_file.exists():
        print(f"LOSO split file not found: {split_file}")
        return {}

    with open(split_file) as f:
        loso_splits = json.load(f)

    sensor_ids = list(loso_splits.keys())
    if num_sensors and num_sensors < len(sensor_ids):
        sensor_ids = sensor_ids[:num_sensors]

    import pandas as pd
    manifest = pd.read_csv(manifest_path)

    per_sensor_results = {}

    for i, sensor_id in enumerate(sensor_ids):
        print(f"\n--- LOSO Sensor {i+1}/{len(sensor_ids)}: {sensor_id} ---")

        split = loso_splits[sensor_id]

        # Load model fresh from checkpoint
        model = PSDClassifier.load_from_checkpoint(ckpt_path, **model_kwargs)
        model.eval()
        model.to(device)

        # Create test dataset for this sensor
        test_dataset = PSDDataset(manifest, split["test"], psd_length=psd_length)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        if len(test_dataset) == 0:
            print(f"  No test samples for sensor {sensor_id}, skipping.")
            continue

        results = collect_predictions(model, test_loader, device)
        metrics = compute_all_metrics(
            results["labels"], results["preds"], results["probs"], CLASS_NAMES
        )

        print(f"  Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")
        per_sensor_results[sensor_id] = metrics

    # Summary
    if per_sensor_results:
        accs = [m["accuracy"] for m in per_sensor_results.values()]
        f1s = [m["macro_f1"] for m in per_sensor_results.values()]
        print(f"\n=== LOSO Summary ({len(per_sensor_results)} sensors) ===")
        print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"Macro-F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"Worst sensor acc: {np.min(accs):.4f}")

        save_loso_summary(per_sensor_results, str(Path(results_dir) / "loso_results.csv"))

    return per_sensor_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--protocol", type=str, default="pooled",
                        choices=["pooled", "loso", "fewshot_loso"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/finetune_classify.yaml")
    parser.add_argument("--eval_config", type=str, default="configs/eval_loso.yaml")
    parser.add_argument("--tag", type=str, default="pretrained")
    parser.add_argument("--num_sensors", type=int, default=None)
    parser.add_argument("--data.fraction", type=float, default=None, dest="fraction")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Load configs
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if "defaults" in cfg and "base" in cfg["defaults"]:
        with open(cfg["defaults"]["base"]) as f:
            base = yaml.safe_load(f)
        from src.train_pretrain import deep_merge
        cfg = deep_merge(base, cfg)

    with open(args.eval_config) as f:
        eval_cfg = yaml.safe_load(f)

    seed = args.seed or cfg.get("seed", 42)
    fraction = args.fraction or cfg.get("data", {}).get("fraction", 1.0)
    results_dir = cfg.get("results_dir", "results/")
    psd_length = cfg.get("dataset", {}).get("psd_length", 200)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_classifier_ckpt(ckpt_path):
        """Load PSDClassifier checkpoint, handling torch.compile _orig_mod. prefix."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = ckpt["state_dict"]
        # Strip _orig_mod. prefix added by torch.compile
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        hparams = ckpt.get("hyper_parameters", {})
        # Remove class_weights from hparams if present (it's a buffer, not a constructor arg)
        hparams.pop("class_weights", None)
        model = PSDClassifier(**hparams)
        model.load_state_dict(cleaned, strict=False)
        return model

    if args.protocol == "pooled":
        # Load model
        model = _load_classifier_ckpt(args.ckpt)

        splits_dir = cfg.get("dataset", {}).get("splits_dir", "data/splits/")
        split_file = Path(splits_dir) / f"pooled_seed{seed}_frac{fraction}.json"

        dm = ElectroSenseDataModule(
            manifest_path=cfg.get("dataset", {}).get("manifest_path", "data/manifest.csv"),
            split_path=str(split_file),
            psd_length=psd_length,
            batch_size=cfg.get("training", {}).get("batch_size", 128),
        )

        evaluate_pooled(model, dm, results_dir, tag=args.tag, device=device)

    elif args.protocol == "loso":
        num_sensors = args.num_sensors or eval_cfg.get("evaluation", {}).get("num_loso_sensors", 10)

        evaluate_loso(
            model_cls=PSDClassifier,
            model_kwargs={},
            ckpt_path=args.ckpt,
            manifest_path=cfg.get("dataset", {}).get("manifest_path", "data/manifest.csv"),
            splits_dir=cfg.get("dataset", {}).get("splits_dir", "data/splits/"),
            fraction=fraction,
            seed=seed,
            num_sensors=num_sensors,
            results_dir=results_dir,
            device=device,
            psd_length=psd_length,
        )

    else:
        print(f"Protocol '{args.protocol}' not yet implemented.")
        sys.exit(1)


if __name__ == "__main__":
    main()
