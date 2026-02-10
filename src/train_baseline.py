"""
Entry point: Train CNN baseline from scratch (no SSL).

Usage:
    python src/train_baseline.py --config configs/finetune_classify.yaml
    python src/train_baseline.py --config configs/finetune_classify.yaml --data.fraction 0.25
"""

import argparse
import json
import sys
from pathlib import Path

import lightning as L
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datamodule import ElectroSenseDataModule
from src.models.classifier import PSDClassifier


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if "defaults" in cfg and "base" in cfg["defaults"]:
        with open(cfg["defaults"]["base"], "r") as f:
            base = yaml.safe_load(f)
        from src.train_pretrain import deep_merge
        return deep_merge(base, cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="CNN Baseline Training")
    parser.add_argument("--config", type=str, default="configs/finetune_classify.yaml")
    parser.add_argument("--data.fraction", type=float, default=None, dest="fraction")
    parser.add_argument("--encoder_type", type=str, default="cnn",
                        choices=["cnn", "transformer"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = args.seed or cfg.get("seed", 42)
    L.seed_everything(seed, workers=True)

    fraction = args.fraction or cfg.get("data", {}).get("fraction", 1.0)

    # Split file
    splits_dir = cfg.get("dataset", {}).get("splits_dir", "data/splits/")
    split_file = Path(splits_dir) / f"pooled_seed{seed}_frac{fraction}.json"

    if not split_file.exists():
        print(f"Split file not found: {split_file}")
        sys.exit(1)

    psd_length = cfg.get("dataset", {}).get("psd_length", 200)
    batch_size = cfg.get("training", {}).get("batch_size", 128)
    num_workers = cfg.get("num_workers", 4)

    dm = ElectroSenseDataModule(
        manifest_path=cfg.get("dataset", {}).get("manifest_path", "data/manifest.csv"),
        split_path=str(split_file),
        psd_length=psd_length,
        batch_size=batch_size,
        num_workers=num_workers,
        contrastive=False,
    )
    dm.setup()

    # Class weights
    num_classes = cfg.get("num_classes", 6)
    class_weights = dm.get_class_weights()[:num_classes]

    # Model: from scratch, no pretrained
    head_cfg = cfg.get("head", {})
    train_cfg = cfg.get("training", {})

    model = PSDClassifier(
        psd_length=psd_length,
        d_model=128,
        num_classes=num_classes,
        head_type=head_cfg.get("type", "mlp"),
        head_hidden=head_cfg.get("hidden_dim", 128),
        head_dropout=head_cfg.get("dropout", 0.2),
        encoder_type=args.encoder_type,
        pretrained_ckpt=None,
        freeze_encoder=False,
        lr=train_cfg.get("lr", 5e-5),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        max_epochs=train_cfg.get("max_epochs", 50),
        class_weights=class_weights,
    )

    # Callbacks
    results_dir = cfg.get("results_dir", "results/")
    ckpt_dir = Path(results_dir) / "checkpoints" / f"baseline_{args.encoder_type}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=f"baseline_{args.encoder_type}" + "-{epoch:02d}-{val_macro_f1:.4f}",
            monitor="val_macro_f1",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        L.pytorch.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=train_cfg.get("early_stopping_patience", 10),
        ),
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = L.Trainer(
        max_epochs=train_cfg.get("max_epochs", 50),
        accelerator="auto",
        devices=1,
        precision=cfg.get("precision", "16-mixed"),
        callbacks=callbacks,
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        default_root_dir=results_dir,
        deterministic=True,
    )

    print("=" * 60)
    print(f"Baseline Training ({args.encoder_type.upper()} from scratch)")
    print(f"  Data fraction: {fraction}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {train_cfg.get('max_epochs', 50)}")
    print("=" * 60)

    trainer.fit(model, datamodule=dm)

    best_path = trainer.checkpoint_callback.best_model_path
    print(f"\nBest baseline checkpoint: {best_path}")

    meta = {
        "best_ckpt": best_path,
        "encoder_type": args.encoder_type,
        "seed": seed,
        "fraction": fraction,
    }
    with open(Path(results_dir) / f"baseline_{args.encoder_type}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
