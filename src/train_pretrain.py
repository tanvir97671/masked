"""
Entry point: SSL pretraining (MPAE + SICR).

Usage:
    python src/train_pretrain.py --config configs/pretrain_mpae.yaml
    python src/train_pretrain.py --config configs/pretrain_mpae.yaml --data.fraction 0.25
"""

import argparse
import sys
from pathlib import Path

import lightning as L
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datamodule import ElectroSenseDataModule
from src.models.mpae import MaskedPSDAutoencoder


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Load base config if specified
    if "defaults" in cfg and "base" in cfg["defaults"]:
        base_path = cfg["defaults"]["base"]
        with open(base_path, "r") as f:
            base_cfg = yaml.safe_load(f)
        # Merge (config overrides base)
        merged = deep_merge(base_cfg, cfg)
        return merged
    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description="MPAE + SICR SSL Pretraining")
    parser.add_argument("--config", type=str, default="configs/pretrain_mpae.yaml")
    parser.add_argument("--data.fraction", type=float, default=None, dest="fraction")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = args.seed or cfg.get("seed", 42)
    L.seed_everything(seed, workers=True)

    fraction = args.fraction or cfg.get("data", {}).get("fraction", 1.0)

    # Determine split file
    splits_dir = cfg.get("dataset", {}).get("splits_dir", "data/splits/")
    split_file = Path(splits_dir) / f"pooled_seed{seed}_frac{fraction}.json"

    if not split_file.exists():
        print(f"Split file not found: {split_file}")
        print("Run: python src/data/splits.py --fraction", fraction, "--seed", seed)
        sys.exit(1)

    # DataModule (contrastive mode for SICR)
    psd_length = cfg.get("dataset", {}).get("psd_length", 200)
    batch_size = cfg.get("training", {}).get("batch_size", 256)
    num_workers = cfg.get("num_workers", 4)

    dm = ElectroSenseDataModule(
        manifest_path=cfg.get("dataset", {}).get("manifest_path", "data/manifest.csv"),
        split_path=str(split_file),
        psd_length=psd_length,
        batch_size=batch_size,
        num_workers=num_workers,
        contrastive=True,  # Enable cross-sensor pairs for SICR
    )

    # Model
    enc_cfg = cfg.get("encoder", {})
    mpae_cfg = cfg.get("mpae", {})
    sicr_cfg = cfg.get("sicr", {})
    train_cfg = cfg.get("training", {})

    model = MaskedPSDAutoencoder(
        psd_length=psd_length,
        patch_size=enc_cfg.get("patch_size", 16),
        d_model=enc_cfg.get("d_model", 128),
        n_layers=enc_cfg.get("n_layers", 4),
        n_heads=enc_cfg.get("n_heads", 4),
        ffn_dim=enc_cfg.get("ffn_dim", 256),
        dropout=enc_cfg.get("dropout", 0.1),
        mask_ratio=mpae_cfg.get("mask_ratio", 0.6),
        decoder_dim=mpae_cfg.get("decoder_dim", 64),
        decoder_layers=mpae_cfg.get("decoder_layers", 2),
        lambda_sicr=sicr_cfg.get("lambda_sicr", 0.1),
        sicr_temperature=sicr_cfg.get("temperature", 0.07),
        sicr_proj_dim=sicr_cfg.get("projection_dim", 64),
        encoder_type=enc_cfg.get("type", "transformer"),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        warmup_epochs=train_cfg.get("warmup_epochs", 5),
        max_epochs=train_cfg.get("max_epochs", 80),
    )

    # Callbacks
    log_cfg = cfg.get("logging", {})
    results_dir = cfg.get("results_dir", "results/")
    ckpt_dir = Path(results_dir) / "checkpoints" / "pretrain"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="mpae-{epoch:02d}-{val_loss:.4f}",
            monitor=log_cfg.get("monitor", "val_loss"),
            mode="min",
            save_top_k=log_cfg.get("save_top_k", 2),
            save_last=True,
        ),
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    # Trainer
    trainer = L.Trainer(
        max_epochs=train_cfg.get("max_epochs", 80),
        accelerator="auto",
        devices=1,
        precision=cfg.get("precision", "16-mixed"),
        callbacks=callbacks,
        log_every_n_steps=log_cfg.get("log_every_n_steps", 50),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        default_root_dir=results_dir,
        deterministic=True,
    )

    print("=" * 60)
    print("MPAE + SICR Self-Supervised Pretraining")
    print(f"  Data fraction: {fraction}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {train_cfg.get('max_epochs', 80)}")
    print(f"  Mask ratio: {mpae_cfg.get('mask_ratio', 0.6)}")
    print(f"  SICR lambda: {sicr_cfg.get('lambda_sicr', 0.1)}")
    print(f"  Encoder: {enc_cfg.get('type', 'transformer')}")
    print(f"  Checkpoint dir: {ckpt_dir}")
    print("=" * 60)

    trainer.fit(model, datamodule=dm)

    # Save best checkpoint path
    best_path = trainer.checkpoint_callback.best_model_path
    print(f"\nBest checkpoint: {best_path}")

    meta = {"best_ckpt": best_path, "seed": seed, "fraction": fraction}
    import json
    with open(Path(results_dir) / "pretrain_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
