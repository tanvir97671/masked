"""
Entry point: Supervised fine-tuning on pretrained encoder.

Usage:
    python src/train_finetune.py --config configs/finetune_classify.yaml \
        --pretrained_ckpt results/checkpoints/pretrain/mpae-last.ckpt
    python src/train_finetune.py --config configs/finetune_classify.yaml \
        --data.fraction 0.25 --pretrained_ckpt results/checkpoints/pretrain/mpae-last.ckpt
"""

import argparse
import json
import os
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
    parser = argparse.ArgumentParser(description="Supervised Fine-tuning")
    parser.add_argument("--config", type=str, default="configs/finetune_classify.yaml")
    parser.add_argument("--pretrained_ckpt", "--pretrained", type=str, default=None, dest="pretrained_ckpt")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest CSV")
    parser.add_argument("--split", type=str, default=None, help="Path to split JSON file")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--accelerator", type=str, default=None, help="gpu/cpu/auto")
    parser.add_argument("--devices", type=int, default=None, help="Number of devices")
    parser.add_argument("--data.fraction", type=float, default=None, dest="fraction")
    parser.add_argument("--freeze_encoder", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = args.seed or cfg.get("seed", 42)
    L.seed_everything(seed, workers=True)

    fraction = args.fraction or cfg.get("data", {}).get("fraction", 1.0)

    # Try to find pretrained checkpoint from meta file
    pretrained_ckpt = args.pretrained_ckpt
    if pretrained_ckpt is None:
        pretrained_ckpt = cfg.get("model", {}).get("pretrained_ckpt", None)
    if pretrained_ckpt is None:
        meta_path = Path(cfg.get("results_dir", "results/")) / "pretrain_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            pretrained_ckpt = meta.get("best_ckpt")
            print(f"Auto-detected pretrained checkpoint: {pretrained_ckpt}")

    # Split file (CLI --split overrides auto-detection)
    if args.split:
        split_file = Path(args.split)
    else:
        splits_dir = cfg.get("dataset", {}).get("splits_dir", "data/splits/")
        split_file = Path(splits_dir) / f"pooled_seed{seed}_frac{fraction}.json"

    if not split_file.exists():
        print(f"Split file not found: {split_file}")
        sys.exit(1)

    # DataModule (standard, non-contrastive)
    psd_length = cfg.get("dataset", {}).get("psd_length", 200)
    batch_size = args.batch_size or cfg.get("training", {}).get("batch_size", 128)
    num_workers = cfg.get("num_workers", 4)
    manifest_path = args.manifest or cfg.get("dataset", {}).get("manifest_path", "data/manifest.csv")

    dm = ElectroSenseDataModule(
        manifest_path=manifest_path,
        split_path=str(split_file),
        psd_length=psd_length,
        batch_size=batch_size,
        num_workers=num_workers,
        contrastive=False,
        pin_memory=cfg.get("pin_memory", True),
        persistent_workers=cfg.get("persistent_workers", True),
        prefetch_factor=cfg.get("prefetch_factor", 4),
    )
    dm.setup()

    # Get class weights
    model_cfg = cfg.get("model", {})
    class_weights = None
    if model_cfg.get("class_weights") == "auto":
        class_weights = dm.get_class_weights()
        # Only keep weights for non-unkn classes
        num_classes = cfg.get("num_classes", 6)
        class_weights = class_weights[:num_classes]
        print(f"Class weights: {class_weights.tolist()}")

    # Model
    head_cfg = cfg.get("head", {})
    train_cfg = cfg.get("training", {})
    enc_cfg = cfg.get("encoder", cfg.get("model", {}))

    # Infer encoder config from pretrained checkpoint if available
    num_classes = cfg.get("num_classes", 6)
    max_epochs = args.epochs or train_cfg.get("max_epochs", 50)

    model = PSDClassifier(
        psd_length=psd_length,
        patch_size=enc_cfg.get("patch_size", 16),
        d_model=enc_cfg.get("d_model", 128),
        n_layers=enc_cfg.get("n_layers", 4),
        n_heads=enc_cfg.get("n_heads", 4),
        ffn_dim=enc_cfg.get("ffn_dim", 256),
        dropout=enc_cfg.get("dropout", 0.1),
        num_classes=num_classes,
        head_type=head_cfg.get("type", "mlp"),
        head_hidden=head_cfg.get("hidden_dim", 128),
        head_dropout=head_cfg.get("dropout", 0.2),
        encoder_type=enc_cfg.get("type", "transformer"),
        gradient_checkpointing=enc_cfg.get("gradient_checkpointing", False),
        pretrained_ckpt=pretrained_ckpt,
        freeze_encoder=args.freeze_encoder or model_cfg.get("freeze_encoder", False),
        lr=train_cfg.get("lr", 5e-5),
        encoder_lr_scale=model_cfg.get("encoder_lr_scale", 0.1),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        max_epochs=max_epochs,
        class_weights=class_weights,
    )

    # Compile encoder sub-modules (not the whole LightningModule!)
    if train_cfg.get("compile", False):
        print("Compiling encoder with torch.compile (PyTorch 2.0+)...")
        model.encoder = torch.compile(model.encoder, mode="default")

    # Callbacks
    results_dir = cfg.get("results_dir", "results/")
    log_cfg = cfg.get("logging", {})
    ckpt_dir = Path(results_dir) / "checkpoints" / "finetune"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="classifier-{epoch:02d}-{val_macro_f1:.4f}",
            monitor=log_cfg.get("monitor", "val_macro_f1"),
            mode=log_cfg.get("monitor_mode", "max"),
            save_top_k=log_cfg.get("save_top_k", 3),
            save_last=True,
        ),
        L.pytorch.callbacks.EarlyStopping(
            monitor=log_cfg.get("monitor", "val_macro_f1"),
            mode=log_cfg.get("monitor_mode", "max"),
            patience=train_cfg.get("early_stopping_patience", 10),
        ),
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    # Trainer
    accelerator = args.accelerator or "gpu"
    devices = args.devices or 1

    # WandB logger (if WANDB_API_KEY is set)
    loggers = []
    if os.environ.get("WANDB_API_KEY"):
        from lightning.pytorch.loggers import WandbLogger
        tag = "pretrained" if pretrained_ckpt else "scratch"
        wandb_logger = WandbLogger(
            project="ieee-psd-ssl",
            name=f"finetune-{tag}-frac{fraction}",
            save_dir=results_dir,
            log_model=False,
        )
        loggers.append(wandb_logger)
        print("WandB logging enabled")

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=cfg.get("precision", "bf16-mixed"),
        callbacks=callbacks,
        logger=loggers if loggers else True,
        log_every_n_steps=log_cfg.get("log_every_n_steps", 20),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        default_root_dir=results_dir,
        deterministic=False,
        benchmark=True,
    )

    tag = "pretrained" if pretrained_ckpt else "scratch"
    print("=" * 60)
    print(f"Supervised Fine-tuning ({tag})")
    print(f"  Pretrained: {pretrained_ckpt or 'None (from scratch)'}")
    print(f"  Frozen encoder: {args.freeze_encoder or model_cfg.get('freeze_encoder', False)}")
    print(f"  Data fraction: {fraction}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Checkpoint dir: {ckpt_dir}")
    print("=" * 60)

    trainer.fit(model, datamodule=dm)

    # Save meta
    best_path = trainer.checkpoint_callback.best_model_path
    print(f"\nBest checkpoint: {best_path}")

    meta = {
        "best_ckpt": best_path,
        "pretrained_from": pretrained_ckpt,
        "seed": seed,
        "fraction": fraction,
        "tag": tag,
    }
    with open(Path(results_dir) / f"finetune_meta_{tag}.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
