"""
Generate deterministic train/val/test splits for different evaluation protocols.

Protocols:
  - pooled: standard stratified random split across all sensors
  - loso: leave-one-sensor-out
  - fewshot_loso: LOSO + K-shot fine-tuning on target sensor
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def generate_pooled_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    fraction: float = 1.0,
) -> Dict[str, List[int]]:
    """
    Standard stratified split across all sensors combined.

    Returns dict with 'train', 'val', 'test' index lists.
    """
    rng = np.random.RandomState(seed)

    # Optionally subsample (fraction)
    if fraction < 1.0:
        n_keep = max(1, int(len(df) * fraction))
        indices = rng.choice(len(df), size=n_keep, replace=False)
        indices = sorted(indices)
    else:
        indices = list(range(len(df)))

    labels = df.iloc[indices]["label_idx"].values

    # First split: train+val vs test
    test_ratio = 1.0 - train_ratio - val_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(indices, labels))

    # Second split: train vs val
    trainval_labels = labels[trainval_idx]
    val_relative = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_relative, random_state=seed)
    train_sub_idx, val_sub_idx = next(sss2.split(trainval_idx, trainval_labels))

    train_indices = [int(indices[trainval_idx[i]]) for i in train_sub_idx]
    val_indices = [int(indices[trainval_idx[i]]) for i in val_sub_idx]
    test_indices = [int(indices[i]) for i in test_idx]

    return {"train": train_indices, "val": val_indices, "test": test_indices}


def generate_loso_splits(
    df: pd.DataFrame,
    num_sensors: Optional[int] = None,
    val_ratio: float = 0.15,
    seed: int = 42,
    fraction: float = 1.0,
) -> Dict[str, Dict[str, List[int]]]:
    """
    Generate Leave-One-Sensor-Out splits.

    For each held-out sensor:
      - test: all samples from that sensor
      - train + val: stratified split of remaining sensors

    Args:
        num_sensors: If set, only generate splits for this many sensors
                     (randomly selected for speed).

    Returns:
        Dict mapping sensor_id -> {'train': [...], 'val': [...], 'test': [...]}.
    """
    rng = np.random.RandomState(seed)

    all_sensors = sorted(df["sensor_id"].unique())

    if num_sensors is not None and num_sensors < len(all_sensors):
        selected = rng.choice(all_sensors, size=num_sensors, replace=False)
        selected = sorted(selected)
    else:
        selected = all_sensors

    splits = {}

    for sensor_id in selected:
        test_mask = df["sensor_id"] == sensor_id
        train_mask = ~test_mask

        test_indices = df.index[test_mask].tolist()
        train_df = df[train_mask]

        # Subsample if fraction < 1
        if fraction < 1.0:
            n_keep = max(1, int(len(train_df) * fraction))
            keep_idx = rng.choice(len(train_df), size=n_keep, replace=False)
            train_df = train_df.iloc[keep_idx]

        # Split train into train/val
        if len(train_df) < 10:
            train_indices = train_df.index.tolist()
            val_indices = []
        else:
            labels = train_df["label_idx"].values
            try:
                sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=val_ratio, random_state=seed
                )
                t_idx, v_idx = next(sss.split(train_df.index, labels))
                train_indices = train_df.index[t_idx].tolist()
                val_indices = train_df.index[v_idx].tolist()
            except ValueError:
                # Not enough samples for stratified split
                n_val = max(1, int(len(train_df) * val_ratio))
                all_idx = train_df.index.tolist()
                rng.shuffle(all_idx)
                val_indices = all_idx[:n_val]
                train_indices = all_idx[n_val:]

        splits[str(sensor_id)] = {
            "train": [int(i) for i in train_indices],
            "val": [int(i) for i in val_indices],
            "test": [int(i) for i in test_indices],
        }

    return splits


def generate_fewshot_loso_splits(
    df: pd.DataFrame,
    k_values: List[int] = [5, 10, 50],
    num_sensors: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Dict[int, Dict[str, List[int]]]]:
    """
    Generate Few-Shot LOSO splits.

    For each held-out sensor and each K:
      - finetune: K random labeled samples per class from target sensor
      - test: remaining samples from target sensor

    Returns:
        Dict mapping sensor_id -> {K -> {'finetune': [...], 'test': [...]}}.
    """
    rng = np.random.RandomState(seed)

    all_sensors = sorted(df["sensor_id"].unique())

    if num_sensors is not None and num_sensors < len(all_sensors):
        selected = rng.choice(all_sensors, size=num_sensors, replace=False)
        selected = sorted(selected)
    else:
        selected = all_sensors

    splits = {}

    for sensor_id in selected:
        sensor_df = df[df["sensor_id"] == sensor_id]
        sensor_splits = {}

        for k in k_values:
            finetune_indices = []
            test_indices = []

            for label_idx in sorted(sensor_df["label_idx"].unique()):
                class_df = sensor_df[sensor_df["label_idx"] == label_idx]
                class_indices = class_df.index.tolist()

                if len(class_indices) <= k:
                    # Not enough samples; use all for finetune, none for test
                    finetune_indices.extend(class_indices)
                else:
                    rng.shuffle(class_indices)
                    finetune_indices.extend(class_indices[:k])
                    test_indices.extend(class_indices[k:])

            sensor_splits[k] = {
                "finetune": [int(i) for i in finetune_indices],
                "test": [int(i) for i in test_indices],
            }

        splits[str(sensor_id)] = sensor_splits

    return splits


def generate_splits(
    manifest_path: str = "data/manifest.csv",
    output_dir: str = "data/splits/",
    seed: int = 42,
    fraction: float = 1.0,
    num_loso_sensors: Optional[int] = None,
    fewshot_k_values: List[int] = [5, 10, 50, 100],
    exclude_unknown: bool = True,
) -> None:
    """
    Generate and save all split files.
    """
    df = pd.read_csv(manifest_path)

    if exclude_unknown:
        df = df[df["label"] != "unkn"].reset_index(drop=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Pooled split
    pooled = generate_pooled_split(df, seed=seed, fraction=fraction)
    with open(output_dir / f"pooled_seed{seed}_frac{fraction}.json", "w") as f:
        json.dump(pooled, f)
    print(f"Pooled split: train={len(pooled['train'])}, val={len(pooled['val'])}, test={len(pooled['test'])}")

    # 2) LOSO splits
    loso = generate_loso_splits(
        df, num_sensors=num_loso_sensors, seed=seed, fraction=fraction
    )
    with open(output_dir / f"loso_seed{seed}_frac{fraction}.json", "w") as f:
        json.dump(loso, f)
    print(f"LOSO splits: {len(loso)} held-out sensors")

    # 3) Few-shot LOSO splits
    fewshot = generate_fewshot_loso_splits(
        df, k_values=fewshot_k_values, num_sensors=num_loso_sensors, seed=seed
    )
    with open(output_dir / f"fewshot_loso_seed{seed}.json", "w") as f:
        json.dump(fewshot, f, default=int)
    print(f"Few-shot LOSO splits: {len(fewshot)} sensors × {len(fewshot_k_values)} K values")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data splits")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--output_dir", type=str, default="data/splits/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=1.0, help="Data fraction (0.25 for quick run)")
    parser.add_argument("--num_loso_sensors", type=int, default=None)
    args = parser.parse_args()
    generate_splits(
        args.manifest,
        args.output_dir,
        args.seed,
        args.fraction,
        args.num_loso_sensors,
    )
