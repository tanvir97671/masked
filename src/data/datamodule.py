"""
PyTorch Lightning DataModule for ElectroSense PSD data.

Supports pooled, LOSO, and few-shot LOSO protocols.
Provides DataLoaders for SSL pretraining and supervised classification.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessing import estimate_snr, preprocess_psd


class PSDDataset(Dataset):
    """
    Dataset for PSD samples.

    Each item returns:
        psd: FloatTensor of shape (psd_length,)
        label: int (class index)
        sensor_id: str
        snr: float (estimated SNR in dB)
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        indices: List[int],
        psd_length: int = 200,
        normalize: bool = True,
        clip: bool = True,
    ):
        self.manifest = manifest
        self.indices = indices
        self.psd_length = psd_length
        self.normalize = normalize
        self.clip = clip

        # Cache for loaded npy files (path -> array)
        self._npy_cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def _load_npy(self, path: str) -> np.ndarray:
        if path not in self._npy_cache:
            data = np.load(path, allow_pickle=True)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            self._npy_cache[path] = data
        return self._npy_cache[path]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, float]:
        row = self.manifest.iloc[self.indices[idx]]

        npy_path = row["npy_path"]
        row_index = int(row["row_index"])
        label_idx = int(row["label_idx"])
        sensor_id = str(row["sensor_id"])

        # Load PSD
        psd_array = self._load_npy(npy_path)
        psd = psd_array[row_index]

        # Estimate SNR before preprocessing
        snr = estimate_snr(psd)

        # Preprocess
        psd_tensor = preprocess_psd(
            psd,
            target_length=self.psd_length,
            clip=self.clip,
            normalize=self.normalize,
        )

        return psd_tensor, label_idx, sensor_id, snr


class PSDContrastiveDataset(Dataset):
    """
    Dataset that yields (anchor, positive) pairs for SICR contrastive learning.

    Positive = same frequency band from a different sensor.
    Falls back to same-sensor augmentation if no cross-sensor pair available.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        indices: List[int],
        psd_length: int = 200,
        normalize: bool = True,
    ):
        self.base_dataset = PSDDataset(manifest, indices, psd_length, normalize)
        self.manifest = manifest
        self.indices = indices

        # Build index: label -> {sensor_id -> [local_idx_in_indices]}
        self._label_sensor_map: Dict[int, Dict[str, List[int]]] = {}
        for local_idx, global_idx in enumerate(indices):
            row = manifest.iloc[global_idx]
            label = int(row["label_idx"])
            sensor = str(row["sensor_id"])
            if label not in self._label_sensor_map:
                self._label_sensor_map[label] = {}
            if sensor not in self._label_sensor_map[label]:
                self._label_sensor_map[label][sensor] = []
            self._label_sensor_map[label][sensor].append(local_idx)

        self.rng = np.random.RandomState(42)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Returns:
            anchor_psd, positive_psd, label_idx, is_cross_sensor (1 or 0)
        """
        anchor_psd, label_idx, sensor_id, _ = self.base_dataset[idx]

        # Find positive: same label, different sensor
        label_sensors = self._label_sensor_map.get(label_idx, {})
        other_sensors = [s for s in label_sensors if s != sensor_id]

        if other_sensors:
            pos_sensor = self.rng.choice(other_sensors)
            pos_local_idx = self.rng.choice(label_sensors[pos_sensor])
            pos_psd, _, _, _ = self.base_dataset[pos_local_idx]
            is_cross = 1
        else:
            # Fallback: same sensor, same label (different sample)
            same_list = label_sensors.get(sensor_id, [idx])
            if len(same_list) > 1:
                candidates = [i for i in same_list if i != idx]
                pos_local_idx = self.rng.choice(candidates) if candidates else idx
            else:
                pos_local_idx = idx
            pos_psd, _, _, _ = self.base_dataset[pos_local_idx]
            is_cross = 0

        return anchor_psd, pos_psd, label_idx, is_cross


class ElectroSenseDataModule(L.LightningDataModule):
    """
    Lightning DataModule for ElectroSense PSD data.
    """

    def __init__(
        self,
        manifest_path: str = "data/manifest.csv",
        split_path: str = "data/splits/pooled_seed42_frac1.0.json",
        psd_length: int = 200,
        batch_size: int = 128,
        num_workers: int = 4,
        contrastive: bool = False,
        pin_memory: bool = True,
        persistent_workers: bool = True,  # Keep workers alive
        prefetch_factor: int = 4,          # Prefetch 4 batches per worker
    ):
        super().__init__()
        self.manifest_path = manifest_path
        self.split_path = split_path
        self.psd_length = psd_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.contrastive = contrastive
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

        self.manifest: Optional[pd.DataFrame] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load manifest and create datasets from split file."""
        self.manifest = pd.read_csv(self.manifest_path)

        with open(self.split_path, "r") as f:
            splits = json.load(f)

        train_idx = splits["train"]
        val_idx = splits["val"]
        test_idx = splits["test"]

        if self.contrastive:
            self.train_dataset = PSDContrastiveDataset(
                self.manifest, train_idx, self.psd_length
            )
        else:
            self.train_dataset = PSDDataset(
                self.manifest, train_idx, self.psd_length
            )

        self.val_dataset = PSDDataset(
            self.manifest, val_idx, self.psd_length
        )
        self.test_dataset = PSDDataset(
            self.manifest, test_idx, self.psd_length
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights from training set."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup() first.")

        if isinstance(self.train_dataset, PSDContrastiveDataset):
            indices = self.train_dataset.indices
        else:
            indices = self.train_dataset.indices

        labels = self.manifest.iloc[indices]["label_idx"].values
        unique, counts = np.unique(labels, return_counts=True)
        weights = 1.0 / counts.astype(np.float64)
        weights = weights / weights.sum() * len(unique)
        full_weights = np.ones(7)  # 7 classes including unkn
        for u, w in zip(unique, weights):
            full_weights[int(u)] = w
        return torch.tensor(full_weights, dtype=torch.float32)
