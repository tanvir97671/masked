"""
SNR estimation utilities for batched PSD data.
"""

import numpy as np
import torch


def estimate_snr_single(psd: np.ndarray) -> float:
    """
    Estimate SNR from a single PSD vector (numpy).

    SNR = 10 * log10(peak / noise_floor)
    noise_floor = median of lowest 20% of bins.
    """
    sorted_psd = np.sort(psd)
    n_low = max(1, int(0.2 * len(sorted_psd)))
    noise_floor = np.median(sorted_psd[:n_low])

    if noise_floor <= 0:
        noise_floor = np.abs(noise_floor) + 1e-12

    peak = np.max(psd)
    if peak <= 0:
        return 0.0

    snr_db = 10.0 * np.log10(max(peak / noise_floor, 1e-12))
    return float(snr_db)


def estimate_snr_batch(psd_batch: torch.Tensor) -> torch.Tensor:
    """
    Estimate SNR for a batch of PSD vectors.

    Args:
        psd_batch: (batch, psd_length) tensor

    Returns:
        snr: (batch,) tensor of estimated SNR in dB
    """
    batch_size = psd_batch.size(0)
    psd_length = psd_batch.size(1)
    n_low = max(1, int(0.2 * psd_length))

    # Sort along frequency axis
    sorted_psd, _ = torch.sort(psd_batch, dim=1)

    # Noise floor: median of lowest 20%
    noise_floor = sorted_psd[:, :n_low].median(dim=1).values
    noise_floor = noise_floor.abs().clamp(min=1e-12)

    # Peak
    peak = psd_batch.max(dim=1).values.clamp(min=1e-12)

    # SNR in dB
    snr = 10.0 * torch.log10((peak / noise_floor).clamp(min=1e-12))

    return snr
