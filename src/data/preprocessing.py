"""
PSD preprocessing utilities.

Handles: robust normalization, clipping, padding/trimming to fixed length,
and SNR estimation from raw PSD vectors.
"""

import numpy as np
import torch


def robust_normalize(
    psd: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Robust per-sample normalization using median and IQR.

    x_hat = (x - median(x)) / (IQR(x) + epsilon)
    """
    median = np.median(psd)
    q75 = np.percentile(psd, 75)
    q25 = np.percentile(psd, 25)
    iqr = q75 - q25
    return (psd - median) / (iqr + epsilon)


def clip_percentile(
    psd: np.ndarray,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> np.ndarray:
    """Clip PSD values to specified percentile range."""
    low = np.percentile(psd, low_pct)
    high = np.percentile(psd, high_pct)
    return np.clip(psd, low, high)


def pad_or_trim(psd: np.ndarray, target_length: int = 200) -> np.ndarray:
    """
    Pad (with edge values) or center-crop PSD to a fixed length.
    """
    current_length = len(psd)
    if current_length == target_length:
        return psd
    elif current_length > target_length:
        # Center-crop
        start = (current_length - target_length) // 2
        return psd[start : start + target_length]
    else:
        # Pad symmetrically with edge values
        pad_total = target_length - current_length
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(psd, (pad_left, pad_right), mode="edge")


def estimate_snr(psd: np.ndarray) -> float:
    """
    Estimate SNR from a PSD vector.

    SNR = 10 * log10(max(psd) / noise_floor)
    where noise_floor = median of the lowest 20% of bins.
    """
    sorted_psd = np.sort(psd)
    n_low = max(1, int(0.2 * len(sorted_psd)))
    noise_floor = np.median(sorted_psd[:n_low])

    if noise_floor <= 0:
        noise_floor = np.abs(noise_floor) + 1e-12

    peak = np.max(psd)
    if peak <= 0:
        return 0.0

    snr_linear = peak / noise_floor
    snr_db = 10.0 * np.log10(max(snr_linear, 1e-12))
    return float(snr_db)


def preprocess_psd(
    psd: np.ndarray,
    target_length: int = 200,
    clip: bool = True,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    normalize: bool = True,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Full preprocessing pipeline for a single PSD vector.

    Steps:
        1. Ensure float64
        2. Replace NaN/Inf
        3. Optional percentile clipping
        4. Pad or trim to fixed length
        5. Optional robust normalization
        6. Convert to torch.FloatTensor

    Args:
        psd: Raw PSD vector (1D numpy array).
        target_length: Fixed output length.
        clip: Whether to apply percentile clipping.
        normalize: Whether to apply robust normalization.

    Returns:
        Preprocessed PSD as torch.FloatTensor of shape (target_length,).
    """
    psd = psd.astype(np.float64)

    # Handle NaN/Inf
    psd = np.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0)

    if clip:
        psd = clip_percentile(psd, clip_low, clip_high)

    psd = pad_or_trim(psd, target_length)

    if normalize:
        psd = robust_normalize(psd, epsilon)

    return torch.tensor(psd, dtype=torch.float32)
