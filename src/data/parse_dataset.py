"""
Parse the extracted ElectroSense PSD dataset into a unified manifest CSV.

Walks sensor folders, loads .npy PSD arrays + .csv metadata,
assigns technology labels based on licensed-band frequency mappings.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Licensed-band frequency boundaries (MHz) for technology labeling
# Same mapping as the original ElectroSense framework
BAND_MAPPING: List[Tuple[float, float, str]] = [
    (80.0, 130.0, "fm"),
    (180.0, 240.0, "dab"),
    (360.0, 400.0, "tetra"),
    (420.0, 780.0, "dvbt"),
    (790.0, 830.0, "lte"),
    (930.0, 980.0, "gsm"),
]

LABEL_TO_IDX: Dict[str, int] = {
    "dab": 0,
    "dvbt": 1,
    "fm": 2,
    "gsm": 3,
    "lte": 4,
    "tetra": 5,
    "unkn": 6,
}

IDX_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_IDX.items()}


def freq_to_label(center_freq_mhz: float) -> str:
    """Map a center frequency (MHz) to a technology label."""
    for low, high, label in BAND_MAPPING:
        if low <= center_freq_mhz <= high:
            return label
    return "unkn"


def _infer_freq_from_filename(npy_path: str) -> Optional[float]:
    """
    Try to extract center frequency from ElectroSense filename.
    Pattern: SpectrumBands_<begin>_<end>_<label>_...
    Frequencies in filename are in MHz.
    """
    import re
    fname = Path(npy_path).stem
    match = re.match(r"SpectrumBands_(\d+)_(\d+)_", fname)
    if match:
        begin = float(match.group(1))
        end = float(match.group(2))
        return (begin + end) / 2.0
    return None


def _infer_label_from_filename(npy_path: str) -> str:
    """
    Try to extract technology label directly from filename.
    Pattern: SpectrumBands_80_110_fm_... → 'fm'
    """
    import re
    fname = Path(npy_path).stem.lower()
    # Look for known labels in filename
    for label in ["fm", "dab", "tetra", "dvbt", "lte", "gsm"]:
        # Match as a word boundary to avoid false positives
        if re.search(rf"[_\-]{label}[_\-]", f"_{fname}_"):
            return label
    return "unkn"


def parse_single_npy(
    npy_path: str,
    csv_path: Optional[str],
    sensor_id: str,
) -> List[Dict]:
    """
    Parse one .npy file (+ optional .csv sidecar) into sample records.

    Each .npy may contain a 2D array: (num_time_segments, num_freq_bins).
    The .csv may contain metadata with begin_freq, end_freq, SNR, etc.
    """
    records = []

    try:
        psd_data = np.load(npy_path, allow_pickle=True)
    except Exception as e:
        print(f"  WARNING: Could not load {npy_path}: {e}")
        return records

    # Handle various shapes
    if psd_data.ndim == 1:
        psd_data = psd_data.reshape(1, -1)
    elif psd_data.ndim == 0:
        # Scalar or empty
        return records

    num_segments, num_bins = psd_data.shape

    # Try to read metadata from CSV sidecar
    meta = {}
    if csv_path and os.path.exists(csv_path):
        try:
            df_meta = pd.read_csv(csv_path)
            if len(df_meta) > 0:
                row = df_meta.iloc[0]
                for col in ["begin_freq", "end_freq", "SNR", "snr", "id_sensor"]:
                    if col in row:
                        meta[col.lower()] = row[col]
        except Exception:
            pass

    # Determine center frequency for labeling
    begin_freq = meta.get("begin_freq", None)
    end_freq = meta.get("end_freq", None)
    snr_val = meta.get("snr", None)

    if begin_freq is not None and end_freq is not None:
        center_freq_mhz = (float(begin_freq) + float(end_freq)) / 2.0
        # Convert from Hz to MHz if needed
        if center_freq_mhz > 1e6:
            center_freq_mhz /= 1e6
        elif center_freq_mhz > 1e3:
            center_freq_mhz /= 1e3
    else:
        # Try to infer from filename
        # ElectroSense filenames: SpectrumBands_80_110_fm_Swis_80_110.npy
        center_freq_mhz = _infer_freq_from_filename(npy_path)

    label = freq_to_label(center_freq_mhz) if center_freq_mhz is not None else "unkn"
    
    # Also try to extract label directly from filename as a fallback
    if label == "unkn":
        label_from_fn = _infer_label_from_filename(npy_path)
        if label_from_fn != "unkn":
            label = label_from_fn
    
    label_idx = LABEL_TO_IDX[label]

    for seg_idx in range(num_segments):
        record = {
            "npy_path": npy_path,
            "row_index": seg_idx,
            "sensor_id": sensor_id,
            "num_bins": num_bins,
            "center_freq_mhz": center_freq_mhz,
            "begin_freq": begin_freq,
            "end_freq": end_freq,
            "snr": snr_val,
            "label": label,
            "label_idx": label_idx,
        }
        records.append(record)

    return records


def parse_electrosense(
    data_dir: str = "data/spectrum_bands",
    output_path: str = "data/manifest.csv",
) -> pd.DataFrame:
    """
    Walk the extracted ElectroSense dataset, parse all .npy files,
    and produce a manifest CSV.

    Args:
        data_dir: Path to extracted spectrum_bands/ folder.
        output_path: Where to save the manifest CSV.

    Returns:
        DataFrame with all sample records.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    all_records = []

    # Find all .npy files first, then group by sensor
    all_npy = sorted(data_path.rglob("*.npy"))
    if not all_npy:
        print(f"WARNING: No .npy files found under {data_path}")
        return pd.DataFrame()

    print(f"Found {len(all_npy)} .npy files total.")

    # Infer sensor from directory structure
    # Real ElectroSense path: .../spectrum_bands_2/SensorName/MonthDay/file.npy
    # The sensor is typically the grandparent of the .npy file
    sensor_to_files = {}
    for npy_file in all_npy:
        # Walk up to find a reasonable sensor name:
        # If path contains 'spectrum_bands_2', sensor is the dir right after it
        parts = npy_file.parts
        sensor_id = None
        for i, part in enumerate(parts):
            if "spectrum_bands" in part.lower() and i + 1 < len(parts):
                sensor_id = parts[i + 1]
                break
        if sensor_id is None:
            # Fallback: use grandparent directory name
            sensor_id = npy_file.parent.parent.name if len(npy_file.parts) > 2 else npy_file.parent.name
        
        if sensor_id not in sensor_to_files:
            sensor_to_files[sensor_id] = []
        sensor_to_files[sensor_id].append(npy_file)

    print(f"Found {len(sensor_to_files)} sensors: {sorted(sensor_to_files.keys())}")

    for sensor_id in tqdm(sorted(sensor_to_files.keys()), desc="Parsing sensors"):
        npy_files = sensor_to_files[sensor_id]

        for npy_file in npy_files:
            # Look for matching .csv sidecar
            csv_path = npy_file.with_suffix(".csv")
            if not csv_path.exists():
                csv_path = None
            else:
                csv_path = str(csv_path)

            records = parse_single_npy(str(npy_file), csv_path, sensor_id)
            all_records.extend(records)

    df = pd.DataFrame(all_records)

    if len(df) == 0:
        print("WARNING: No samples found! Check dataset extraction.")
        return df

    # Summary stats
    print(f"\n=== Dataset Manifest ===")
    print(f"Total samples: {len(df):,}")
    print(f"Sensors: {df['sensor_id'].nunique()}")
    print(f"Class distribution:")
    print(df["label"].value_counts().to_string())
    print(f"Bins per PSD (unique values): {sorted(df['num_bins'].unique())}")

    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nManifest saved to {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse ElectroSense PSD dataset")
    parser.add_argument("--data_dir", type=str, default="data/spectrum_bands")
    parser.add_argument("--output", type=str, default="data/manifest.csv")
    args = parser.parse_args()
    parse_electrosense(args.data_dir, args.output)
