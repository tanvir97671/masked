"""
Create a minimal synthetic test dataset compatible with ElectroSense format.
This is useful for rapid local testing without downloading 1.7GB.

Usage:
    python src/data/create_minimal_test_data.py --output_dir data/spectrum_bands
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def create_minimal_dataset(output_dir: str, n_sensors: int = 3, samples_per_class: int = 10):
    """
    Create a minimal synthetic dataset in ElectroSense format.

    Args:
        output_dir: Root directory for synthetic data.
        n_sensors: Number of sensor folders.
        samples_per_class: Samples per class per sensor.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ElectroSense classes
    frequencies = {
        "dab": (180, 240),       # MHz
        "dvbt": (420, 780),
        "fm": (80, 130),
        "gsm": (930, 980),
        "lte": (790, 830),
        "tetra": (360, 400),
        "unkn": (0, 3000),
    }

    all_metadata = []

    for sensor_id in range(1, n_sensors + 1):
        sensor_dir = output_path / f"sensor_{sensor_id:03d}"
        sensor_dir.mkdir(exist_ok=True)

        for class_idx, (label, (f_min, f_max)) in enumerate(frequencies.items()):
            for sample_idx in range(samples_per_class):
                # Generate synthetic PSD: random walk with noise
                np.random.seed(sensor_id * 1000 + class_idx * 100 + sample_idx)
                psd = np.cumsum(
                    np.random.randn(256) * 0.1
                ) + np.random.randn(256) * 2
                psd = np.clip(psd, -50, 50)

                # Save as .npy
                file_idx = class_idx * samples_per_class + sample_idx
                npy_path = sensor_dir / f"data_{file_idx:05d}.npy"
                np.save(str(npy_path), psd)

                # Metadata
                snr = np.random.uniform(5, 30)
                all_metadata.append({
                    "file": str(npy_path.relative_to(output_path)),
                    "label": label,
                    "id_sensor": sensor_id,
                    "begin_freq": f_min,
                    "end_freq": f_max,
                    "snr": round(snr, 2),
                })

    # Save manifest CSV
    manifest_path = output_path.parent / "manifest_test.csv"
    with open(str(manifest_path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "id_sensor", "begin_freq", "end_freq", "snr"])
        writer.writeheader()
        writer.writerows(all_metadata)

    print(f"✓ Created minimal test dataset:")
    print(f"  - {n_sensors} sensors × {samples_per_class} samples × {len(frequencies)} classes")
    print(f"  - Total: {len(all_metadata)} samples")
    print(f"  - Location: {output_path}")
    print(f"  - Manifest: {manifest_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create minimal synthetic dataset")
    parser.add_argument("--output_dir", type=str, default="data/spectrum_bands/",
                        help="Output directory")
    parser.add_argument("--n_sensors", type=int, default=3, help="Number of sensors")
    parser.add_argument("--samples_per_class", type=int, default=10,
                        help="Samples per class per sensor")
    args = parser.parse_args()

    create_minimal_dataset(args.output_dir, args.n_sensors, args.samples_per_class)
