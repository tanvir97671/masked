"""
Download and extract the ElectroSense PSD Spectrum Dataset from Zenodo.
Verifies MD5 checksum after download.
"""

import argparse
import hashlib
import os
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_URL = "https://zenodo.org/api/records/7521246/files/spectrum_bands.tar.gz/content"
EXPECTED_MD5 = "285f25b558959a773f335ed038e1b053"
FILENAME = "spectrum_bands.tar.gz"


def compute_md5(filepath: str, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def download_file(url: str, dest: str) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="Downloading"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_tarball(tar_path: str, extract_dir: str) -> None:
    """Extract a .tar.gz archive."""
    print(f"Extracting {tar_path} to {extract_dir} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    print("Extraction complete.")


def download_dataset(data_dir: str = "data/", skip_if_exists: bool = True) -> str:
    """
    Download and extract the ElectroSense PSD dataset.

    Args:
        data_dir: Root data directory.
        skip_if_exists: Skip download if extracted folder already exists.

    Returns:
        Path to the extracted dataset folder.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    tar_path = data_dir / FILENAME
    extract_dir = data_dir / "spectrum_bands"

    # Skip if already extracted
    if skip_if_exists and extract_dir.exists():
        n_dirs = len([d for d in extract_dir.iterdir() if d.is_dir()])
        if n_dirs > 0:
            print(f"Dataset already extracted at {extract_dir} ({n_dirs} sensor folders). Skipping.")
            return str(extract_dir)

    # Download
    if not tar_path.exists():
        print(f"Downloading ElectroSense PSD dataset from Zenodo...")
        print(f"URL: {ZENODO_URL}")
        download_file(ZENODO_URL, str(tar_path))
    else:
        print(f"Tarball already exists at {tar_path}. Skipping download.")

    # Verify MD5
    print("Verifying MD5 checksum...")
    actual_md5 = compute_md5(str(tar_path))
    if actual_md5 != EXPECTED_MD5:
        print(f"ERROR: MD5 mismatch! Expected {EXPECTED_MD5}, got {actual_md5}")
        print("File is corrupted. Re-downloading...")
        tar_path.unlink()  # Delete corrupted file
        print(f"Downloading ElectroSense PSD dataset from Zenodo...")
        download_file(ZENODO_URL, str(tar_path))
        actual_md5 = compute_md5(str(tar_path))
        if actual_md5 != EXPECTED_MD5:
            print(f"ERROR: MD5 still mismatches after re-download!")
            sys.exit(1)
    else:
        print("MD5 checksum verified OK.")

    # Extract
    extract_tarball(str(tar_path), str(data_dir))

    return str(extract_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ElectroSense PSD dataset")
    parser.add_argument("--data_dir", type=str, default="data/", help="Data directory")
    args = parser.parse_args()
    path = download_dataset(args.data_dir)
    print(f"Dataset ready at: {path}")
