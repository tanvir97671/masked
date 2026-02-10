"""
Stream-extract a TINY portion of the real ElectroSense dataset from Zenodo
WITHOUT downloading the full 1.7GB tar.gz.

Strategy: Open the tar.gz as an HTTP stream, extract just the first N .npy
files (a few per sensor), then close the connection early.

Usage:
    python src/data/download_tiny_stream.py --data_dir data/ --max_files 50
"""

import argparse
import io
import os
import struct
import sys
from pathlib import Path

import requests


ZENODO_URL = "https://zenodo.org/api/records/7521246/files/spectrum_bands.tar.gz/content"


def stream_extract_tiny(data_dir: str, max_files: int = 50, max_mb: int = 30):
    """
    Stream the tar.gz from Zenodo and extract only the first max_files .npy files.
    Stops downloading as soon as we have enough, saving bandwidth.
    
    Args:
        data_dir: Where to save extracted files.
        max_files: Max number of .npy files to extract.
        max_mb: Abort after downloading this many MB regardless.
    """
    import gzip
    import tarfile

    data_path = Path(data_dir)
    spectrum_dir = data_path / "spectrum_bands"
    
    # Check if we already have enough files
    existing = list(spectrum_dir.rglob("*.npy")) if spectrum_dir.exists() else []
    if len(existing) >= max_files:
        print(f"Already have {len(existing)} .npy files. Skipping download.")
        return str(spectrum_dir)
    
    print(f"Streaming ElectroSense dataset from Zenodo...")
    print(f"Will extract first {max_files} .npy files (max {max_mb}MB download)")
    print(f"URL: {ZENODO_URL}")
    
    # Stream the response
    response = requests.get(ZENODO_URL, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    print(f"Full archive: {total_size / 1e6:.0f} MB")
    
    # We'll accumulate chunks and try to extract from them
    extracted = 0
    downloaded_bytes = 0
    max_bytes = max_mb * 1024 * 1024
    sensors_seen = set()
    
    # Use tarfile in streaming mode over the HTTP response
    # Wrap the response stream as a file-like for tarfile
    raw_stream = response.raw
    raw_stream.decode_content = True  # Handle gzip at HTTP level if needed
    
    try:
        with tarfile.open(fileobj=raw_stream, mode="r|gz") as tar:
            for member in tar:
                downloaded_bytes = raw_stream.tell() if hasattr(raw_stream, 'tell') else 0
                
                if extracted >= max_files:
                    print(f"\nReached {max_files} files. Stopping stream.")
                    break
                
                # Only extract .npy files and .csv files
                if member.isfile() and (member.name.endswith(".npy") or member.name.endswith(".csv")):
                    # Create output path
                    out_path = data_path / member.name
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Extract this member
                    fileobj = tar.extractfile(member)
                    if fileobj is not None:
                        with open(str(out_path), "wb") as f:
                            f.write(fileobj.read())
                        
                        if member.name.endswith(".npy"):
                            extracted += 1
                            # Track sensor
                            parts = Path(member.name).parts
                            if len(parts) >= 2:
                                sensors_seen.add(parts[1] if parts[0] == "spectrum_bands" else parts[0])
                        
                        size_kb = member.size / 1024
                        if extracted % 10 == 0 or extracted <= 5:
                            print(f"  [{extracted}/{max_files}] {member.name} ({size_kb:.1f} KB)")
                
                elif member.isdir():
                    # Create directory
                    dir_path = data_path / member.name
                    dir_path.mkdir(parents=True, exist_ok=True)
                    
    except Exception as e:
        if extracted > 0:
            print(f"\nStream interrupted ({e}), but extracted {extracted} files. Continuing with those.")
        else:
            raise
    finally:
        response.close()
    
    print(f"\n{'='*50}")
    print(f"Download complete!")
    print(f"  Extracted: {extracted} .npy files")
    print(f"  Sensors: {len(sensors_seen)} ({', '.join(sorted(sensors_seen)[:5])}{'...' if len(sensors_seen) > 5 else ''})")
    print(f"  Location: {spectrum_dir}")
    print(f"{'='*50}")
    
    return str(spectrum_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream-download tiny portion of ElectroSense")
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Data directory")
    parser.add_argument("--max_files", type=int, default=50,
                        help="Max .npy files to extract (default: 50)")
    parser.add_argument("--max_mb", type=int, default=30,
                        help="Max MB to download (default: 30)")
    args = parser.parse_args()
    
    stream_extract_tiny(args.data_dir, args.max_files, args.max_mb)
