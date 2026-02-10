#!/bin/bash
# Download ElectroSense PSD dataset from Zenodo.
# Usage:  bash scripts/download_data.sh [data_dir]

DATA_DIR="${1:-data/}"
echo "Downloading ElectroSense dataset to $DATA_DIR ..."
python src/data/download.py --data_dir "$DATA_DIR"
echo "Done."
