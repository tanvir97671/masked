#!/bin/bash
# Quick smoke test runner
# Usage:  bash scripts/run_smoke_test.sh [data_dir]

set -e
DATA_DIR="${1:-data/}"
echo "Running smoke test on data in $DATA_DIR ..."
python tests/smoke_test.py --data_dir "$DATA_DIR"
