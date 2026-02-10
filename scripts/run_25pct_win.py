"""
Windows-compatible 25% pipeline runner.

Usage:
    python scripts/run_25pct_win.py
"""

import json
import subprocess
import sys
from pathlib import Path

SEED = 42
FRACTION = 0.25
DATA_DIR = "data/"
RESULTS_DIR = "results/"
PYTHON = sys.executable


def run(cmd, desc=""):
    """Run a command and check for errors."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  CMD: {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: Command failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    print("MPAE+SICR Research Pipeline (25% data, Windows)")
    print("=" * 60)

    # 1. Download
    run(f"{PYTHON} src/data/download.py --data_dir {DATA_DIR}",
        "Step 1: Download dataset")

    # 2. Parse
    run(f"{PYTHON} src/data/parse_dataset.py --data_dir {DATA_DIR}spectrum_bands --output {DATA_DIR}manifest.csv",
        "Step 2: Parse dataset")

    # 3. Splits
    run(f"{PYTHON} src/data/splits.py --manifest {DATA_DIR}manifest.csv --output_dir {DATA_DIR}splits/ --seed {SEED} --fraction {FRACTION}",
        "Step 3: Generate splits")

    # 4. Smoke test
    run(f"{PYTHON} tests/smoke_test.py --data_dir {DATA_DIR}",
        "Step 4: Smoke test")

    # 5. SSL Pretrain
    run(f"{PYTHON} src/train_pretrain.py --config configs/pretrain_mpae.yaml --data.fraction {FRACTION} --seed {SEED}",
        "Step 5: SSL Pretraining (MPAE + SICR)")

    # 6. Fine-tune
    run(f"{PYTHON} src/train_finetune.py --config configs/finetune_classify.yaml --data.fraction {FRACTION} --seed {SEED}",
        "Step 6: Supervised Fine-tuning")

    # 6b. Baseline CNN
    run(f"{PYTHON} src/train_baseline.py --config configs/finetune_classify.yaml --data.fraction {FRACTION} --seed {SEED} --encoder_type cnn",
        "Step 6b: Baseline CNN")

    # 7. Evaluate
    meta_path = Path(RESULTS_DIR) / "finetune_meta_pretrained.json"
    if meta_path.exists():
        with open(meta_path) as f:
            ckpt = json.load(f)["best_ckpt"]

        run(f'{PYTHON} src/evaluate.py --protocol pooled --ckpt "{ckpt}" --config configs/finetune_classify.yaml --tag pretrained --data.fraction {FRACTION}',
            "Step 7a: Pooled evaluation (pretrained)")

        run(f'{PYTHON} src/evaluate.py --protocol loso --ckpt "{ckpt}" --config configs/finetune_classify.yaml --eval_config configs/eval_loso.yaml --num_sensors 5 --data.fraction {FRACTION}',
            "Step 7b: LOSO evaluation")

    # Baseline eval
    baseline_meta = Path(RESULTS_DIR) / "baseline_cnn_meta.json"
    if baseline_meta.exists():
        with open(baseline_meta) as f:
            bl_ckpt = json.load(f)["best_ckpt"]
        run(f'{PYTHON} src/evaluate.py --protocol pooled --ckpt "{bl_ckpt}" --config configs/finetune_classify.yaml --tag baseline_cnn --data.fraction {FRACTION}',
            "Step 7c: Baseline evaluation")

    # 8. Calibrate
    if meta_path.exists():
        with open(meta_path) as f:
            ckpt = json.load(f)["best_ckpt"]
        run(f'{PYTHON} src/calibrate.py --ckpt "{ckpt}" --config configs/finetune_classify.yaml --method both --data.fraction {FRACTION}',
            "Step 8: Calibration")

    # Paper tables
    run(f"{PYTHON} src/utils/logging_utils.py --results_dir {RESULTS_DIR}",
        "Generating paper tables")

    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print(f"  Results in: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
