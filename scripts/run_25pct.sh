#!/bin/bash
# =============================================================
# 25% Data Pipeline: Pretrain → Finetune → Baseline → Eval → Calibrate
# Designed for Lightning AI L40s GPU.
# =============================================================

set -e  # Exit on error

SEED=42
FRACTION=0.25
CONFIG_PRETRAIN="configs/pretrain_mpae.yaml"
CONFIG_FINETUNE="configs/finetune_classify.yaml"
CONFIG_EVAL="configs/eval_loso.yaml"
DATA_DIR="data/"
RESULTS_DIR="results/"

echo "========================================"
echo "  MPAE+SICR Research Pipeline (25% data)"
echo "========================================"

# Step 1: Download dataset (skip if exists)
echo ""
echo "[Step 1/8] Downloading dataset..."
python src/data/download.py --data_dir "$DATA_DIR"

# Step 2: Parse dataset into manifest
echo ""
echo "[Step 2/8] Parsing dataset..."
python src/data/parse_dataset.py --data_dir "${DATA_DIR}spectrum_bands" --output "${DATA_DIR}manifest.csv"

# Step 3: Generate splits (25% fraction)
echo ""
echo "[Step 3/8] Generating splits (fraction=${FRACTION})..."
python src/data/splits.py --manifest "${DATA_DIR}manifest.csv" --output_dir "${DATA_DIR}splits/" --seed $SEED --fraction $FRACTION

# Step 4: Run smoke test
echo ""
echo "[Step 4/8] Running smoke test..."
python tests/smoke_test.py --data_dir "$DATA_DIR"

# Step 5: SSL Pretraining (MPAE + SICR)
echo ""
echo "[Step 5/8] SSL Pretraining (MPAE + SICR)..."
python src/train_pretrain.py --config "$CONFIG_PRETRAIN" --data.fraction $FRACTION --seed $SEED

# Step 6: Supervised Fine-tuning (with pretrained encoder)
echo ""
echo "[Step 6/8] Supervised Fine-tuning (pretrained)..."
python src/train_finetune.py --config "$CONFIG_FINETUNE" --data.fraction $FRACTION --seed $SEED

# Step 6b: Baseline (CNN from scratch)
echo ""
echo "[Step 6b/8] Baseline CNN training..."
python src/train_baseline.py --config "$CONFIG_FINETUNE" --data.fraction $FRACTION --seed $SEED --encoder_type cnn

# Step 6c: Baseline (Transformer from scratch — ablation)
echo ""
echo "[Step 6c/8] Baseline Transformer training..."
python src/train_baseline.py --config "$CONFIG_FINETUNE" --data.fraction $FRACTION --seed $SEED --encoder_type transformer

# Step 7: Evaluate
echo ""
echo "[Step 7/8] Evaluating..."

# Find best finetune checkpoint
FINETUNE_CKPT=$(python -c "
import json
with open('${RESULTS_DIR}finetune_meta_pretrained.json') as f:
    print(json.load(f)['best_ckpt'])
")

echo "  Using checkpoint: $FINETUNE_CKPT"

# Pooled evaluation
python src/evaluate.py --protocol pooled --ckpt "$FINETUNE_CKPT" --config "$CONFIG_FINETUNE" --tag pretrained --data.fraction $FRACTION

# LOSO evaluation (5 sensors for speed)
python src/evaluate.py --protocol loso --ckpt "$FINETUNE_CKPT" --config "$CONFIG_FINETUNE" --eval_config "$CONFIG_EVAL" --num_sensors 5 --data.fraction $FRACTION

# Evaluate baseline too
BASELINE_CKPT=$(python -c "
import json
with open('${RESULTS_DIR}baseline_cnn_meta.json') as f:
    print(json.load(f)['best_ckpt'])
")
python src/evaluate.py --protocol pooled --ckpt "$BASELINE_CKPT" --config "$CONFIG_FINETUNE" --tag baseline_cnn --data.fraction $FRACTION

# Step 8: Calibrate
echo ""
echo "[Step 8/8] Calibrating..."
python src/calibrate.py --ckpt "$FINETUNE_CKPT" --config "$CONFIG_FINETUNE" --method both --data.fraction $FRACTION

# Generate paper tables
echo ""
echo "Generating paper-ready tables..."
python src/utils/logging_utils.py --results_dir "$RESULTS_DIR" --output_dir "${RESULTS_DIR}paper_tables/"

echo ""
echo "========================================"
echo "  Pipeline Complete!"
echo "  Results saved to: $RESULTS_DIR"
echo "========================================"
