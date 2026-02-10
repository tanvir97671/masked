#!/bin/bash
# =============================================================
# Full Data Pipeline (100%)
# =============================================================

set -e

SEED=42
FRACTION=1.0
CONFIG_PRETRAIN="configs/pretrain_mpae.yaml"
CONFIG_FINETUNE="configs/finetune_classify.yaml"
CONFIG_EVAL="configs/eval_loso.yaml"
DATA_DIR="data/"
RESULTS_DIR="results/"

echo "========================================"
echo "  MPAE+SICR Research Pipeline (FULL data)"
echo "========================================"

# Step 1: Download
python src/data/download.py --data_dir "$DATA_DIR"

# Step 2: Parse
python src/data/parse_dataset.py --data_dir "${DATA_DIR}spectrum_bands" --output "${DATA_DIR}manifest.csv"

# Step 3: Splits
python src/data/splits.py --manifest "${DATA_DIR}manifest.csv" --output_dir "${DATA_DIR}splits/" --seed $SEED --fraction $FRACTION

# Step 4: Smoke test
python tests/smoke_test.py --data_dir "$DATA_DIR"

# Step 5: SSL Pretrain
python src/train_pretrain.py --config "$CONFIG_PRETRAIN" --seed $SEED

# Step 6: Fine-tune
python src/train_finetune.py --config "$CONFIG_FINETUNE" --seed $SEED

# Step 6b: Baselines
python src/train_baseline.py --config "$CONFIG_FINETUNE" --seed $SEED --encoder_type cnn
python src/train_baseline.py --config "$CONFIG_FINETUNE" --seed $SEED --encoder_type transformer

# Step 7: Evaluate
FINETUNE_CKPT=$(python -c "
import json
with open('${RESULTS_DIR}finetune_meta_pretrained.json') as f:
    print(json.load(f)['best_ckpt'])
")

python src/evaluate.py --protocol pooled --ckpt "$FINETUNE_CKPT" --config "$CONFIG_FINETUNE" --tag pretrained
python src/evaluate.py --protocol loso --ckpt "$FINETUNE_CKPT" --config "$CONFIG_FINETUNE" --eval_config "$CONFIG_EVAL"

BASELINE_CKPT=$(python -c "
import json
with open('${RESULTS_DIR}baseline_cnn_meta.json') as f:
    print(json.load(f)['best_ckpt'])
")
python src/evaluate.py --protocol pooled --ckpt "$BASELINE_CKPT" --config "$CONFIG_FINETUNE" --tag baseline_cnn

# Step 8: Calibrate
python src/calibrate.py --ckpt "$FINETUNE_CKPT" --config "$CONFIG_FINETUNE" --method both

# Paper tables
python src/utils/logging_utils.py --results_dir "$RESULTS_DIR"

echo "========================================"
echo "  Full Pipeline Complete!"
echo "========================================"
