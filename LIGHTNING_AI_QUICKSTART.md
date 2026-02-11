# Lightning AI Studio Quick Start (L40S Optimized)

## One-Command Full Pipeline
```bash
git clone https://github.com/tanvir97671/masked.git && cd masked && \
pip install -r requirements.txt && \
python src/data/download_tiny_stream.py && \
python src/data/parse_dataset.py && \
python src/data/splits.py --manifest data/manifest.csv --output data/splits/ --fraction 0.25 --seed 42 && \
python src/train_pretrain.py --config configs/pretrain_mpae.yaml --manifest data/manifest.csv --split data/splits/pooled_seed42_frac0.25.json --epochs 20 --accelerator gpu --devices 1 && \
python src/train_finetune.py --config configs/finetune_classify.yaml --manifest data/manifest.csv --split data/splits/pooled_seed42_frac0.25.json --pretrained results/lightning_logs/version_0/checkpoints/epoch=*.ckpt --epochs 50 --accelerator gpu --devices 1
```

## Step-by-Step (25% Dataset)

### 1. Setup (5 minutes)
```bash
cd masked  # Already cloned
pip install -r requirements.txt
```

### 2. Download & Parse Data (~2 minutes, 30MB)
```bash
python src/data/download_tiny_stream.py
python src/data/parse_dataset.py
```
Expected: `Found 50 .npy files, 11 sensors, 21,351 total samples`

### 3. Generate Splits (~5 seconds)
```bash
python src/data/splits.py \
  --manifest data/manifest.csv \
  --output data/splits/ \
  --fraction 0.25 \
  --seed 42
```
Expected: `train=3735, val=801, test=801`

### 4. Pretrain MPAE + SICR (~30-40 min, 20 epochs)
```bash
python src/train_pretrain.py \
  --config configs/pretrain_mpae.yaml \
  --manifest data/manifest.csv \
  --split data/splits/pooled_seed42_frac0.25.json \
  --epochs 20 \
  --accelerator gpu \
  --devices 1
```
**Config highlights:**
- Model: 512-d, 8 layers, 14.2M params
- Batch size: 512 (contrastive pairs)
- Precision: bf16-mixed
- Compiled: Yes
- GPU usage: ~40GB VRAM

**Monitor:** `watch -n 1 nvidia-smi`

**Expected training time:** 30-40 minutes (235 batches × 20 epochs)

---

### 5. Finetune Classifier (~60-80 min, 50 epochs)
```bash
# Find pretrained checkpoint
ls results/lightning_logs/version_0/checkpoints/

# Finetune
python src/train_finetune.py \
  --config configs/finetune_classify.yaml \
  --manifest data/manifest.csv \
  --split data/splits/pooled_seed42_frac0.25.json \
  --pretrained results/lightning_logs/version_0/checkpoints/epoch=19-step=4700.ckpt \
  --epochs 50 \
  --accelerator gpu \
  --devices 1
```
**Config highlights:**
- Model: 512-d encoder + 256-d MLP head
- Batch size: 512
- Early stopping: patience=10
- GPU usage: ~38GB VRAM

**Expected training time:** 60-80 minutes (235 batches × 50 epochs, stops early)

---

### 6. Evaluate (~30 seconds)
```bash
python src/evaluate.py \
  --manifest data/manifest.csv \
  --split data/splits/pooled_seed42_frac0.25.json \
  --checkpoint results/lightning_logs/version_1/checkpoints/epoch=*.ckpt
```

---

## Cost Breakdown (25% Dataset)

| Stage | Time | Cost @ $1.7/hr | GPU Util |
|-------|------|----------------|----------|
| Setup + Data | 7 min | $0.20 | 0-5% |
| Pretrain (20 ep) | 35 min | $1.00 | 85% |
| Finetune (50 ep) | 70 min | $2.00 | 80% |
| Evaluate | 1 min | $0.03 | 60% |
| **Total** | **~2 hours** | **~$3.40** | - |

**Old (unoptimized):** ~10 hours, ~$17  
**Savings:** ~70% cost reduction

---

## Full Dataset (100%, Optional)

```bash
# Skip download_tiny_stream, use full download
python src/data/download.py  # ~1.7GB, 20 min
python src/data/parse_dataset.py  # ~5 min
python src/data/splits.py --manifest data/manifest.csv --output data/splits/ --fraction 1.0 --seed 42

# Pretrain (80 epochs)
python src/train_pretrain.py \
  --config configs/pretrain_mpae.yaml \
  --manifest data/manifest.csv \
  --split data/splits/pooled_seed42_frac1.0.json \
  --epochs 80 \
  --accelerator gpu \
  --devices 1

# Finetune (100 epochs)
python src/train_finetune.py \
  --config configs/finetune_classify.yaml \
  --manifest data/manifest.csv \
  --split data/splits/pooled_seed42_frac1.0.json \
  --pretrained results/lightning_logs/version_0/checkpoints/epoch=*.ckpt \
  --epochs 100 \
  --accelerator gpu \
  --devices 1
```

**Estimated time:** 6-8 hours  
**Estimated cost:** $10-14

---

## Monitoring Tips

### GPU Usage
```bash
watch -n 1 nvidia-smi
```
Look for:
- GPU Util: 80-95%
- Memory: 38-42GB / 48GB
- Power: 260-290W

### Training Logs
```bash
# Watch training logs
tail -f results/lightning_logs/version_*/events.out.tfevents.*

# Or use TensorBoard
tensorboard --logdir results/lightning_logs/
```

### Checkpoints
```bash
ls -lh results/lightning_logs/version_*/checkpoints/
```
Best checkpoint saved automatically based on validation loss/F1.

---

## Troubleshooting

### OOM (Out of Memory)
```yaml
# configs/pretrain_mpae.yaml
training:
  batch_size: 256  # Reduce from 512
  accumulate_grad_batches: 2  # Effective = 512
```

### Slow Data Loading
```bash
# Check if workers are alive
ps aux | grep python
# Should see 8+ DataLoader worker processes
```

### Torch Compile Errors
```yaml
# configs/*.yaml
training:
  compile: false  # Disable compilation
```

---

## Expected Results (25% Dataset)

After full training:
- **Test Accuracy:** ~65-75%
- **Test Macro F1:** ~0.60-0.70
- **Best Class (FM):** ~85% precision
- **Worst Class (tetra):** ~50% recall

Full dataset should reach ~80-85% accuracy.

---

## Next Steps

1. **LOSO Evaluation:** Cross-sensor generalization
   ```bash
   python src/evaluate.py \
     --manifest data/manifest.csv \
     --split data/splits/loso_seed42_frac1.0.json \
     --checkpoint results/.../epoch=*.ckpt
   ```

2. **Few-Shot Learning:**
   ```bash
   python src/train_finetune.py \
     --config configs/finetune_classify.yaml \
     --split data/splits/fewshot_k5_seed42_frac1.0.json \
     ...
   ```

3. **Temperature Calibration:**
   ```bash
   python src/calibrate.py \
     --checkpoint results/.../epoch=*.ckpt \
     --manifest data/manifest.csv \
     --split data/splits/pooled_seed42_frac1.0.json
   ```

---

See [L40S_OPTIMIZATIONS.md](L40S_OPTIMIZATIONS.md) for technical details on all optimizations.
