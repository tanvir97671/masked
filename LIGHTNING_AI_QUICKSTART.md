# Lightning AI Studio Quick Start (L4 GPU, 24GB)

## Prerequisites
- Lightning AI Studio with **L4 GPU** (24GB VRAM)
- Python 3.12+, PyTorch 2.x with CUDA

---

## Step-by-Step (25% Dataset — Credit-Saving Mode)

### 1. Clone & Install (~2 min)
```bash
git clone https://github.com/tanvir97671/masked.git
cd masked
pip install -r requirements.txt
```

### 2. Set WandB API Key (optional but recommended)
```bash
export WANDB_API_KEY="your-key-here"
```
Get your key from https://wandb.ai/authorize

### 3. Download & Parse Data (~2 min, 30MB stream)
```bash
python src/data/download_tiny_stream.py
python src/data/parse_dataset.py
```
Expected: `Found 50 .npy files, 11 sensors, 21,351 total samples`

### 4. Generate Splits (~5 sec)
```bash
python src/data/splits.py \
  --manifest data/manifest.csv \
  --output data/splits/ \
  --fraction 0.25 \
  --seed 42
```
Expected: `train=~3735, val=~801, test=~801`

### 5. SSL Pretrain — MPAE + SICR (~20-30 min, 20 epochs)
```bash
python src/train_pretrain.py \
  --config configs/pretrain_mpae.yaml \
  --manifest data/manifest.csv \
  --split data/splits/pooled_seed42_frac0.25.json \
  --epochs 20 \
  --batch_size 256 \
  --accelerator gpu \
  --devices 1
```
**Config:**  256-d model, 6 layers, bf16-mixed, gradient checkpointing

### 6. Supervised Fine-tune (~10-15 min, 30 epochs)
```bash
python src/train_finetune.py \
  --config configs/finetune_classify.yaml \
  --manifest data/manifest.csv \
  --split data/splits/pooled_seed42_frac0.25.json \
  --pretrained results/checkpoints/pretrain/mpae-last.ckpt \
  --epochs 30 \
  --batch_size 256 \
  --accelerator gpu \
  --devices 1
```

### 7. Baseline CNN (comparison, ~5-10 min)
```bash
python src/train_baseline.py \
  --config configs/finetune_classify.yaml \
  --manifest data/manifest.csv \
  --split data/splits/pooled_seed42_frac0.25.json \
  --epochs 30 \
  --batch_size 256 \
  --accelerator gpu \
  --devices 1 \
  --encoder_type cnn
```

### 8. Evaluate
```bash
python src/evaluate.py \
  --protocol pooled \
  --ckpt results/checkpoints/finetune/classifier-last.ckpt \
  --config configs/finetune_classify.yaml \
  --tag pretrained
```

---

## L4 GPU Optimizations Applied
| Setting | Value | Purpose |
|---------|-------|---------|
| d_model | 256 | Fits L4 24GB memory |
| n_layers | 6 | Sufficient depth for PSD data |
| ffn_dim | 1024 | 4x expansion ratio |
| batch_size | 256 | Max for L4 with bf16 |
| precision | bf16-mixed | Halves memory, native L4 support |
| gradient_checkpointing | true | Trades compute for memory |
| torch.compile | true | ~20-40% speedup |
| num_workers | 4 | Balanced for L4 instance |

## WandB Integration
Set `WANDB_API_KEY` environment variable before training. All training scripts
auto-detect it and log to project `ieee-psd-ssl`. If not set, default CSV logging is used.

## Estimated Total Time (25% data)
- Setup + data: ~5 min
- Pretrain 20 epochs: ~20-30 min
- Finetune 30 epochs: ~10-15 min
- Baseline 30 epochs: ~5-10 min
- **Total: ~45-60 min**
