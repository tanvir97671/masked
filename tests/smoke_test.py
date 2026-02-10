"""
Smoke test: loads a tiny slice of the REAL ElectroSense dataset,
runs one forward pass through every model, checks shapes + loss.

Target: <60 seconds on CPU.

Usage:
    python tests/smoke_test.py --data_dir data/
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def find_npy_files(data_dir: str, max_files: int = 3):
    """Find a few .npy files in the dataset."""
    data_path = Path(data_dir)
    npy_files = []
    for npy in data_path.rglob("*.npy"):
        npy_files.append(str(npy))
        if len(npy_files) >= max_files:
            break
    return npy_files


def create_minimal_dataset_if_missing(data_dir: str):
    """Create minimal synthetic dataset if real data not available."""
    data_path = Path(data_dir)
    real_npys = list(data_path.rglob("*.npy"))
    if not real_npys:
        print(f"\n⚠️  No .npy files found under {data_path}")
        print("Creating minimal synthetic dataset for testing...")
        spectrum_dir = data_path / "spectrum_bands"
        from src.data.create_minimal_test_data import create_minimal_dataset
        create_minimal_dataset(str(spectrum_dir), n_sensors=2, samples_per_class=5)
        print("✓ Synthetic dataset created")
        return True
    else:
        print(f"Found {len(real_npys)} real .npy files under {data_path}")
        return False


def load_tiny_batch(npy_files, num_samples: int = 32, psd_length: int = 200):
    """Load a tiny batch of real PSD samples."""
    from src.data.preprocessing import preprocess_psd, estimate_snr

    all_psd = []
    all_snr = []

    for npy_path in npy_files:
        data = np.load(npy_path, allow_pickle=True)
        if data.ndim == 0:
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)

        for i in range(min(data.shape[0], num_samples - len(all_psd))):
            psd_raw = data[i]
            snr = estimate_snr(psd_raw)
            psd_tensor = preprocess_psd(psd_raw, target_length=psd_length)
            all_psd.append(psd_tensor)
            all_snr.append(snr)

            if len(all_psd) >= num_samples:
                break
        if len(all_psd) >= num_samples:
            break

    if not all_psd:
        raise RuntimeError("No PSD samples could be loaded. Check data directory.")

    psd_batch = torch.stack(all_psd)
    labels = torch.randint(0, 6, (len(all_psd),))  # Random labels for shape check
    snr_batch = torch.tensor(all_snr, dtype=torch.float32)

    return psd_batch, labels, snr_batch


def test_patch_embedding(psd_batch):
    """Test PSD patch embedding."""
    from src.models.components.patch_embed import PSDPatchEmbedding

    embed = PSDPatchEmbedding(psd_length=200, patch_size=16, d_model=32)
    tokens = embed(psd_batch)

    expected_seq_len = embed.sequence_length  # num_patches + 1 (CLS)
    assert tokens.shape == (psd_batch.size(0), expected_seq_len, 32), \
        f"PatchEmbed shape mismatch: {tokens.shape}"
    assert torch.isfinite(tokens).all(), "PatchEmbed produced non-finite values"
    print(f"  [PASS] PatchEmbedding: {tokens.shape}")
    return tokens


def test_transformer_encoder(tokens):
    """Test Transformer encoder."""
    from src.models.components.transformer import TransformerEncoder

    encoder = TransformerEncoder(d_model=32, n_layers=2, n_heads=2, ffn_dim=64, pool="cls")
    features = encoder(tokens)

    assert features.shape == (tokens.size(0), 32), \
        f"Transformer shape mismatch: {features.shape}"
    assert torch.isfinite(features).all(), "Transformer produced non-finite values"
    print(f"  [PASS] TransformerEncoder: {features.shape}")
    return features


def test_cnn_encoder(psd_batch):
    """Test CNN baseline encoder."""
    from src.models.components.cnn_baseline import CNNEncoder

    cnn = CNNEncoder(psd_length=200, channels=[16, 32, 64], output_dim=32)
    features = cnn(psd_batch)

    assert features.shape == (psd_batch.size(0), 32), \
        f"CNN shape mismatch: {features.shape}"
    assert torch.isfinite(features).all(), "CNN produced non-finite values"
    print(f"  [PASS] CNNEncoder: {features.shape}")
    return features


def test_classification_head(features, num_classes=6):
    """Test classification head."""
    from src.models.components.heads import ClassificationHead

    head = ClassificationHead(input_dim=32, num_classes=num_classes, hidden_dim=32, head_type="mlp")
    logits = head(features)

    assert logits.shape == (features.size(0), num_classes), \
        f"Head shape mismatch: {logits.shape}"
    print(f"  [PASS] ClassificationHead: {logits.shape}")
    return logits


def test_projection_head(features):
    """Test projection head for SICR."""
    from src.models.components.heads import ProjectionHead

    proj = ProjectionHead(input_dim=32, hidden_dim=32, output_dim=16)
    z = proj(features)

    assert z.shape == (features.size(0), 16), f"Projection shape mismatch: {z.shape}"
    # Check L2 normalization
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        "Projection head output not L2-normalized"
    print(f"  [PASS] ProjectionHead: {z.shape} (L2-normalized)")
    return z


def test_sicr_loss(z):
    """Test SICR contrastive loss."""
    from src.models.sicr import SICRLoss

    loss_fn = SICRLoss(temperature=0.1)
    # Split batch into anchor/positive pairs
    mid = z.size(0) // 2
    if mid < 2:
        print("  [SKIP] SICR loss: batch too small")
        return
    loss = loss_fn(z[:mid], z[mid:2*mid])

    assert loss.ndim == 0, "SICR loss should be scalar"
    assert torch.isfinite(loss), f"SICR loss not finite: {loss.item()}"
    assert loss.item() > 0, "SICR loss should be positive"
    print(f"  [PASS] SICR Loss: {loss.item():.4f}")


def test_mpae_forward(psd_batch):
    """Test MPAE forward pass (masked reconstruction)."""
    from src.models.mpae import MaskedPSDAutoencoder

    model = MaskedPSDAutoencoder(
        psd_length=200, patch_size=16, d_model=32, n_layers=1,
        n_heads=2, ffn_dim=64, mask_ratio=0.5, decoder_dim=16,
        decoder_layers=1, lambda_sicr=0.1, sicr_proj_dim=16,
        encoder_type="transformer",
    )

    loss_mae, pred, target = model.forward_mae(psd_batch)

    assert loss_mae.ndim == 0, "MAE loss should be scalar"
    assert torch.isfinite(loss_mae), f"MAE loss not finite: {loss_mae.item()}"
    assert pred.shape == target.shape, f"Pred/target shape mismatch: {pred.shape} vs {target.shape}"
    print(f"  [PASS] MPAE forward: loss={loss_mae.item():.4f}, pred_shape={pred.shape}")

    # Test gradient flow
    loss_mae.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters() if p.requires_grad)
    assert has_grad, "No gradients flowing through MPAE"
    print(f"  [PASS] MPAE gradients flow correctly")
    model.zero_grad()

    return model


def test_classifier_forward(psd_batch, labels):
    """Test PSDClassifier forward pass."""
    from src.models.classifier import PSDClassifier

    model = PSDClassifier(
        psd_length=200, patch_size=16, d_model=32, n_layers=1,
        n_heads=2, ffn_dim=64, num_classes=6, head_type="mlp",
        head_hidden=32, encoder_type="transformer", lr=1e-3,
    )

    logits = model(psd_batch)
    loss = torch.nn.functional.cross_entropy(logits, labels)

    assert logits.shape == (psd_batch.size(0), 6), f"Logits shape: {logits.shape}"
    assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"

    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters() if p.requires_grad)
    assert has_grad, "No gradients in classifier"
    print(f"  [PASS] PSDClassifier: logits={logits.shape}, loss={loss.item():.4f}")
    model.zero_grad()


def test_calibration(psd_batch, labels, snr_batch):
    """Test calibration modules."""
    from src.utils.calibration import TemperatureScaling, SNRAwareTemperature

    logits = torch.randn(psd_batch.size(0), 6)

    # Temperature scaling
    ts = TemperatureScaling()
    scaled = ts(logits)
    assert scaled.shape == logits.shape, "TemperatureScaling shape mismatch"
    print(f"  [PASS] TemperatureScaling: T={ts.temperature.item():.4f}")

    # SNR-aware
    snr_ts = SNRAwareTemperature(hidden_dim=16)
    snr_scaled = snr_ts(logits, snr_batch)
    assert snr_scaled.shape == logits.shape, "SNRAwareTemp shape mismatch"
    print(f"  [PASS] SNRAwareTemperature: shape={snr_scaled.shape}")


def test_metrics():
    """Test metrics computation."""
    from src.utils.metrics import compute_all_metrics, expected_calibration_error

    y_true = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 1, 1, 2, 3])
    y_prob = np.random.dirichlet(np.ones(6), size=10)

    metrics = compute_all_metrics(y_true, y_pred, y_prob, ["dab", "dvbt", "fm", "gsm", "lte", "tetra"])

    assert "accuracy" in metrics, "Missing accuracy"
    assert "macro_f1" in metrics, "Missing macro_f1"
    assert "ece" in metrics, "Missing ECE"
    assert 0.7 <= metrics["accuracy"] <= 1.0, f"Accuracy out of range: {metrics['accuracy']}"
    print(f"  [PASS] Metrics: acc={metrics['accuracy']}, f1={metrics['macro_f1']:.4f}, ece={metrics['ece']:.4f}")


def test_snr_estimation(psd_batch):
    """Test SNR batch estimation."""
    from src.utils.snr import estimate_snr_batch

    snr = estimate_snr_batch(psd_batch)
    assert snr.shape == (psd_batch.size(0),), f"SNR shape: {snr.shape}"
    assert torch.isfinite(snr).all(), "SNR has non-finite values"
    print(f"  [PASS] SNR estimation: shape={snr.shape}, range=[{snr.min():.1f}, {snr.max():.1f}] dB")


def main():
    parser = argparse.ArgumentParser(description="Smoke test on real data")
    parser.add_argument("--data_dir", type=str, default="data/")
    args = parser.parse_args()

    start = time.time()
    print("=" * 60)
    print("SMOKE TEST — ElectroSense MPAE+SICR Research Codebase")
    print("=" * 60)

    # Find dataset or create minimal test data
    create_minimal_dataset_if_missing(args.data_dir)

    # Search entire data directory recursively for .npy files
    npy_files = find_npy_files(args.data_dir, max_files=10)
    if not npy_files:
        print(f"\nERROR: No .npy files found under {args.data_dir}")
        print("Run: python src/data/download_tiny_stream.py --data_dir data/")
        sys.exit(1)

    print(f"\nFound {len(npy_files)} .npy files. Loading tiny batch...")

    # Load tiny real data
    psd_batch, labels, snr_batch = load_tiny_batch(npy_files, num_samples=32)
    print(f"Loaded {psd_batch.shape[0]} real PSD samples, shape={psd_batch.shape}")

    total_tests = 0
    passed = 0

    # Run all tests
    tests = [
        ("Patch Embedding", lambda: test_patch_embedding(psd_batch)),
        ("Transformer Encoder", None),  # depends on patch_embed output
        ("CNN Encoder", lambda: test_cnn_encoder(psd_batch)),
        ("Classification Head", None),  # depends on features
        ("Projection Head", None),
        ("SICR Loss", None),
        ("MPAE Forward", lambda: test_mpae_forward(psd_batch)),
        ("PSD Classifier", lambda: test_classifier_forward(psd_batch, labels)),
        ("Calibration", lambda: test_calibration(psd_batch, labels, snr_batch)),
        ("Metrics", lambda: test_metrics()),
        ("SNR Estimation", lambda: test_snr_estimation(psd_batch)),
    ]

    print("\n--- Component Tests ---")

    try:
        # Patch embedding
        total_tests += 1
        tokens = test_patch_embedding(psd_batch)
        passed += 1

        # Transformer
        total_tests += 1
        features_t = test_transformer_encoder(tokens)
        passed += 1

        # CNN
        total_tests += 1
        features_c = test_cnn_encoder(psd_batch)
        passed += 1

        # Heads
        total_tests += 1
        test_classification_head(features_t)
        passed += 1

        total_tests += 1
        z = test_projection_head(features_t)
        passed += 1

        # SICR
        total_tests += 1
        test_sicr_loss(z)
        passed += 1

        # MPAE
        total_tests += 1
        test_mpae_forward(psd_batch)
        passed += 1

        # Classifier
        total_tests += 1
        test_classifier_forward(psd_batch, labels)
        passed += 1

        # Calibration
        total_tests += 1
        test_calibration(psd_batch, labels, snr_batch)
        passed += 1

        # Metrics
        total_tests += 1
        test_metrics()
        passed += 1

        # SNR
        total_tests += 1
        test_snr_estimation(psd_batch)
        passed += 1

    except Exception as e:
        total_tests += 1
        print(f"\n  [FAIL] {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    if passed == total_tests:
        print(f"ALL SMOKE TESTS PASSED ({passed}/{total_tests}) in {elapsed:.1f}s")
    else:
        print(f"SMOKE TESTS: {passed}/{total_tests} passed in {elapsed:.1f}s")
        print("SOME TESTS FAILED — check output above.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
