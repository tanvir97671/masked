"""
PSD-Former: Dual-branch model (spectral Transformer + temporal TCN).

Spectral branch: Transformer encoder on patched PSD tokens.
Temporal branch: TCN on consecutive PSD sweeps (if available).

Falls back to spectral-only when temporal data is unavailable.
"""

import torch
import torch.nn as nn
import lightning as L

from .components.patch_embed import PSDPatchEmbedding
from .components.transformer import TransformerEncoder
from .components.tcn import TemporalConvNet
from .components.heads import ClassificationHead


class PSDFormer(L.LightningModule):
    """
    PSD-Former dual-branch classification model.
    """

    def __init__(
        self,
        psd_length: int = 200,
        patch_size: int = 16,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        num_classes: int = 6,
        head_type: str = "mlp",
        head_hidden: int = 128,
        head_dropout: float = 0.2,
        use_temporal: bool = False,
        tcn_channels: list = None,
        gradient_checkpointing: bool = False,
        lr: float = 5e-5,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.use_temporal = use_temporal
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        # Spectral branch
        self.patch_embed = PSDPatchEmbedding(
            psd_length=psd_length,
            patch_size=patch_size,
            d_model=d_model,
            dropout=dropout,
        )
        self.spectral_encoder = TransformerEncoder(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            pool="cls",
            gradient_checkpointing=gradient_checkpointing,
        )

        # Temporal branch (optional)
        if use_temporal:
            if tcn_channels is None:
                tcn_channels = [64, 64, 128]
            self.temporal_encoder = TemporalConvNet(
                in_channels=psd_length,
                channels=tcn_channels,
                dropout=dropout,
            )
            feature_dim = d_model + tcn_channels[-1]
        else:
            self.temporal_encoder = None
            feature_dim = d_model

        # Classification head
        self.classifier = ClassificationHead(
            input_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=head_hidden,
            dropout=head_dropout,
            head_type=head_type,
        )

        # Loss
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, psd: torch.Tensor, temporal_seq: torch.Tensor = None):
        """
        Args:
            psd: (batch, psd_length) — single PSD frame
            temporal_seq: (batch, psd_length, seq_len) — temporal stack (optional)

        Returns:
            logits: (batch, num_classes)
        """
        # Spectral branch
        tokens = self.patch_embed(psd)
        spectral_features = self.spectral_encoder(tokens)  # (batch, d_model)

        if self.use_temporal and temporal_seq is not None:
            temporal_features = self.temporal_encoder(temporal_seq)  # (batch, tcn_out)
            features = torch.cat([spectral_features, temporal_features], dim=-1)
        else:
            features = spectral_features

        logits = self.classifier(features)
        return logits

    def get_features(self, psd: torch.Tensor) -> torch.Tensor:
        """Get encoder features without classification head."""
        tokens = self.patch_embed(psd)
        return self.spectral_encoder(tokens)

    def training_step(self, batch, batch_idx):
        psd, labels, sensor_ids, snr = batch
        logits = self(psd)
        loss = nn.functional.cross_entropy(
            logits, labels, weight=self.class_weights
        )
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        psd, labels, sensor_ids, snr = batch
        logits = self(psd)
        loss = nn.functional.cross_entropy(
            logits, labels, weight=self.class_weights
        )
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        psd, labels, sensor_ids, snr = batch
        logits = self(psd)
        loss = nn.functional.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_acc", acc, sync_dist=True)
        return {"logits": logits, "labels": labels, "snr": snr}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=1e-7
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
