"""
Supervised fine-tuning classifier.

Loads a pretrained encoder (from MPAE checkpoint) and attaches a classification head.
Supports linear probing (frozen encoder) and full fine-tuning.
"""

import torch
import torch.nn as nn
import lightning as L
import torchmetrics

from .components.patch_embed import PSDPatchEmbedding
from .components.transformer import TransformerEncoder
from .components.cnn_baseline import CNNEncoder
from .components.heads import ClassificationHead
from .mpae import MaskedPSDAutoencoder


class PSDClassifier(L.LightningModule):
    """
    Supervised classification module with optional pretrained encoder.
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
        encoder_type: str = "transformer",
        gradient_checkpointing: bool = False,
        pretrained_ckpt: str = None,
        freeze_encoder: bool = False,
        lr: float = 5e-5,
        encoder_lr_scale: float = 0.1,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.lr = lr
        self.encoder_lr_scale = encoder_lr_scale
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.freeze_encoder = freeze_encoder
        self.encoder_type = encoder_type
        self.num_classes = num_classes

        # Build encoder
        if encoder_type == "transformer":
            self.patch_embed = PSDPatchEmbedding(
                psd_length=psd_length,
                patch_size=patch_size,
                d_model=d_model,
                dropout=dropout,
            )
            self.encoder = TransformerEncoder(
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                pool="cls",
                gradient_checkpointing=gradient_checkpointing,
            )
        else:
            self.patch_embed = None
            self.encoder = CNNEncoder(
                psd_length=psd_length,
                output_dim=d_model,
            )

        # Classification head
        self.classifier = ClassificationHead(
            input_dim=d_model,
            num_classes=num_classes,
            hidden_dim=head_hidden,
            dropout=head_dropout,
            head_type=head_type,
        )

        # Load pretrained weights if provided
        if pretrained_ckpt is not None:
            self._load_pretrained(pretrained_ckpt)

        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()

        # Loss
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        # Metrics
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.test_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

    def _load_pretrained(self, ckpt_path: str):
        """Load encoder weights from MPAE checkpoint."""
        print(f"Loading pretrained encoder from {ckpt_path}")
        mpae = MaskedPSDAutoencoder.load_from_checkpoint(ckpt_path)

        if self.encoder_type == "transformer":
            # Load patch embedding weights
            self.patch_embed.load_state_dict(mpae.patch_embed.state_dict())
            # Load encoder weights (shared encoder for both MAE and SICR)
            self.encoder.load_state_dict(mpae.encoder.state_dict())
        else:
            self.encoder.load_state_dict(mpae.encoder.state_dict())

        print("Pretrained encoder loaded successfully.")

    def _freeze_encoder(self):
        """Freeze encoder parameters for linear probing."""
        if self.patch_embed is not None:
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen (linear probing mode).")

    def forward(self, psd: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psd: (batch, psd_length)
        Returns:
            logits: (batch, num_classes)
        """
        if self.encoder_type == "transformer":
            tokens = self.patch_embed(psd)
            features = self.encoder(tokens)
        else:
            features = self.encoder(psd)

        return self.classifier(features)

    def get_features(self, psd: torch.Tensor) -> torch.Tensor:
        """Get encoder features without classification head."""
        if self.encoder_type == "transformer":
            tokens = self.patch_embed(psd)
            return self.encoder(tokens)
        else:
            return self.encoder(psd)

    def training_step(self, batch, batch_idx):
        psd, labels, sensor_ids, snr = batch
        logits = self(psd)
        loss = nn.functional.cross_entropy(
            logits, labels, weight=self.class_weights
        )
        self.train_acc(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        psd, labels, sensor_ids, snr = batch
        logits = self(psd)
        loss = nn.functional.cross_entropy(logits, labels)
        self.val_acc(logits, labels)
        self.val_f1(logits, labels)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc, prog_bar=True, sync_dist=True)
        self.log("val_macro_f1", self.val_f1, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        psd, labels, sensor_ids, snr = batch
        logits = self(psd)
        self.test_acc(logits, labels)
        self.test_f1(logits, labels)
        self.log("test_acc", self.test_acc, sync_dist=True)
        self.log("test_macro_f1", self.test_f1, sync_dist=True)
        return {"logits": logits, "labels": labels, "snr": snr}

    def configure_optimizers(self):
        if self.freeze_encoder:
            # Only train classifier head
            params = self.classifier.parameters()
            optimizer = torch.optim.AdamW(
                params, lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            # Differential LR: encoder gets lower LR
            encoder_params = []
            head_params = list(self.classifier.parameters())

            if self.patch_embed is not None:
                encoder_params.extend(self.patch_embed.parameters())
            encoder_params.extend(self.encoder.parameters())

            optimizer = torch.optim.AdamW([
                {"params": encoder_params, "lr": self.lr * self.encoder_lr_scale},
                {"params": head_params, "lr": self.lr},
            ], weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=1e-7
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
