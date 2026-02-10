"""
Masked PSD Autoencoder (MPAE) — LightningModule for self-supervised pretraining.

Combines:
  - Masked reconstruction loss (MSE on masked PSD patches)
  - SICR contrastive loss (sensor-invariant cross-sensor pairs)

Supports both Transformer and CNN encoders.
"""

import torch
import torch.nn as nn
import lightning as L

from .components.patch_embed import PSDPatchEmbedding
from .components.transformer import TransformerEncoder
from .components.cnn_baseline import CNNEncoder
from .components.heads import ProjectionHead, DecoderHead
from .sicr import SICRLoss


class MaskedPSDAutoencoder(L.LightningModule):
    """
    Self-supervised pretraining module: MPAE + SICR.
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
        mask_ratio: float = 0.6,
        decoder_dim: int = 64,
        decoder_layers: int = 2,
        lambda_sicr: float = 0.1,
        sicr_temperature: float = 0.07,
        sicr_proj_dim: int = 64,
        encoder_type: str = "transformer",
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
        max_epochs: int = 80,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.psd_length = psd_length
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.lambda_sicr = lambda_sicr
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        # Patch embedding
        self.patch_embed = PSDPatchEmbedding(
            psd_length=psd_length,
            patch_size=patch_size,
            d_model=d_model,
            dropout=dropout,
        )
        num_patches = self.patch_embed.num_patches

        # Encoder (single shared encoder for both MAE and SICR)
        self.encoder_type = encoder_type
        if encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                pool="cls",  # CLS pooling for SICR; MAE manually iterates layers
            )
        else:
            self.encoder = CNNEncoder(
                psd_length=psd_length,
                output_dim=d_model,
            )

        # Decoder for masked reconstruction
        self.decoder = DecoderHead(
            d_model=d_model,
            decoder_dim=decoder_dim,
            num_patches=num_patches,
            patch_size=patch_size,
            decoder_layers=decoder_layers,
        )

        # SICR projection head + loss
        self.projection_head = ProjectionHead(
            input_dim=d_model,
            hidden_dim=d_model,
            output_dim=sicr_proj_dim,
        )
        self.sicr_loss = SICRLoss(temperature=sicr_temperature)

    def random_masking(
        self, batch_size: int, num_patches: int, device: torch.device,
    ):
        """
        Generate random mask indices.

        Returns:
            visible_indices: (batch, num_visible)
            mask_indices: (batch, num_masked)
        """
        num_masked = int(num_patches * self.mask_ratio)
        num_visible = num_patches - num_masked

        # Random permutation per sample
        noise = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)

        visible_indices = ids_shuffle[:, :num_visible]
        mask_indices = ids_shuffle[:, num_visible:]

        # Sort for deterministic ordering
        visible_indices, _ = torch.sort(visible_indices, dim=1)
        mask_indices, _ = torch.sort(mask_indices, dim=1)

        return visible_indices, mask_indices

    def forward_mae(self, psd: torch.Tensor):
        """
        Forward pass for masked autoencoder reconstruction.

        Args:
            psd: (batch, psd_length)

        Returns:
            loss_mae: scalar reconstruction loss
            pred_patches: (batch, num_masked, patch_size)
            target_patches: (batch, num_masked, patch_size)
        """
        batch_size = psd.size(0)

        # Get patches (without CLS token processing — just raw patches)
        effective_len = self.patch_embed.num_patches * self.patch_size
        psd_trunc = psd[:, :effective_len]
        patches = psd_trunc.view(batch_size, self.patch_embed.num_patches, self.patch_size)

        # Generate mask
        visible_idx, mask_idx = self.random_masking(
            batch_size, self.patch_embed.num_patches, psd.device
        )

        # Extract visible patches
        batch_idx = torch.arange(batch_size, device=psd.device).unsqueeze(1)
        visible_patches = patches[batch_idx, visible_idx]  # (B, V, patch_size)

        # Project visible patches -> tokens
        visible_tokens = self.patch_embed.projection(visible_patches)  # (B, V, d_model)

        # Simple positional encoding addition for visible tokens
        pos = self.patch_embed.pos_enc.pe[0, 1:, :]  # skip CLS pos
        visible_pos = pos[visible_idx]  # (B, V, d_model)
        visible_tokens = visible_tokens + visible_pos

        # Encode visible tokens through transformer layers
        if self.encoder_type == "transformer":
            for layer in self.encoder.layers:
                visible_tokens = layer(visible_tokens)
            visible_tokens = self.encoder.final_norm(visible_tokens)
        # For CNN encoder, fall back to full encoding (no masking advantage)

        # Decode: predict masked patches
        pred_patches = self.decoder(visible_tokens, visible_idx, mask_idx)

        # Target: original patch values at masked positions
        target_patches = patches[batch_idx, mask_idx]  # (B, M, patch_size)

        # MSE loss on masked patches only
        loss_mae = nn.functional.mse_loss(pred_patches, target_patches)

        return loss_mae, pred_patches, target_patches

    def forward_sicr(
        self,
        anchor_psd: torch.Tensor,
        positive_psd: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for SICR contrastive loss.

        Args:
            anchor_psd: (batch, psd_length)
            positive_psd: (batch, psd_length)

        Returns:
            loss_sicr: scalar contrastive loss
        """
        # Encode both through full pipeline (with CLS pooling)
        anchor_tokens = self.patch_embed(anchor_psd)
        positive_tokens = self.patch_embed(positive_psd)

        if self.encoder_type == "transformer":
            z_a = self.encoder(anchor_tokens)
            z_p = self.encoder(positive_tokens)
        else:
            z_a = self.encoder(anchor_psd)
            z_p = self.encoder(positive_psd)

        # Project
        z_a = self.projection_head(z_a)
        z_p = self.projection_head(z_p)

        return self.sicr_loss(z_a, z_p)

    def get_encoder_output(self, psd: torch.Tensor) -> torch.Tensor:
        """
        Get encoder embedding for downstream tasks.

        Args:
            psd: (batch, psd_length)
        Returns:
            (batch, d_model)
        """
        tokens = self.patch_embed(psd)
        if self.encoder_type == "transformer":
            return self.encoder(tokens)
        else:
            return self.encoder(psd)

    def training_step(self, batch, batch_idx):
        # Contrastive dataset: (anchor, positive, label, is_cross_sensor)
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            anchor, positive, labels, is_cross = batch

            # MAE loss on anchor
            loss_mae, _, _ = self.forward_mae(anchor)

            # SICR loss
            loss_sicr = self.forward_sicr(anchor, positive)

            loss = loss_mae + self.lambda_sicr * loss_sicr

            self.log("train_loss_mae", loss_mae, prog_bar=True)
            self.log("train_loss_sicr", loss_sicr, prog_bar=True)
            self.log("train_loss", loss, prog_bar=True)
        else:
            # Standard dataset without contrastive pairs
            psd = batch[0] if isinstance(batch, (list, tuple)) else batch
            loss_mae, _, _ = self.forward_mae(psd)
            loss = loss_mae
            self.log("train_loss_mae", loss_mae, prog_bar=True)
            self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        psd = batch[0] if isinstance(batch, (list, tuple)) else batch
        loss_mae, _, _ = self.forward_mae(psd)
        self.log("val_loss", loss_mae, prog_bar=True, sync_dist=True)
        return loss_mae

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
