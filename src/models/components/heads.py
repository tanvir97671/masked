"""
Head modules: Classification, Projection (contrastive), and Decoder (MAE reconstruction).
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Classification head: linear or MLP.
    """

    def __init__(
        self,
        input_dim: int = 128,
        num_classes: int = 6,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        head_type: str = "mlp",  # "linear" or "mlp"
    ):
        super().__init__()
        if head_type == "linear":
            self.head = nn.Linear(input_dim, num_classes)
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            raise ValueError(f"Unknown head type: {head_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            logits: (batch, num_classes)
        """
        return self.head(x)


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning (SICR).

    Maps encoder features to a lower-dimensional space for similarity computation.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            z: (batch, output_dim) — L2-normalized
        """
        z = self.net(x)
        return nn.functional.normalize(z, dim=-1)


class DecoderHead(nn.Module):
    """
    Lightweight decoder for MAE reconstruction of masked PSD patches.

    Takes encoded visible tokens + mask tokens, processes through
    small transformer/MLP layers, and outputs reconstructed patch values.
    """

    def __init__(
        self,
        d_model: int = 128,
        decoder_dim: int = 64,
        num_patches: int = 12,
        patch_size: int = 16,
        decoder_layers: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.decoder_dim = decoder_dim
        self.num_patches = num_patches
        self.patch_size = patch_size

        # Project from encoder dim to decoder dim
        self.input_proj = nn.Linear(d_model, decoder_dim)

        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_dim) * 0.02)

        # Positional embedding for decoder
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, decoder_dim) * 0.02
        )

        # Decoder transformer layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=4,
            dim_feedforward=decoder_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_layers)

        # Output projection: predict patch values
        self.output_proj = nn.Linear(decoder_dim, patch_size)

    def forward(
        self,
        encoded_tokens: torch.Tensor,
        visible_indices: torch.Tensor,
        mask_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoded_tokens: (batch, num_visible, d_model) — encoder output for visible patches
            visible_indices: (batch, num_visible) — indices of visible patches
            mask_indices: (batch, num_masked) — indices of masked patches

        Returns:
            pred_patches: (batch, num_masked, patch_size) — predicted values for masked patches
        """
        batch_size = encoded_tokens.size(0)
        num_visible = encoded_tokens.size(1)
        num_masked = mask_indices.size(1)

        # Project encoded tokens
        visible_tokens = self.input_proj(encoded_tokens)  # (B, V, decoder_dim)

        # Create mask tokens
        mask_tokens = self.mask_token.expand(batch_size, num_masked, -1)

        # Combine visible + mask tokens
        total_len = num_visible + num_masked
        all_tokens = torch.zeros(
            batch_size, self.num_patches, self.decoder_dim,
            device=encoded_tokens.device, dtype=encoded_tokens.dtype,
        )

        # Place visible tokens at their positions
        batch_idx = torch.arange(batch_size, device=encoded_tokens.device).unsqueeze(1)
        all_tokens[batch_idx, visible_indices] = visible_tokens
        all_tokens[batch_idx, mask_indices] = mask_tokens

        # Add positional encoding
        all_tokens = all_tokens + self.pos_embed[:, :self.num_patches, :]

        # Decode
        decoded = self.decoder(all_tokens)

        # Extract masked positions and predict patch values
        masked_decoded = decoded[batch_idx, mask_indices]  # (B, M, decoder_dim)
        pred_patches = self.output_proj(masked_decoded)  # (B, M, patch_size)

        return pred_patches
