"""
PSD Patch Embedding with Frequency-Aware Positional Encoding.

Splits a PSD vector of length B into N = B / P non-overlapping patches,
projects each to d dimensions, and adds sinusoidal positional encoding
based on patch index (and optionally real center frequency).
"""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Can use patch index or actual frequency metadata.
    """

    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            x + positional encoding, same shape.
        """
        return x + self.pe[:, : x.size(1), :]


class PSDPatchEmbedding(nn.Module):
    """
    Split PSD vector into patches and project to d_model dimensions.

    Input: (batch, psd_length)
    Output: (batch, num_patches, d_model)
    """

    def __init__(
        self,
        psd_length: int = 200,
        patch_size: int = 16,
        d_model: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.psd_length = psd_length
        self.patch_size = patch_size
        self.d_model = d_model

        # Number of patches (truncate last incomplete patch)
        self.num_patches = psd_length // patch_size

        # Linear projection from patch to d_model
        self.projection = nn.Linear(patch_size, d_model)

        # CLS token (optional, for classification)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=self.num_patches + 1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        return_patches: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, psd_length) raw PSD vector
            return_patches: if True, also return raw patches before projection

        Returns:
            tokens: (batch, num_patches + 1, d_model) — includes CLS token at position 0
        """
        batch_size = x.size(0)

        # Truncate to fit exact patches
        effective_len = self.num_patches * self.patch_size
        x = x[:, :effective_len]

        # Reshape into patches: (batch, num_patches, patch_size)
        patches = x.view(batch_size, self.num_patches, self.patch_size)

        # Project patches
        tokens = self.projection(patches)  # (batch, num_patches, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (batch, num_patches+1, d_model)

        # Add positional encoding
        tokens = self.pos_enc(tokens)
        tokens = self.layer_norm(tokens)
        tokens = self.dropout(tokens)

        if return_patches:
            return tokens, patches
        return tokens

    @property
    def output_dim(self) -> int:
        return self.d_model

    @property
    def sequence_length(self) -> int:
        return self.num_patches + 1  # +1 for CLS token
