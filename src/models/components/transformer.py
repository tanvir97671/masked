"""
Lightweight Transformer Encoder for PSD token sequences.

Uses pre-norm (LayerNorm before attention/FFN) for stable training.
"""

import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    """Single pre-norm Transformer encoder layer."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Pre-norm self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=src_key_padding_mask,
        )
        x = x + attn_out

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer encoder layers with final LayerNorm.

    Input: (batch, seq_len, d_model) — from PSDPatchEmbedding
    Output: (batch, d_model) — CLS token or mean-pooled
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        pool: str = "cls",  # "cls" or "mean"
        gradient_checkpointing: bool = False,  # Save memory for large models
    ):
        super().__init__()
        self.d_model = d_model
        self.pool = pool
        self.gradient_checkpointing = gradient_checkpointing

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, d_model)
        """
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Use checkpoint for memory efficiency
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, src_key_padding_mask, use_reentrant=False
                )
            else:
                x = layer(x, src_key_padding_mask)

        x = self.final_norm(x)

        if self.pool == "cls":
            return x[:, 0, :]  # CLS token
        elif self.pool == "mean":
            return x.mean(dim=1)
        else:
            raise ValueError(f"Unknown pool mode: {self.pool}")

    @property
    def output_dim(self) -> int:
        return self.d_model
