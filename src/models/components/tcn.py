"""
Temporal Convolutional Network (TCN) for temporal PSD sequences.

Used as the temporal branch in PSD-Former when multiple consecutive
PSD sweeps are available for the same sensor-band.
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    """Single TCN block with dilated causal convolution + residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        ))
        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        ))
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, length)
        Returns:
            (batch, out_channels, length)
        """
        out = self.conv1(x)
        # Remove future padding (causal) — conv1 adds self.padding extra on right
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        # Remove future padding (causal) — conv2 adds self.padding extra on right
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.relu(out)
        out = self.dropout(out)

        # Residual must match: truncate to same length as out
        res = self.residual(x)
        if res.size(-1) != out.size(-1):
            res = res[:, :, :out.size(-1)]
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Multi-layer TCN with increasing dilation.

    Input: (batch, in_channels, seq_length)
    Output: (batch, out_channels) — global average pooled
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: list = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 64, 128]

        layers = []
        for i, out_ch in enumerate(channels):
            in_ch = in_channels if i == 0 else channels[i - 1]
            dilation = 2 ** i
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )

        self.network = nn.Sequential(*layers)
        self._output_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_length)
        Returns:
            (batch, output_dim) — global average pooled
        """
        out = self.network(x)  # (batch, channels[-1], seq_length)
        return out.mean(dim=-1)  # global avg pool

    @property
    def output_dim(self) -> int:
        return self._output_dim
