"""
1D CNN / ResNet baseline encoder for PSD vectors.

Used as Baseline B1 (supervised from scratch) and as an
alternative encoder for MPAE ablation (CNN vs Transformer).
"""

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """1D residual block with two conv layers."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.GELU()

        # Shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class CNNEncoder(nn.Module):
    """
    Small 1D ResNet encoder for PSD vectors.

    Input: (batch, psd_length) — raw PSD
    Output: (batch, output_dim) — feature vector
    """

    def __init__(
        self,
        psd_length: int = 200,
        channels: list = None,
        output_dim: int = 128,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]

        self._output_dim = output_dim

        # Initial conv
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.GELU(),
        )

        # Residual blocks
        blocks = []
        in_ch = channels[0]
        for out_ch in channels:
            blocks.append(ResidualBlock1D(in_ch, out_ch, stride=2))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        # Global average pool + projection
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projector = nn.Linear(channels[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, psd_length) or (batch, 1, psd_length)
        Returns:
            (batch, output_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, psd_length)

        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)  # (batch, channels[-1])
        x = self.projector(x)
        return x

    @property
    def output_dim(self) -> int:
        return self._output_dim
