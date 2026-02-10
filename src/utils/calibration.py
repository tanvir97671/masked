"""
Post-hoc calibration methods.

1. TemperatureScaling: single temperature on val set.
2. SNRAwareTemperature: MLP that takes estimated SNR -> per-sample temperature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class TemperatureScaling(nn.Module):
    """
    Standard temperature scaling for calibration.

    Learns a single temperature T on validation set (minimizes NLL).
    Parameterized as log(T) to ensure T > 0.
    """

    def __init__(self):
        super().__init__()
        # Store log(T) to guarantee T > 0 after exp()
        self.log_temperature = nn.Parameter(torch.zeros(1))  # log(1.0) = 0

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Fit temperature on validation logits/labels.

        Args:
            logits: (N, C) pre-softmax logits from the model.
            labels: (N,) ground truth labels.

        Returns:
            Final NLL loss.
        """
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)

        def eval_fn():
            optimizer.zero_grad()
            scaled = self(logits)
            loss = F.cross_entropy(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(eval_fn)

        final_loss = F.cross_entropy(self(logits), labels).item()
        print(f"Temperature scaling: T={self.temperature.item():.4f}, NLL={final_loss:.4f}")
        return final_loss


class SNRAwareTemperature(nn.Module):
    """
    SNR-conditioned temperature scaling.

    A small MLP predicts per-sample temperature from estimated SNR.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Ensure T > 0
        )
        # Initialize to produce T ≈ 1
        with torch.no_grad():
            self.net[-2].bias.fill_(0.5)

    def forward(self, logits: torch.Tensor, snr: torch.Tensor) -> torch.Tensor:
        """
        Scale logits by SNR-conditioned temperature.

        Args:
            logits: (batch, C)
            snr: (batch,) estimated SNR in dB

        Returns:
            Scaled logits: (batch, C)
        """
        snr_input = snr.unsqueeze(-1)  # (batch, 1)
        temperature = self.net(snr_input)  # (batch, 1)
        temperature = temperature.clamp(min=0.1, max=10.0)  # Safety clamp
        return logits / temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        snr: torch.Tensor,
        lr: float = 1e-3,
        epochs: int = 200,
    ) -> float:
        """
        Fit SNR-aware temperature on validation data.

        Args:
            logits: (N, C) pre-softmax logits.
            labels: (N,) ground truth.
            snr: (N,) estimated SNR values.

        Returns:
            Final NLL loss.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            scaled = self(logits, snr)
            loss = F.cross_entropy(scaled, labels)
            loss.backward()
            optimizer.step()

        final_loss = F.cross_entropy(self(logits, snr), labels).item()
        print(f"SNR-aware temperature: NLL={final_loss:.4f}")
        return final_loss
