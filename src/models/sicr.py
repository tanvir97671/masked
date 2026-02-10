"""
Sensor-Invariant Contrastive Regularization (SICR).

InfoNCE loss that pulls together representations of the same frequency band
from different sensors, and pushes apart different bands.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SICRLoss(nn.Module):
    """
    Sensor-Invariant Contrastive Regularization loss.

    Given a batch of (anchor, positive) embedding pairs where:
      - anchor and positive are the same band from different sensors
      - negatives are all other samples in the batch

    Computes InfoNCE:
        L = -log( exp(sim(z, z+)/τ) / Σ_k exp(sim(z, z_k)/τ) )
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_anchor: torch.Tensor,
        z_positive: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_anchor: (batch, proj_dim) — L2-normalized embeddings of anchors
            z_positive: (batch, proj_dim) — L2-normalized embeddings of positives

        Returns:
            Scalar InfoNCE loss.
        """
        batch_size = z_anchor.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=z_anchor.device, requires_grad=True)

        # Concatenate all embeddings: [anchors; positives]
        # shape: (2B, proj_dim)
        z_all = torch.cat([z_anchor, z_positive], dim=0)

        # Similarity matrix: (2B, 2B)
        sim = torch.mm(z_all, z_all.t()) / self.temperature

        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -1e9)

        # Positive pairs: anchor[i] <-> positive[i]
        # For anchor[i] (row i), positive is at index i + B
        # For positive[i] (row i + B), positive is at index i
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size),
        ]).to(sim.device)

        loss = F.cross_entropy(sim, labels)
        return loss


def compute_sicr_loss(
    encoder: nn.Module,
    projection_head: nn.Module,
    anchor_psd: torch.Tensor,
    positive_psd: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Convenience function: encode + project + SICR loss.

    Args:
        encoder: Produces (batch, d_model) from PSD tokens.
        projection_head: Maps (batch, d_model) -> (batch, proj_dim), L2-normalized.
        anchor_psd: (batch, num_tokens, d_model) — embedded anchor PSD tokens.
        positive_psd: (batch, num_tokens, d_model) — embedded positive PSD tokens.

    Returns:
        Scalar SICR loss.
    """
    z_a = projection_head(encoder(anchor_psd))
    z_p = projection_head(encoder(positive_psd))
    criterion = SICRLoss(temperature)
    return criterion(z_a, z_p)
