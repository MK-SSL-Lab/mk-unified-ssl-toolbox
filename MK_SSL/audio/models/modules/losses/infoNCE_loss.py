import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss with trainable bilinear similarity (COLA style).

    * For a batch of B positive pairs we build a 2B × 2B similarity matrix.
    * The diagonal (self‑similarity) is **masked** so it cannot become the
      highest logit, matching Saeed et al. 2021.

    Args:
        temperature (float): Logit scaling factor (τ); must be non‑zero.
        input_dim (int): Dimensionality of each embedding (default 512).

    Inputs:
        out0 (Tensor): Embeddings from view 1, shape ``(B, D)``.
        out1 (Tensor): Embeddings from view 2, shape ``(B, D)``.

    Returns:
        Tensor: Scalar loss.
    """

    def __init__(self, temperature: float = 0.2, input_dim: int = 512):
        super().__init__()
        if abs(temperature) < 1e-8:
            raise ValueError("Temperature must be non‑zero.")
        self.temperature = temperature
        # Bilinear similarity: zᵀ W z′   (weight shape 1×D×D)
        self.similarity = nn.Bilinear(input_dim, input_dim, 1, bias=False)

    def forward(self, out0: torch.Tensor, out1: torch.Tensor) -> torch.Tensor:
        # --- normalise embeddings ---
        z0 = F.normalize(out0, dim=1)
        z1 = F.normalize(out1, dim=1)
        z  = torch.cat([z0, z1], dim=0)                 # (2B, D)

        # --- bilinear similarity matrix ---
        W = self.similarity.weight.squeeze(0)           # (D, D)
        logits = (z @ W) @ z.T                          # (2B, 2B)

        # --- mask self‑similarities ---
        logits.fill_diagonal_(float('-inf'))

        # --- scale by temperature ---
        logits.div_(self.temperature)

        # --- build targets ---
        B = out0.size(0)
        targets = torch.arange(B, device=out0.device)
        targets = torch.cat([targets + B, targets], dim=0)  # (2B,)

        return F.cross_entropy(logits, targets)
