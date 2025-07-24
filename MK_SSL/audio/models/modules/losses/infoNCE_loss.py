import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss with bilinear similarity.

    Maximises agreement between paired embeddings while contrasting them
    against negatives in the batch.

    Args:
        temperature (float, optional): Logit scaling factor. Must be non‑zero.
            Defaults to 0.2.
        input_dim (int, optional): Dimensionality of each embedding.
            Defaults to 512.

    Inputs:
        out0 (Tensor): Embeddings from view 1 of shape ``(B, D)``.
        out1 (Tensor): Embeddings from view 2 of shape ``(B, D)``.

    Returns:
        Tensor: Scalar loss value.

    Raises:
        ValueError: If ``temperature`` is zero.
    """


    def __init__(self, temperature: float = 0.2, input_dim: int = 512):
        super().__init__()
        if abs(temperature) < 1e-8:
            raise ValueError("Temperature must be non‑zero.")
        self.temperature = temperature
        # single‑output bilinear: weight shape (1, D, D)
        self.similarity = nn.Bilinear(input_dim, input_dim, 1, bias=False)

    def forward(self, out0: torch.Tensor, out1: torch.Tensor) -> torch.Tensor:
        # Normalise for numerical stability
        z0, z1 = F.normalize(out0, dim=1), F.normalize(out1, dim=1)
        z = torch.cat([z0, z1], dim=0)                    # (2B, D)

        # === vectorised bilinear similarity ===
        W = self.similarity.weight.squeeze(0)             # (D, D)
        logits = (z @ W) @ z.T                            # (2B, 2B)
        logits.div_(self.temperature)

        # Positive‑pair indices
        B = out0.size(0)
        targets = torch.arange(B, device=out0.device)
        targets = torch.cat([targets + B, targets], dim=0)  # (2B,)

        loss = F.cross_entropy(logits, targets)
        return loss
