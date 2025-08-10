import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentGraphLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for GraphCL-style contrastive learning.

    This computes the SimCLR/GraphCL objective over two corresponding batches of
    projected embeddings (z_i, z_j), using in-batch negatives and cosine similarity.

    Args:
        temperature (float): Temperature scaling factor (> 1e-8). Default: 0.1
        normalize (bool): If True, L2-normalizes embeddings before similarity. Default: True
    """

    def __init__(self, temperature: float = 0.1, normalize: bool = True, **kwargs):
        super().__init__()
        self.temperature = float(temperature)
        self.normalize = bool(normalize)
        self.eps = 1e-8
        if abs(self.temperature) < self.eps:
            raise ValueError(f"Illegal temperature: abs({self.temperature}) < {self.eps}")

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent between two batches of projected embeddings.

        Args:
            z_i (torch.Tensor): Embeddings from view 1 with shape (B, D).
            z_j (torch.Tensor): Embeddings from view 2 with shape (B, D).

        Returns:
            torch.Tensor: Scalar loss.
        """
        if z_i.dim() != 2 or z_j.dim() != 2 or z_i.size(0) != z_j.size(0):
            raise ValueError(
                f"Expected (B, D) for both views with same B. Got {tuple(z_i.shape)} and {tuple(z_j.shape)}."
            )

        B = z_i.size(0)
        device = z_i.device

        # Cosine similarity via L2-normalized embeddings (GraphCL/SimCLR practice).
        if self.normalize:
            z_i = F.normalize(z_i, dim=1)
            z_j = F.normalize(z_j, dim=1)

        # Stack: [z_i; z_j] -> (2B, D)
        z = torch.cat([z_i, z_j], dim=0)

        # Pairwise cosine similarities scaled by temperature -> logits (2B, 2B)
        logits = (z @ z.t()) / self.temperature

        # Mask self-similarity on the diagonal
        diag_mask = torch.eye(2 * B, dtype=torch.bool, device=device)
        logits.masked_fill_(diag_mask, float("-inf"))

        # Positive pairs are on the ±B diagonals
        # pos_logits shape -> (2B,)
        pos_logits = torch.cat([torch.diag(logits, B), torch.diag(logits, -B)], dim=0)

        # Denominator: logsumexp over all (2B - 1) others for each row (self already -inf)
        denom_logsumexp = torch.logsumexp(logits, dim=1)

        # NT-Xent: -log( exp(pos) / sum(exp(others)) ) == -(pos - logsumexp)
        loss = -(pos_logits - denom_logsumexp).mean()
        return loss
