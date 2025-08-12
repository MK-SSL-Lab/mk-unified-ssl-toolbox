import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentGraphLoss(nn.Module):
    """
    NT-Xent loss for two batches of projected embeddings (z_i, z_j) with in-batch negatives.

    Args:
        temperature: Temperature scaling factor (> 1e-8). Default: 0.1.
        normalize: If True, L2-normalize embeddings before similarity. Default: True.
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
        Args:
            z_i: (B, D)
            z_j: (B, D)

        Returns:
            Scalar loss.
        """
        if z_i.dim() != 2 or z_j.dim() != 2 or z_i.size(0) != z_j.size(0):
            raise ValueError(
                f"Expected (B, D) for both views with same B. Got {tuple(z_i.shape)} and {tuple(z_j.shape)}."
            )

        B = z_i.size(0)
        device = z_i.device

        if self.normalize:
            z_i = F.normalize(z_i, dim=1)
            z_j = F.normalize(z_j, dim=1)

        # (2B, D)
        z = torch.cat([z_i, z_j], dim=0)

        # Cosine sim scaled by temperature → logits (2B, 2B)
        logits = (z @ z.t()) / self.temperature

        # Mask self-similarity on the diagonal
        diag_mask = torch.eye(2 * B, dtype=torch.bool, device=device)
        logits.masked_fill_(diag_mask, float("-inf"))

        # Row-wise positive indices: for row r, positive is at column (r ± B) mod 2B
        row = torch.arange(2 * B, device=device)
        pos_col = (row + B) % (2 * B)
        pos_logits = logits[row, pos_col]  # (2B,)

        # Numerically stable denominator: logsumexp with row-wise max subtraction
        row_max, _ = logits.max(dim=1, keepdim=True)                 # (2B, 1)
        lse = torch.logsumexp(logits - row_max, dim=1) + row_max.squeeze(1)  # (2B,)

        loss = -(pos_logits - lse).mean()
        return loss
