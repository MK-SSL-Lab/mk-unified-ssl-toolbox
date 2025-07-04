import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss function for contrastive learning.
    
    Args:
        temperature (float): Temperature scaling factor. Must be > 1e-8.
    """

    def __init__(self, temperature: float = 0.1, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.eps = 1e-8
        if abs(self.temperature) < self.eps:
            raise ValueError(
                f"Illegal temperature: abs({self.temperature}) < {self.eps}"
            )

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute the NT-Xent loss between two batches of projected embeddings.

        Args:
            z_i (torch.Tensor): Embeddings from view 1 of shape (B, D).
            z_j (torch.Tensor): Embeddings from view 2 of shape (B, D).

        Returns:
            torch.Tensor: Scalar contrastive loss.
        """
        device = z_i.device
        batch_size = z_i.size(0)

        # Normalize the embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate and compute similarity matrix
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
        sim_matrix = torch.exp(torch.mm(z, z.T) / self.temperature)  # (2B, 2B)

        # Mask out self-similarity
        mask = (
            torch.ones_like(sim_matrix, dtype=torch.bool, device=device)
            ^ torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        )
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # Compute positive pair similarities
        pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / self.temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)  # (2B,)

        # Final NT-Xent loss
        loss = -torch.log(pos_sim / sim_matrix.sum(dim=1)).mean()
        return loss
