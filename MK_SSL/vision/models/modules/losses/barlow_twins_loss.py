import torch
import torch.nn as nn
import torch.nn.functional as F


class BarlowTwinsLoss(nn.Module):
    """Barlow Twins Loss Function for Self-Supervised Learning.

    Encourages embeddings of two augmented views to be both similar
    (diagonal of cross-correlation matrix close to 1) and decorrelated
    (off-diagonal values close to 0).

    Reference:
        - Paper: https://arxiv.org/abs/2103.03230

    Args:
        lambda_param (float): Scaling factor for the off-diagonal penalty. Defaults to 5e-3.
    """

    def __init__(self, lambda_param: float = 5e-3, **kwargs):
        super().__init__()
        self.lambda_param = lambda_param

    def _off_diagonal_elements(self, matrix: torch.Tensor) -> torch.Tensor:
        """Returns flattened off-diagonal elements of a square matrix."""
        n, m = matrix.shape
        assert n == m, "Input matrix must be square."
        return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, proj_1: torch.Tensor, proj_2: torch.Tensor) -> torch.Tensor:
        """Computes the Barlow Twins loss.

        Args:
            proj_1 (Tensor): Batch of projected embeddings from view 1. Shape (B, D).
            proj_2 (Tensor): Batch of projected embeddings from view 2. Shape (B, D).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Normalize the batch
        proj_1 = F.normalize(proj_1, dim=-1)
        proj_2 = F.normalize(proj_2, dim=-1)

        # Standardize each feature (z-score)
        norm_1 = (proj_1 - proj_1.mean(dim=0)) / proj_1.std(dim=0)
        norm_2 = (proj_2 - proj_2.mean(dim=0)) / proj_2.std(dim=0)

        # Compute cross-correlation matrix
        batch_size = proj_1.size(0)
        cross_corr = torch.matmul(norm_1.T, norm_2) / batch_size

        # Diagonal loss: Encourage correlation on matching dimensions → 1
        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()

        # Off-diagonal loss: Encourage decorrelation on non-matching dims → 0
        off_diag = self._off_diagonal_elements(cross_corr).pow_(2).sum()

        # Total loss
        total_loss = on_diag + self.lambda_param * off_diag
        return total_loss
