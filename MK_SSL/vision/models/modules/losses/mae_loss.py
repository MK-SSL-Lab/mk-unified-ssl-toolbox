import torch
import torch.nn as nn

class MAELoss(nn.Module):
    """
    Loss function for MAE: Mean Squared Error (MSE) computed only on masked patches.
    Optionally applies per-patch normalization to improve representation quality.
    """
    def __init__(self, normalize_target: bool = False):
        super().__init__()
        self.normalize_target = normalize_target
        self.loss_fn = nn.MSELoss()

    def forward(self, pred, target, mask):
        """
        Args:
            pred (Tensor): predicted patches, shape (B, N, patch_dim)
            target (Tensor): ground truth patches, shape (B, N, patch_dim)
            mask (Tensor): mask indicating which patches were masked, shape (B, N)
        Returns:
            loss (Tensor): scalar loss value
        """
        if self.normalize_target:
            mean = target.mean(dim=-1, keepdim=True)
            std = target.std(dim=-1, keepdim=True) + 1e-6
            target = (target - mean) / std

        # Compute MSE only on masked patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, N)
        loss = (loss * mask).sum() / mask.sum()
        return loss