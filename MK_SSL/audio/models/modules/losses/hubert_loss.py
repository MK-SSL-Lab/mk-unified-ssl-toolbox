import torch
import torch.nn as nn
import torch.nn.functional as F


class HuBERTLoss(nn.Module):
    """
    Loss for HuBERT pretraining. Computes cross-entropy between predicted logits and
    precomputed pseudo-labels (hidden units) over masked time steps.

    Args:
        reduction (str): Reduction method. Default is "mean".
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits (Tensor): Prediction logits of shape (B, T, C)
            targets (Tensor): Pseudo-labels (cluster ids) of shape (B, T)
            mask_indices (Tensor): Boolean mask of shape (B, T) where True indicates a masked position

        Returns:
            loss (Tensor): Cross-entropy loss over masked positions
        """
        if logits.shape[:2] != targets.shape:
            raise ValueError("Shape mismatch: logits and targets must match on (B, T)")

        B, T, C = logits.shape
        logits = logits[mask_indices]   # (N_masked, C)
        targets = targets[mask_indices] # (N_masked,)

        loss = F.cross_entropy(logits, targets, reduction=self.reduction)
        return loss
