import torch
import torch.nn as nn
import torch.nn.functional as F

class HuBERTLoss(nn.Module):
    """
    Cross-entropy loss for HuBERT pretraining.
    Expects logits and targets for masked positions only.
    """

    def __init__(self, reduction: str = "mean", **kwargs):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (Tensor): Prediction logits of shape (N_masked, C).
            targets (Tensor): Pseudo-labels of shape (N_masked,).

        Returns:
            loss (Tensor): Cross-entropy loss over masked positions.
        """
        if logits.ndim != 2:
            raise ValueError(f"Expected logits of shape (N_masked, C), got {logits.shape}.")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError(f"Expected targets of shape ({logits.shape[0]},), got {targets.shape}.")
        loss = F.cross_entropy(logits, targets, reduction=self.reduction)
        return loss
