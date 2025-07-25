import torch
import torch.nn as nn

class MAELoss(nn.Module):
    """
    Mean squared error computed only on masked patches.
    """
    def __init__(self, normalize_target: bool = False):
        super().__init__()
        self.normalize_target = normalize_target

    def forward(self, pred, target, mask):
        if self.normalize_target:
            mean = target.mean(dim=-1, keepdim=True)
            std = target.std(dim=-1, keepdim=True) + 1e-6
            target = (target - mean) / std
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        return (loss * mask).sum() / mask.sum()
