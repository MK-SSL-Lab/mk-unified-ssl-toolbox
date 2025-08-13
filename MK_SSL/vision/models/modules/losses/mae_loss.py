import torch
import torch.nn as nn

class MAELoss(nn.Module):
    def __init__(self, normalize_target: bool = True):
        super().__init__()
        self.normalize_target = normalize_target

    def forward(self, pred, target, mask):
        mask = mask.to(pred.dtype)

        if self.normalize_target:
            mean = target.float().mean(dim=-1, keepdim=True)
            std = target.float().std(dim=-1, keepdim=True).clamp_min(1e-4)
            target = ((target.float() - mean) / std).to(pred.dtype)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                     # (B, N)
        denom = mask.sum().clamp_min(1e-6)
        return (loss * mask).sum() / denom
