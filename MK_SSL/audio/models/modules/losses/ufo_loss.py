import torch
import torch.nn as nn
from typing import Optional


class UFO(nn.Module):
    """
    Utterance-Frame Objective (UFO) for EAT model.

    This loss combines:
    - Frame-level loss: MSE on masked features.
    - Utterance-level loss: MSE between CLS and mean target.

    Args:
        lambda_u (float): Weight for utterance-level loss.
    """

    def __init__(self, lambda_u: float = 1.0):
        super().__init__()
        self.lambda_u = lambda_u
        self.mse = nn.MSELoss()

    def forward(
        self,
        student_decoded: torch.Tensor,
        teacher_masked_target: torch.Tensor,
        student_cls_token: torch.Tensor,
        teacher_all_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_decoded (torch.Tensor): Predicted masked features, shape (B, E, H, W).
            teacher_masked_target (torch.Tensor): Target features for masked positions, shape (B, E, H, W).
            student_cls_token (torch.Tensor): CLS token from student encoder, shape (B, E).
            teacher_all_targets (torch.Tensor): Full teacher output, shape (B, P, E).

        Returns:
            torch.Tensor: UFO loss (Lf + λ·Lu).
        """
        loss_f = self.mse(student_decoded, teacher_masked_target)

        teacher_mean = teacher_all_targets.mean(dim=1)  # (B, E)
        loss_u = self.mse(student_cls_token, teacher_mean)

        return loss_f + self.lambda_u * loss_u
