import torch
import torch.nn as nn
import torch.nn.functional as F


class BYOLLoss(nn.Module):
    """BYOL Loss: Symmetric similarity loss between online and target networks.

    Encourages predictions from the online network to match the projections
    from the target network using cosine similarity.

    Reference:
        - Paper: https://arxiv.org/abs/2006.07733

    Args:
        **kwargs: Extra arguments (unused, for compatibility).
    """

    def __init__(self, **kwargs):
        super().__init__()

    @staticmethod
    def similarity_loss(predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes cosine similarity loss between two vectors.

        Args:
            predict (torch.Tensor): Online prediction vector.
            target (torch.Tensor): Target projection vector.

        Returns:
            torch.Tensor: Cosine similarity loss value.
        """
        predict = F.normalize(predict, dim=-1, p=2)
        target = F.normalize(target, dim=-1, p=2)
        return 2 - 2 * (predict * target).sum(dim=-1)

    def forward(
        self,
        output_pair_1: tuple[torch.Tensor, torch.Tensor],
        output_pair_2: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Computes BYOL loss between two augmented views.

        Args:
            output_pair_1 (Tuple): Tuple (online_pred_1, target_proj_1).
            output_pair_2 (Tuple): Tuple (online_pred_2, target_proj_2).

        Returns:
            torch.Tensor: Averaged symmetric BYOL loss.
        """
        pred_1, target_1 = output_pair_1
        pred_2, target_2 = output_pair_2

        loss_a = self.similarity_loss(pred_1, target_2.detach())
        loss_b = self.similarity_loss(pred_2, target_1.detach())

        return (loss_a + loss_b).mean()
