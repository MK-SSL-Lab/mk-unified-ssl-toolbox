import torch
import torch.nn as nn
import torch.nn.functional as F


class SwAVLoss(nn.Module):
    """SwAV Loss: Cross-view consistency using online clustering assignments.

    Reference:
        - Paper: https://arxiv.org/abs/2006.09882
        - Code: https://github.com/facebookresearch/swav

    Args:
        num_crops (int): Number of total views (global + local) per sample.
        temperature (float): Temperature scaling applied to prototype outputs. Default is 0.1.
        **kwargs: Additional arguments for compatibility.
    """

    def __init__(self, num_crops: int, temperature: float = 0.1, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.num_crops = num_crops

    def _cross_entropy(self, target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Computes the cross-entropy between a soft target and logits."""
        return torch.mean(torch.sum(target * F.log_softmax(logits, dim=1), dim=1))

    def forward(
        self,
        assignments: tuple[torch.Tensor, torch.Tensor, list],
        codes: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Computes SwAV loss between code assignments and soft clustering targets.

        Args:
            assignments (Tuple):
                - c1: logits from view 1
                - c2: logits from view 2
                - c_c: list of logits from remaining crops (local views)
            codes (Tuple):
                - q1: sinkhorn assignments for view 1
                - q2: sinkhorn assignments for view 2

        Returns:
            torch.Tensor: Final averaged SwAV loss.
        """
        loss = 0.0

        c1, c2, local_logits = assignments
        q1, q2 = codes

        p1, p2 = c1 / self.temperature, c2 / self.temperature

        # Global-to-global cross-view loss
        loss += self._cross_entropy(q1, p2) / (self.num_crops - 1)
        loss += self._cross_entropy(q2, p1) / (self.num_crops - 1)

        # Global-to-local cross-view loss
        for crop_logits in local_logits:
            p_local = crop_logits / self.temperature
            loss += self._cross_entropy(q1, p_local) / (self.num_crops - 1)
            loss += self._cross_entropy(q2, p_local) / (self.num_crops - 1)

        return loss / 2
