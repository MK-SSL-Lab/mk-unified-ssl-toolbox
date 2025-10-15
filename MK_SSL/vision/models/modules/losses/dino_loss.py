import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """DINO Loss: Cross-view knowledge distillation for self-supervised learning.

    Uses cross-entropy between the sharpened teacher output and the student
    output, encouraging invariance across views while leveraging a dynamic center.

    Reference:
        - Paper: https://arxiv.org/abs/2104.14294

    Args:
        projection_dim (int): Dimension of the projection output.
        temp_student (float): Temperature for softmax on student outputs.
        temp_teacher (float): Temperature for sharpening teacher outputs.
        center_momentum (float): EMA decay factor for updating teacher center.
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        projection_dim: int,
        temp_student: float,
        temp_teacher: float,
        center_momentum: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.projection_dim = projection_dim
        self.temp_student = temp_student
        self.temp_teacher = temp_teacher
        self.center_momentum = center_momentum

        # Dynamic center used to stabilize teacher predictions
        self.register_buffer("center", torch.zeros(1, projection_dim))

    def forward(
        self,
        student_outputs: list[torch.Tensor],
        teacher_outputs: list[torch.Tensor]
    ) -> torch.Tensor:
        """Computes DINO loss across non-matching augmented view pairs.

        Args:
            student_outputs (List[Tensor]): Outputs from student encoder (one per view).
            teacher_outputs (List[Tensor]): Outputs from teacher encoder (only global views).

        Returns:
            torch.Tensor: Averaged loss over view pairs.
        """
        total_loss = 0.0
        loss_count = 0

        for teacher_idx, t_out in enumerate(teacher_outputs):
            for student_idx, s_out in enumerate(student_outputs):
                if student_idx == teacher_idx:
                    continue  # Skip same-view comparisons
                total_loss += self._compute_cross_entropy(t_out, s_out)
                loss_count += 1

        avg_loss = total_loss / loss_count

        # Update teacher center using only teacher global outputs
        self._update_center(teacher_outputs)

        return avg_loss

    def _compute_cross_entropy(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor
    ) -> torch.Tensor:
        """Computes soft cross-entropy between teacher and student predictions."""
        teacher_probs = F.softmax(
            (teacher_output.detach() - self.center) / self.temp_teacher, dim=1
        )
        student_logits = student_output / self.temp_student
        loss = -(teacher_probs * F.log_softmax(student_logits, dim=1)).sum(dim=1).mean()
        return loss

    @torch.no_grad()
    def _update_center(self, teacher_outputs: list[torch.Tensor]):
        """Updates the running average center using teacher's global outputs."""
        concatenated = torch.cat(teacher_outputs, dim=0)
        batch_center = concatenated.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + (1 - self.center_momentum) * batch_center
