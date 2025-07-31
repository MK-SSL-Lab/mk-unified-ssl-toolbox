import torch
import torch.nn as nn
from typing import Optional, Tuple

from MK_SSL.audio.models.utils.registry import register_method
from MK_SSL.audio.models.modules.losses.ufo_loss import UFO
from MK_SSL.audio.models.modules.transformations.base_masking import InverseBlockMasking
from MK_SSL.audio.models.modules.backbones import ViTAudioEncoder
from MK_SSL.audio.models.modules.heads import CNNAudioDecoder


@register_method("eat")
class EAT(nn.Module):
    """
    EAT: Efficient Audio Transformer

    Implements student-teacher architecture with inverse block masking and
    utterance-frame objective (UFO) loss for SSL on spectrogram inputs.

    Args:
        embed_dim (int): Embedding size.
        mask_ratio (float): Percentage of patches to mask.
        block_size (Tuple[int, int]): Size of inverse mask blocks.
        lambda_u (float): Weight for utterance-level loss.
        ema_tau (float): EMA decay rate for teacher update.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        mask_ratio: float = 0.8,
        block_size: Tuple[int, int] = (5, 5),
        lambda_u: float = 1.0,
        ema_tau: float = 0.996,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.block_size = block_size
        self.ema_tau = ema_tau

        self.student_encoder = ViTAudioEncoder(embed_dim=embed_dim, output_all_layers=False)
        self.teacher_encoder = ViTAudioEncoder(embed_dim=embed_dim, output_all_layers=True)
        self.decoder = CNNAudioDecoder(input_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.loss_fn = UFO(lambda_u)

        self._init_teacher()

    def _init_teacher(self):
        for param_s, param_t in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        for param_s, param_t in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            param_t.data.mul_(self.ema_tau).add_(param_s.data * (1.0 - self.ema_tau))

    def forward(
        self,
        x: torch.Tensor,
        patch_grid: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input spectrogram (B, P, E).
            patch_grid (Tuple[int, int]): Grid shape (T, F) before flattening.

        Returns:
            torch.Tensor: Scalar loss.
        """
        B, P, E = x.shape
        T, F = patch_grid

        mask = InverseBlockMasking((T, F), self.mask_ratio, self.block_size)().view(B, T * F)
        visible_mask = mask
        masked_mask = ~mask

        x_vis = x[visible_mask].view(B, -1, E)
        cls = self.cls_token.expand(B, -1, -1)
        student_input = torch.cat([cls, x_vis], dim=1)
        student_out = self.student_encoder(student_input)

        student_cls = student_out[:, 0]
        student_tokens = student_out[:, 1:]
        student_tokens_2d = student_tokens.view(B, -1, F // self.block_size[1], E).permute(0, 3, 1, 2)
        decoded = self.decoder(student_tokens_2d)

        with torch.no_grad():
            teacher_out = self.teacher_encoder(x)
            teacher_all = torch.stack(teacher_out, dim=0).mean(dim=0)

        target_masked = teacher_all[masked_mask].view_as(decoded)

        loss = self.loss_fn(decoded, target_masked, student_cls, teacher_all)
        return loss
