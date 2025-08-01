import torch
import torch.nn as nn
from typing import Optional, Tuple


from MK_SSL.audio.models.modules.losses.ufo_loss import UFO
from MK_SSL.audio.models.utils.base_masking import InverseBlockMasking
from MK_SSL.audio.models.modules.backbones import ViTAudioEncoder
from MK_SSL.audio.models.modules.decoders import CNNAudioDecoder
from MK_SSL.audio.models.modules.feature_extractors import SpectrogramPatchEmbedder
from MK_SSL.audio.models.modules.transformations.eat_transform import LogMelSpectrogramTransform

from MK_SSL.audio.models.utils.registry import register_method


class EAT(nn.Module):
    """
    EAT: Efficient Audio Transformer

    Student-teacher architecture with inverse block masking and Utterance-Frame Objective (UFO).

    Args:
        embed_dim (int): Embedding dimension.
        mask_ratio (float): Ratio of patches to mask.
        block_size (Tuple[int, int]): Block size for inverse masking.
        lambda_u (float): Weight for utterance loss.
        ema_tau (float): EMA decay rate for teacher.
        num_clones (int): Number of masked clones per input.
        sample_rate (int): Input waveform sample rate.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        mask_ratio: float = 0.8,
        block_size: Tuple[int, int] = (5, 5),
        lambda_u: float = 1.0,
        ema_tau: float = 0.996,
        num_clones: int = 1,
        sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.block_size = block_size
        self.ema_tau = ema_tau
        self.num_clones = num_clones

        self.logmel_transform = LogMelSpectrogramTransform(sample_rate=sample_rate)
        self.feature_extractor = SpectrogramPatchEmbedder(embed_dim=embed_dim)
        self.student_encoder = ViTAudioEncoder(embed_dim=embed_dim, output_all_layers=False)
        self.teacher_encoder = ViTAudioEncoder(embed_dim=embed_dim, output_all_layers=True)
        self.decoder = CNNAudioDecoder(input_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.loss_fn = UFO(lambda_u)

        self._init_teacher()

    def _init_teacher(self):
        for ps, pt in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            pt.data.copy_(ps.data)
            pt.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        for ps, pt in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            pt.data.mul_(self.ema_tau).add_(ps.data * (1.0 - self.ema_tau))

    def forward(
        self,
        wav: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            wav (torch.Tensor): Input waveform (B, 1, T)

        Returns:
            torch.Tensor: UFO loss
        """
        logmel = self.logmel_transform(wav)                    # (B, 1, F, T)
        patches, patch_grid = self.feature_extractor(logmel)  # (B, P, E), (T, F)

        B, P, E = patches.shape
        T, F = patch_grid

        with torch.no_grad():
            teacher_out = self.teacher_encoder(patches)
            teacher_avg = torch.stack(teacher_out).mean(dim=0)  # (B, P, E)

        total_loss = 0.0
        for _ in range(self.num_clones):
            mask = InverseBlockMasking((T, F), self.mask_ratio, self.block_size)().view(T * F)
            visible_mask = mask
            masked_mask = ~mask

            x_vis = []
            for i in range(B):
                x_vis.append(patches[i][visible_mask])
            x_vis = torch.stack(x_vis)

            cls = self.cls_token.expand(B, -1, -1)
            student_input = torch.cat([cls, x_vis], dim=1)
            student_out = self.student_encoder(student_input)
            student_cls = student_out[:, 0]
            student_tokens = student_out[:, 1:]

            h = T // self.block_size[0]
            w = F // self.block_size[1]
            student_2d = student_tokens.view(B, h, w, E).permute(0, 3, 1, 2)
            decoded = self.decoder(student_2d)

            tgt_masked = []
            for i in range(B):
                tgt_masked.append(teacher_avg[i][masked_mask])
            tgt_masked = torch.stack(tgt_masked).view_as(decoded)

            loss = self.loss_fn(decoded, tgt_masked, student_cls, teacher_avg)
            total_loss += loss

        return total_loss / self.num_clones




register_method(
    name= "eat",
    model_cls= EAT,
    loss= UFO,
    transformation= LogMelSpectrogramTransform,
    default_params={},
    logs=lambda model, loss: (
        "\n"
        "---------------- COLA Configuration ----------------\n"
        f"Input Type                       : Log-mel spectrograms (B, 1, F, T)\n"
        f"Backbone Architecture            : {model.backbone.__class__.__name__}\n"
        "Loss                             : InfoNCE Loss\n"
        "Augmentation                     : COLAAudioTransform"

    )
)