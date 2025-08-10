import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from typing import Tuple, List

from MK_SSL.audio.models.utils.base_masking import InverseBlockMasking
from MK_SSL.audio.models.modules.backbones import ViTAudioEncoder
from MK_SSL.audio.models.modules.decoders import CNNAudioDecoder
from MK_SSL.audio.models.modules.feature_extractors import SpectrogramPatchEmbedder
from MK_SSL.audio.models.modules.transformations.eat_transform import LogMelSpectrogramTransform
from MK_SSL.audio.models.utils.registry import register_method

from MK_SSL.audio.models.modules.losses.ufo_loss import UFO


class EAT(nn.Module):
    """Efficient Audio Transformer (student–teacher, inverse-block masking).

    The model **does not compute the loss**.  
    It returns the tensors needed by the trainer so the loss can be
    evaluated there, mirroring the COLA training style.

    Args:
        embed_dim: Patch/transformer embedding dimension.
        mask_ratio: Fraction of blocks to mask.
        block_size: Size (time, freq) of each inverse mask block.
        lambda_u: Weighting for utterance-level term inside UFO (passed later).
        ema_tau: Exponential-moving-average coefficient for the teacher.
        num_clones: Number of masked clones per input.
        sample_rate: Input waveform sample-rate.
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
        **_,
    ) -> None:
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
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self._init_teacher()

    def _init_teacher(self) -> None:
        for ps, pt in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            pt.data.copy_(ps.data)
            pt.requires_grad = False

    @torch.no_grad()
    def update_teacher(self) -> None:
        for ps, pt in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            pt.data.mul_(self.ema_tau).add_(ps.data * (1.0 - self.ema_tau))

    def forward(
        self, wav: torch.Tensor
    ) -> Tuple[
        List[torch.Tensor],  # decoded blocks
        List[torch.Tensor],  # target blocks
        List[torch.Tensor],  # student CLS
        torch.Tensor,        # teacher avg (B,P,E)
    ]:
        """
        Args:
            wav: Tensor shape ``(B, 1, T)``.

        Returns:
            decoded_list: ``len = num_clones``; each tensor ``(B, E, h, w)``
            target_list:  same shapes as ``decoded_list``
            cls_list:     list of student CLS tokens, ``(B, E)``
            teacher_avg:  average of teacher layers, ``(B, P, E)``
        """
        logmel = self.logmel_transform(wav)
        if torch.isnan(logmel).any():
            raise ValueError("NaN in logmel_transform output")

        if (logmel.abs() > 1e4).any():
            raise ValueError("logmel values out of expected range")

        patches, (F_g, T_g) = self.feature_extractor(logmel)

        T, F = T_g, F_g
        B, _, E = patches.shape

        with torch.no_grad():
            teacher_layers = self.teacher_encoder(patches)
            teacher_avg = torch.stack(teacher_layers).mean(dim=0)
            if torch.isnan(teacher_avg).any():
                raise ValueError("NaN in teacher encoder output")


        bh, bw = self.block_size
        seq_len = T * F
        h, w = T // bh, F // bw

        decoded_list: List[torch.Tensor] = []
        target_list: List[torch.Tensor] = []
        cls_list: List[torch.Tensor] = []

        for _ in range(self.num_clones):
            mask_flat = InverseBlockMasking((T, F), self.mask_ratio, self.block_size)().view(seq_len)
            vis_flat = mask_flat
            msk_flat = ~mask_flat

            x_vis = patches[:, vis_flat]

            cls = self.cls_token.expand(B, -1, -1)
            student_inp = torch.cat([cls, x_vis], dim=1)
            student_out = self.student_encoder(student_inp)
            if torch.isnan(student_out).any():
                raise ValueError("NaN in student encoder output")

            student_cls = student_out[:, 0]
            student_tok = student_out[:, 1:]

            full = torch.empty(B, seq_len, E, device=patches.device, dtype=patches.dtype)
            full[:, vis_flat] = student_tok.to(full.dtype)
            full[:, msk_flat] = self.mask_token.to(full.dtype)

            full_2d = full.view(B, T, F, E).permute(0, 3, 1, 2)
            student_blk = Fnn.avg_pool2d(full_2d, kernel_size=self.block_size, stride=self.block_size)

            with torch.no_grad():
                teacher_2d = teacher_avg.view(B, T, F, E).permute(0, 3, 1, 2)
                tgt_blk = Fnn.avg_pool2d(teacher_2d, kernel_size=self.block_size, stride=self.block_size)

            decoded_list.append(self.decoder(student_blk))
            target_list.append(tgt_blk)
            cls_list.append(student_cls)

        return decoded_list, target_list, cls_list, teacher_avg


register_method(
    name="eat",
    model_cls=EAT,
    loss=UFO,
    transformation=LogMelSpectrogramTransform,
    default_params={},
    logs=lambda model, loss: (
        "\n"
        "---------------- EAT Configuration ----------------\n"
        f"{'Input Type':<32}: Raw waveform (B, 1, T)\n"
        f"{'Log-mel Transform':<32}: {model.logmel_transform.__class__.__name__}\n"
        f"{'Patch Embedder':<32}: {model.feature_extractor.__class__.__name__}\n"
        f"{'Student Encoder':<32}: {model.student_encoder.__class__.__name__}\n"
        f"{'Teacher Encoder':<32}: {model.teacher_encoder.__class__.__name__}\n"
        f"{'Decoder':<32}: {model.decoder.__class__.__name__}\n"
        f"{'Embedding Dimension':<32}: {model.embed_dim}\n"
        f"{'Mask Ratio':<32}: {model.mask_ratio}\n"
        f"{'Block Size':<32}: {model.block_size}\n"
        f"{'EMA Tau':<32}: {model.ema_tau}\n"
        f"{'Clones per Sample':<32}: {model.num_clones}\n"
        f"{'Masking Strategy':<32}: InverseBlockMasking\n"
        f"{'Loss':<32}: {UFO.__name__}\n"
    )
)
