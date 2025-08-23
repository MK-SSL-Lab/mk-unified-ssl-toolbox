import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


class Wav2Vec2Backbone(nn.Module):
    """
    Backbone for downstream tasks using a pretrained Wav2Vec2 model.

    Returns time-sequence features and their per-item lengths (token counts),
    suitable for CTC EvaluateNet.
    """

    def __init__(self, pretrained_model: nn.Module):
        super().__init__()
        self.feature_extractor = pretrained_model.feature_extractor
        self.feature_proj = pretrained_model.feature_proj
        self.encoder = pretrained_model.encoder

        # Optional: freeze
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(
        self, waveforms: Tensor, lengths: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            waveforms: (B, 1, T) padded waveforms.
            lengths:   (B,) raw lengths in samples (pre-pad). Optional.

        Returns:
            feats:       (B, T_out, D) contextualized frame features for CTC.
            t_lengths:   (B,) valid lengths in **time tokens** (<= T_out).
        """
        if waveforms.dim() != 3 or waveforms.size(1) != 1:
            raise ValueError(f"Expected input shape (B, 1, T), got {tuple(waveforms.shape)}")

        # Conv stack returns downsampled sequence + per-item **token** lengths
        z, t_lengths = self.feature_extractor(waveforms, lengths)   # z: (B, T_cnn, C)

        # Projection keeps time axis
        z = self.feature_proj(z)                                    # (B, T_cnn, D)

        # Transformer keeps time axis & masking uses t_lengths
        context = self.encoder(z, t_lengths)                        # (B, T_cnn, D)

        # Safety: ensure lengths are long dtype on correct device
        if t_lengths is None:
            t_lengths = torch.full(
                (context.size(0),), context.size(1),
                dtype=torch.long, device=context.device
            )
        else:
            t_lengths = t_lengths.to(device=context.device, dtype=torch.long)
            # Sanity clamps
            T_out = context.size(1)
            t_lengths.clamp_(min=1, max=T_out)

        return context, t_lengths
