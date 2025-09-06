import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

class COLABackbone(nn.Module):
    """
    Backbone for downstream use of a pre-trained COLA model (encoder-only).

    Workflow (aligned with COLA pretext, minus projection):
        wave -> COLA.backbone (log-mel + EfficientNet-B0 + global pooling)
             -> features (B, feature_size)

    Args:
        pretrained_cola (nn.Module): COLA instance exposing:
            - backbone(x, lengths=None) -> (B, feature_size)
        normalize (bool): If True, L2-normalize returned features.

    Inputs:
        waveforms: (B, 1, T) raw mono audio (padded).
        lengths:   Optional (B,) true audio lengths in **samples**.

    Returns:
        features:      (B, feature_size)  # encoder/backbone output
        final_lengths: (B,) in **samples** (same units as input `lengths`)
    """

    def __init__(self, pretrained_cola: nn.Module, normalize: bool = False):
        super().__init__()

        if not hasattr(pretrained_cola, "backbone"):
            raise AttributeError("`pretrained_cola` must have attribute `backbone`.")

        self.backbone = pretrained_cola.backbone
        self.feature_size = getattr(pretrained_cola, "feature_size", None)

        self.normalize = normalize

    def _maybe_l2(self, x: Tensor) -> Tensor:
        if not self.normalize:
            return x
        return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

    @torch.no_grad()
    def forward(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if waveforms.dim() != 3 or waveforms.size(1) != 1:
            raise ValueError(f"Expected input shape (B, 1, T), got {tuple(waveforms.shape)}")

        # Encoder/backbone features (already globally pooled inside COLA.backbone)
        feats = self.backbone(waveforms, lengths=lengths)   # (B, feature_size)
        if torch.isnan(feats).any():
            raise ValueError("NaN encountered in COLA backbone output")

        feats = self._maybe_l2(feats)

        # Return lengths in the same unit as provided: **samples**
        if lengths is None:
            final_lengths = torch.full(
                (waveforms.size(0),),
                waveforms.size(-1),
                dtype=torch.long,
                device=waveforms.device,
            )
        else:
            final_lengths = lengths.to(device=waveforms.device, dtype=torch.long)

        return feats, final_lengths
