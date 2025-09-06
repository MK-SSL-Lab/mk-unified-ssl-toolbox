import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

class COLABackbone(nn.Module):
    """
    Backbone for downstream use of a pre-trained COLA model.

    Workflow (aligned with COLA):
        wave -> (internal COLA backbone: log-mel + EfficientNet-B0 + global pooling)
             -> COLAProjectionHead -> 512-D embedding

    This module applies *no* masking beyond what COLA already supports internally.
    It simply forwards to the exact same backbone + projection head that were used
    for pretraining, so representations stay consistent.

    Args:
        pretrained_cola (nn.Module): An instance of your COLA class exposing:
            - backbone(x, lengths=None) -> feature tensor
            - projection_head(feature)  -> embedding tensor (B, projection_dim)
        normalize (bool): If True, L2-normalize embeddings.

    Inputs:
        waveforms: (B, 1, T) raw mono audio (padded to a common T).
        lengths: Optional (B,) true audio lengths in **samples** before padding.

    Returns:
        embeddings: (B, E)   # E == projection_dim (e.g., 512)
        final_lengths: (B,)  # returned in **samples** (same units as input `lengths`)
    """

    def __init__(self, pretrained_cola: nn.Module, normalize: bool = False):
        super().__init__()

        needed = ["backbone", "projection_head"]
        for n in needed:
            if not hasattr(pretrained_cola, n):
                raise AttributeError(
                    f"`pretrained_cola` lacks `{n}`; expected an instance of your COLA class."
                )

        self.backbone = pretrained_cola.backbone          # EfficientNetAudioEncoder (with COLA-internal path)
        self.projection_head = pretrained_cola.projection_head  # COLAProjectionHead
        # Keep these for reference; helpful if downstream needs dims.
        self.feature_size = getattr(pretrained_cola, "feature_size", None)
        self.projection_dim = getattr(pretrained_cola, "projection_dim", None)

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
        """
        See class docstring for shapes.
        """
        if waveforms.dim() != 3 or waveforms.size(1) != 1:
            raise ValueError(f"Expected input shape (B, 1, T), got {tuple(waveforms.shape)}")

        # Forward through the exact same COLA backbone & head.
        # Note: COLA.backbone already accepts `lengths` for masking pooled stats.
        feats = self.backbone(waveforms, lengths=lengths)   # (B, feature_size) after global pooling
        if torch.isnan(feats).any():
            raise ValueError("NaN encountered in COLA backbone output")

        emb = self.projection_head(feats)                   # (B, projection_dim)
        if torch.isnan(emb).any():
            raise ValueError("NaN encountered in COLA projection head output")

        emb = self._maybe_l2(emb)

        # Return lengths in the same unit as provided: **samples**.
        # If not provided, assume all samples are valid (the padded length T).
        if lengths is None:
            final_lengths = torch.full(
                (waveforms.size(0),),
                waveforms.size(-1),
                dtype=torch.long,
                device=waveforms.device,
            )
        else:
            final_lengths = lengths.to(device=waveforms.device, dtype=torch.long)

        return emb, final_lengths
