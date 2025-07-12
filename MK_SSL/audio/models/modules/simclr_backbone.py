import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from MK_SSL.audio.models import SimCLRSpeech


class SimCLRBackbone(nn.Module):
    """
    Backbone model for downstream tasks using a pretrained SimCLR model.

    This class applies FBANK feature extraction, input projection, and
    the encoder backbone, returning mean-pooled contextual representations.

    Args:
        model (nn.Module): The pretrained SimCLR model (pretext phase).
    """

    def __init__(self, pretrained_model: SimCLRSpeech):
        super().__init__()
        self.fbank = pretrained_model.fbank
        self.input_proj = pretrained_model.input_proj
        self.backbone = pretrained_model.backbone

        # Optional: freeze weights (up to user)
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(
        self,
        waveforms: Tensor,
    ) -> Tensor:
        """
        Args:
            waveforms (Tensor): Input waveform tensor of shape (B, T).
            lengths (Optional[Tensor]): Valid lengths before padding (not used here).

        Returns:
            Tensor: Contextualized feature representations (B, C)
        """
        x = self.fbank(waveforms)  # (B, T, 80)
        x = self.input_proj(x)  # (B, T, embed_dim)
        x = self.backbone(x)  # (B, T, embed_dim)
        return x.mean(dim=1)  # mean pooling → (B, embed_dim)
