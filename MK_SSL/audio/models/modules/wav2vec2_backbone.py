import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class Wav2Vec2Backbone(nn.Module):
    """
    Backbone model for downstream tasks using a pretrained Wav2Vec2 model.
    
    This class disables masking and quantization, and only returns
    contextualized speech representations suitable for classification or regression.

    Args:
        pretrained_model (nn.Module): The pretrained Wav2Vec2 model (pretext phase).
    """

    def __init__(self, pretrained_model: nn.Module):
        super().__init__()
        self.feature_extractor = pretrained_model.feature_extractor
        self.encoder = pretrained_model.encoder
        self.feature_proj = pretrained_model.feature_proj

        # Optional: freeze weights (up to user)
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(
        self, waveforms: Tensor, lengths: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            waveforms (Tensor): Input waveform tensor of shape (B, 1, T).
            lengths (Optional[Tensor]): Valid lengths before padding.

        Returns:
            Tensor: Contextualized feature representations (B, T', C)
        """
        if waveforms.dim() != 3 or waveforms.size(1) != 1:
            raise ValueError(f"Expected input shape (B, 1, T), but got {tuple(waveforms.shape)}")
        
        z, lengths = self.feature_extractor(waveforms, lengths)
        z = self.feature_proj(z)
        context = self.encoder(z, lengths)
        return context
