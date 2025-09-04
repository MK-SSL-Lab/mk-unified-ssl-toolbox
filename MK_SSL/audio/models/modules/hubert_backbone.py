import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class HuBERTBackbone(nn.Module):
    """
    Backbone model for downstream tasks using a pretrained HuBERT model.

    This class applies the feature extractor, projection, normalization, 
    dropout, and transformer encoder, returning mean-pooled representations.

    Args:
        pretrained_model (nn.Module): The pretrained HuBERT model (pretext phase).
    """

    def __init__(self, pretrained_model: nn.Module):
        super().__init__()
        self.feature_extractor = pretrained_model.feature_extractor
        self.feature_projection = pretrained_model.feature_projection
        self.norm = pretrained_model.post_extract_proj_norm
        self.encoder = pretrained_model.encoder
        self.projection_head = pretrained_model.projection_head
        # Optional: freeze weights (up to user)
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(
        self, waveforms: Tensor, lengths: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            waveforms (Tensor): Input waveform tensor of shape (B, 1, T).
            lengths (Optional[Tensor]): Valid lengths before padding (not used here).

        Returns:
            Tensor: Contextualized feature representations (B, C)
        """
        if waveforms.dim() != 3 or waveforms.size(1) != 1:
            raise ValueError(f"Expected input shape (B, 1, T), but got {tuple(waveforms.shape)}")
        
        x = self.feature_extractor(waveforms)[0]  # discard lengths if returned
        x = self.feature_projection(x)
        x = self.norm(x)
        x = self.encoder(x)
        x = self.projection_head(x)
        return x
