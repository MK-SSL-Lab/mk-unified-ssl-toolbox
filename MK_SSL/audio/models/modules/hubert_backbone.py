import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from MK_SSL.audio.models.hubert import HuBERT


class HuBERTBackbone(nn.Module):
    """
    Backbone model for downstream tasks using a pretrained HuBERT model.

    This class applies the feature extractor, projection, normalization, 
    dropout, and transformer encoder, returning mean-pooled representations.

    Args:
        model (nn.Module): The pretrained HuBERT model (pretext phase).
    """

    def __init__(self, pretrained_model: HuBERT):
        super().__init__()
        self.feature_extractor = pretrained_model.feature_extractor
        self.feature_projection = pretrained_model.feature_projection
        self.norm = pretrained_model.post_extract_proj_norm
        self.dropout = pretrained_model.post_extract_proj_dropout
        self.encoder = pretrained_model.encoder

        # Optional: freeze weights (up to user)
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(
        self, waveforms: Tensor, lengths: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            waveforms (Tensor): Input waveform tensor of shape (B, T).
            lengths (Optional[Tensor]): Valid lengths before padding (not used here).

        Returns:
            Tensor: Contextualized feature representations (B, C)
        """
        x = self.feature_extractor(waveforms)[0]  # discard lengths if returned
        x = self.feature_projection(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.encoder(x)
        return x.mean(dim=1)  # mean pooling → (B, embed_dim)
