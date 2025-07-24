import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from MK_SSL.audio.models.modules.transformations import COLAAudioTransform


class COLABackbone(nn.Module):
    """
    Backbone model for downstream tasks using a pretrained COLA model.

    This class wraps the pretrained backbone and returns global feature embeddings,
    skipping the projection head used during contrastive pretraining.

    Args:
        pretrained_model (nn.Module): The pretrained COLA model (pretext phase).
    """

    def __init__(self, pretrained_model: nn.Module):
        super().__init__()
        self.backbone = pretrained_model.backbone  # EfficientNetAudioEncoder
        self.transformation = COLAAudioTransform()
        # Optional: freeze weights (up to user)
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(
        self, waveforms: Tensor, lengths: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            waveforms (Tensor): Input waveform tensor of shape (B,1, T).
            lengths (Optional[Tensor]): Valid lengths before padding (not used here).

        Returns:
            Tensor: Global audio embeddings (B, C)
        """
        view0, _ = self.transformation(waveforms)
        return self.backbone(view0)  # Includes global pooling internally
