import torch
import torch.nn as nn
from typing import Optional, Tuple

from MK_SSL.multimodal.models.modules.backbones import Wav2ClipEncoder
from MK_SSL.multimodal.models.modules.feature_extractors import ResNetFeatureExtractor


class Wav2Clip(nn.Module):
    """
    Wav2CLIP model: encodes audio and image features for contrastive learning.

    Args:
        audio_encoder (nn.Module): Module that encodes waveform input. If None, defaults to Wav2ClipEncoder.
        image_encoder (nn.Module): Pretrained CLIP image encoder (must be provided by user).
        projection_dim (int): Projection dimension to map both modalities.
        freeze_image_encoder (bool): Whether to freeze the image encoder during training.
    """

    def __init__(
        self,
        audio_encoder: Optional[nn.Module] = None,
        image_encoder: Optional[nn.Module] = None,
        projection_dim: int = 512,
        freeze_image_encoder: bool = True,
    ):
        super().__init__()

        if image_encoder is None:
            raise ValueError("You must provide a pretrained (frozen) CLIP image encoder.")

        self.audio_encoder = audio_encoder if audio_encoder is not None else Wav2ClipEncoder(
            backbone=ResNetFeatureExtractor.get_default_resnet_audio(),
            projection_dim=projection_dim,
            input_dim=512
        )

        self.image_encoder = image_encoder

        if freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        audio_waveform: torch.Tensor,
        image_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Wav2Clip model.

        Args:
            audio_waveform (torch.Tensor): Input waveform of shape (B, T).
            image_input (torch.Tensor): Input image or embedding tensor of shape (B, ...) depending on encoder.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of (audio_embed, image_embed)
        """
        audio_embed = self.audio_encoder(audio_waveform)
        image_embed = self.image_encoder(image_input)
        return audio_embed, image_embed
