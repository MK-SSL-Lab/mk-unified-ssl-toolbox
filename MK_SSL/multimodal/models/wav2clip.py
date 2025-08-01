import torch
import torch.nn as nn
from typing import Optional, Tuple

from MK_SSL.multimodal.models.modules.backbones import Wav2ClipAudioEncoder
from MK_SSL.multimodal.models.modules.feature_extractors import ResNetFeatureExtractor
from MK_SSL.multimodal.models.modules.backbones import CLIPImageEncoder
from MK_SSL.multimodal.models.utils import register_method
from MK_SSL.multimodal.models.modules.losses import Wav2ClipLoss


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
        device: str = 'cpu',
        **kwargs
    ):
        super().__init__()
        
        self.device = device


        if image_encoder is not None:
            self.image_encoder = image_encoder
        else:
            self.image_encoder = CLIPImageEncoder(device=self.device, model_name="ViT-B/32" )


        self.audio_encoder = audio_encoder if audio_encoder is not None else Wav2ClipAudioEncoder(
            backbone=ResNetFeatureExtractor.get_default_resnet_audio(),
            projection_dim=projection_dim,
            input_dim=512
        )
        self.wav2clip_loss = Wav2ClipLoss()


    def forward(
        self,
        audio_waveform: torch.Tensor,
        image_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        audio_waveform : torch.Tensor
            Raw audio shaped **(B, 1, T)** (preferred) or **(B, T)**.
        image_input : torch.Tensor
            Input for the image branch (shape depends on the image encoder).

        Returns
        -------
        (audio_embed, image_embed) : Tuple[torch.Tensor, torch.Tensor]
        """
        audio_embed = self.audio_encoder(audio_waveform)
        image_embed = self.image_encoder(image_input)
        return audio_embed, image_embed
    
    def criterion(self, image_embeddings: torch.Tensor, audio_embeddings: torch.Tensor) -> torch.Tensor:
        return self.wav2clip_loss(
            image_embeddings, 
            audio_embeddings
        )

register_method(
    name="wav2clip", 
    model_cls=Wav2Clip, 
    logs=lambda model: (
        "\n"
        "---------------- Wav2CLIP Configuration ----------------\n"
        f"Audio Encoder                    : {model.audio_encoder.__class__.__name__}\n"
        f"Image Encoder                    : {model.image_encoder.__class__.__name__}\n"
        f"Projection Dimension             : {model.audio_encoder.projection.output_dim if hasattr(model.audio_encoder, 'projection') else 'N/A'}\n"
        f"Image Encoder Frozen             : {'Yes' if not any(p.requires_grad for p in model.image_encoder.parameters()) else 'No'}\n"
        "Contrastive Learning             : Audio ↔ Image modality alignment\n"
        "Loss                             : Contrastive (user-defined externally)\n"
    )
)
