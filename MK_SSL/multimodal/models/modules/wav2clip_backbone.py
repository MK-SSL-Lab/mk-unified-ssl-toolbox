import torch
import torch.nn as nn
from torch import Tensor


class Wav2CLIPAudioBackbone(nn.Module):
    """
    Backbone model for downstream tasks using the audio encoder from a pretrained Wav2CLIP model.

    This class removes the projection head (if present) and returns fixed-length audio embeddings.

    Args:
        pretrained_model (nn.Module): The pretrained Wav2CLIP model (pretext phase).
    """

    def __init__(self, pretrained_model: nn.Module):
        super().__init__()
        self.encoder = pretrained_model.audio_encoder

        # Remove projection head if it exists
        if hasattr(self.encoder, "projection"):
            self.encoder.projection = nn.Identity()

        # Optional: freeze weights (up to user)
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, waveforms: Tensor) -> Tensor:
        """
        Args:
            waveforms (Tensor): Input waveform tensor of shape (B, T).

        Returns:
            Tensor: Audio embeddings (B, 512)
        """
        return self.encoder(waveforms)
