import torch
import torch.nn as nn
from torch import Tensor
from MK_SSL.multimodal.models import AudioCLIP


class AudioCLIPAudioBackbone(nn.Module):
    """
    Backbone model for downstream tasks using the audio encoder from a pretrained AudioCLIP model.

    This class wraps only the audio encoder and skips any projection or text components.

    Args:
        pretrained_model (AudioCLIP): The pretrained AudioCLIP model (pretext phase).
    """

    def __init__(self, pretrained_model: AudioCLIP):
        super().__init__()
        self.audio_encoder = pretrained_model.audio_encoder

        # Optional: freeze weights (up to user)
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, waveforms: Tensor) -> Tensor:
        """
        Args:
            waveforms (Tensor): Input waveform tensor of shape (B, T).

        Returns:
            Tensor: Audio embeddings (B, C), already normalized.
        """
        return self.audio_encoder(waveforms)
