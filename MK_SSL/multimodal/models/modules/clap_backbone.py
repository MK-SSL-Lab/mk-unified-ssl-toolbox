import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

class CLAPAudioBackbone(nn.Module):
    """
    Backbone model for downstream tasks using the audio encoder from a pretrained CLAP model.

    This class wraps the audio encoder (e.g., CNN14) and skips the projection head used
    during pretraining, returning fixed-length embeddings.

    Args:
        pretrained_model (nn.Module): The pretrained CLAP model (pretext phase).
    """

    def __init__(self, pretrained_model: nn.Module):
        super().__init__()
        self.audio_encoder = pretrained_model.audio_encoder  # CNN14

        # Optional: freeze weights (up to user)
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(
        self, waveforms: Tensor, 
    ) -> Tensor:
        """
        Args:
            waveforms (Tensor): Input waveform tensor of shape (B, T).
            lengths (Optional[Tensor]): Valid lengths before padding (not used here).

        Returns:
            Tensor: Global audio embeddings (B, C)
        """
        return self.audio_encoder(waveforms)  # Output: (B, 2048)


import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional


class CLAPTextBackbone(nn.Module):
    """
    Backbone model for downstream tasks using the text encoder from a pretrained CLAP model.

    This class wraps the text encoder (e.g., BERT) and skips the projection head used
    during pretraining, returning fixed-length text embeddings.

    Args:
        pretrained_model (nn.Module): The pretrained CLAP model (pretext phase).
    """

    def __init__(self, pretrained_model: nn.Module):
        super().__init__()
        self.text_encoder = pretrained_model.text_encoder  # BERTTextEncoder

        # Optional: freeze weights (up to user)
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(
        self, inputs: Tuple[Tensor, Tensor], 
    ) -> Tensor:
        """
        Args:
            inputs (Tuple[Tensor, Tensor]): Tuple of (input_ids, attention_mask) from tokenized text.
            lengths (Optional[Tensor]): Not used.

        Returns:
            Tensor: Global text embeddings (B, C)
        """
        input_ids, attention_mask = inputs
        return self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)  # Output: (B, 768)
