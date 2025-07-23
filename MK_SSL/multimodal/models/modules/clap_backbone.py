import torch
import torch.nn as nn
from torch import Tensor
from transformers import BatchEncoding

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
        self.mel_spectrogram_transform = pretrained_model.mel_spectrogram_transform


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
        waveforms = self.mel_spectrogram_transform(waveforms)
        return self.audio_encoder(waveforms)  


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
        self, inputs, 
    ) -> Tensor:
        """
        Args:
            inputs (Tensor): Tuple of (input_ids, attention_mask) from tokenized text.
            lengths (Optional[Tensor]): Not used.

        Returns:
            Tensor: Global text embeddings (B, C)
        """

        if isinstance(inputs, BatchEncoding):
            text_input = dict(text_input)
        elif isinstance(inputs, dict):
            pass
        elif isinstance(inputs, torch.Tensor):
            text_input = {"input_ids": text_input, "attention_mask": (text_input != 0).long()}
        else:
            raise TypeError(f"Unsupported text_input type: {type(text_input)}")
        
        input_ids, attention_mask = inputs
        return self.text_encoder(**text_input)  
