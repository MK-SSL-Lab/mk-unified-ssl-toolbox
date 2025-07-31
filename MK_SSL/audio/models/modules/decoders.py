import torch
import torch.nn as nn
from typing import Tuple


class CNNAudioDecoder(nn.Module):
    """
    Lightweight 2D CNN decoder for reconstructing masked spectrogram patches.

    Used in EAT's student network to predict frame-level representations from
    masked positions.

    Args:
        input_dim (int): Dimension of input features.
        hidden_dim (int): Dimension of hidden layers.
        output_dim (int): Dimension of output features (typically same as input_dim).
        num_layers (int): Number of convolutional layers.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_layers: int = 6,
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else hidden_dim
            out_ch = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            if i != num_layers - 1:
                layers.append(nn.GroupNorm(1, out_ch))
                layers.append(nn.GELU())

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Decoded tensor of shape (B, C, H, W).
        """
        return self.decoder(x)
