import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Optional


class ConvFeatureExtractor(nn.Module):
    """
    Convolutional feature extractor for wav2vec 2.0.

    Args:
        variant (str): Normalization type. Either "group_norm" or "layer_norm".
        conv_layers (List[Tuple[int, int, int]]): Configuration of conv layers (out_channels, kernel_size, stride).
        conv_bias (bool): Whether to use bias in convolutional layers.
    """

    def __init__(
        self,
        variant: str,
        conv_layers: List[Tuple[int, int, int]],
        conv_bias: bool = False,
    ):
        super().__init__()

        assert variant in {"group_norm", "layer_norm"}, f"Invalid variant: {variant}"

        layers = []
        in_channels = 1  # raw waveform has 1 channel

        for out_channels, kernel_size, stride in conv_layers:
            conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=conv_bias,
            )

            if variant == "group_norm":
                norm = nn.GroupNorm(1, out_channels)
                layers.extend([conv, norm, nn.GELU()])
            else:  # layer_norm
                norm = nn.LayerNorm(out_channels)
                layers.extend([
                    conv,
                    nn.Sequential(
                        nn.Transpose(1, 2), norm, nn.Transpose(1, 2)
                    ),
                    nn.GELU()
                ])

            in_channels = out_channels

        self.extractor = nn.Sequential(*layers)

    def forward(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            waveforms (Tensor): Input tensor of shape (batch, time).
            lengths (Optional[Tensor]): Valid lengths of each sample before padding.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: Feature tensor of shape (batch, time, channels) and updated lengths.
        """
        x = waveforms.unsqueeze(1)  # (B, 1, T)
        x = self.extractor(x)       # (B, C, T')

        if lengths is not None:
            for module in self.extractor:
                if isinstance(module, nn.Conv1d):
                    lengths = ((lengths - module.kernel_size[0]) // module.stride[0]) + 1

        return x.transpose(1, 2), lengths  # (B, T', C)
