import torch
import torch.nn as nn
from torch import Tensor
import torchaudio.transforms as T

from typing import Optional, Tuple, List


class PositionalConvEmbedding(nn.Module):
    """
    Convolutional positional embedding used in wav2vec 2.0.
    
    Args:
        embed_dim (int): Embedding dimension.
        kernel_size (int): Convolution kernel size.
        groups (int): Number of convolution groups.
    """

    def __init__(self, embed_dim: int, kernel_size: int, groups: int):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        # Input: (B, T, C) → (B, C, T) for conv
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.activation(x)
        return x.transpose(1, 2)  # Back to (B, T, C)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder stack for wav2vec 2.0.

    Args:
        in_features (int): Input feature dimension from the feature extractor.
        embed_dim (int): Embedding dimension.
        num_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        ff_interm_features (int): Dimension of the feedforward network.
        dropout_input (float): Dropout after input projection.
        attention_dropout (float): Dropout in multi-head attention.
        ff_dropout (float): Dropout in feedforward layers.
        final_dropout (float): Dropout after each transformer block.
        layer_norm_first (bool): Whether to use Pre-LN instead of Post-LN.
        layer_drop (float): Probability of dropping a transformer layer.
        pos_conv_kernel (int): Kernel size for positional convolution.
        pos_conv_groups (int): Groups for positional convolution.
    """

    def __init__(
        self,
        in_features: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ff_interm_features: int,
        dropout_input: float,
        attention_dropout: float,
        ff_dropout: float,
        final_dropout: float,
        layer_norm_first: bool,
        layer_drop: float,
        pos_conv_kernel: int,
        pos_conv_groups: int,
    ):
        super().__init__()

        self.embed = nn.Linear(in_features, embed_dim)
        self.dropout = nn.Dropout(dropout_input)

        self.positional_encoding = PositionalConvEmbedding(
            embed_dim=embed_dim,
            kernel_size=pos_conv_kernel,
            groups=pos_conv_groups,
        )

        self.transformer_layers = nn.ModuleList()
        self.layer_drop = layer_drop
        self.layer_norm_first = layer_norm_first

        for _ in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_interm_features,
                dropout=final_dropout,
                activation="gelu",
                batch_first=True,
            )
            if layer_norm_first:
                layer.norm1 = nn.LayerNorm(embed_dim)
                layer.norm2 = nn.LayerNorm(embed_dim)
            self.transformer_layers.append(layer)

        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through the full transformer stack.

        Args:
            x (Tensor): Input tensor of shape (B, T, C).
            lengths (Optional[Tensor]): Valid lengths for padding mask.

        Returns:
            Tensor: Output tensor of shape (B, T, C).
        """
        x = self.embed(x)
        x = self.dropout(x)
        x = x + self.positional_encoding(x)

        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
        else:
            mask = None

        for layer in self.transformer_layers:
            if self.training and torch.rand(1).item() < self.layer_drop:
                continue
            x = layer(x, src_key_padding_mask=mask)

        x = self.final_layer_norm(x)
        return x

    def extract_features(
        self,
        x: Tensor,
        lengths: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> List[Tensor]:
        """
        Extract intermediate features from each transformer layer.

        Args:
            x (Tensor): Input of shape (B, T, C).
            lengths (Optional[Tensor]): Padding mask lengths.
            num_layers (Optional[int]): Number of layers to run (if early exit).

        Returns:
            List[Tensor]: Outputs of each transformer layer.
        """
        x = self.embed(x)
        x = self.dropout(x)
        x = x + self.positional_encoding(x)

        features = []
        max_len = x.size(1)

        if lengths is not None:
            mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
        else:
            mask = None

        for i, layer in enumerate(self.transformer_layers):
            if self.training and torch.rand(1).item() < self.layer_drop:
                continue
            x = layer(x, src_key_padding_mask=mask)
            features.append(x)
            if num_layers is not None and i + 1 >= num_layers:
                break

        return features



