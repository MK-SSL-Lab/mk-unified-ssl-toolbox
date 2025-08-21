import torch
import torch.nn as nn
from torch import Tensor
import torchaudio.transforms as T
import torchaudio
from torchvision.models import efficientnet_b0
from typing import Optional, List


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
        self.conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding='same',
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
        pos_conv_groups: int,
        pos_conv_kernel: int,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_interm_features = ff_interm_features
        self.dropout_input = dropout_input
        self.attention_dropout = attention_dropout
        self.ff_dropout = ff_dropout
        self.final_dropout = final_dropout
        self.layer_norm_first = layer_norm_first
        self.pos_conv_kernel = pos_conv_kernel
        self.pos_conv_groups = pos_conv_groups
        self.layer_drop = layer_drop


        self.positional_encoding = PositionalConvEmbedding(
            embed_dim=self.embed_dim,
            kernel_size=self.pos_conv_kernel,
            groups=self.pos_conv_groups,
        )

        self.transformer_layers = nn.ModuleList()

        for _ in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_interm_features,
                dropout=final_dropout,
                activation="gelu",
                batch_first=True,
            )
            if self.layer_norm_first:
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
        # x = self.embed(x)
        # x = self.dropout(x)
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

    def extract_layer(
        self,
        x: Tensor,
        layer: int,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        """Return the output of a specific transformer layer."""
        feats = self.extract_features(x, lengths, num_layers=layer)
        return feats[layer - 1]


class EfficientNetAudioEncoder(nn.Module):
    """Encoder backbone for COLA (Contrastive Learning of Audio).

    Converts raw waveform to a 1280-D latent vector using EfficientNet-B0
    applied to log-mel spectrograms, with padding masked out.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_mels: int = 64,
        mel_win_len: int = 25,
        mel_hop_len: int = 10,
        f_min: int = 60,
        f_max: int = 7_800,
    ):
        super().__init__()

        # --- Time–frequency front-end ---
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=int(sample_rate * mel_win_len / 1_000),
            hop_length=int(sample_rate * mel_hop_len / 1_000),
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )
        self.log1p = torch.log1p
        self.sample_rate = sample_rate
        self.hop_len = int(sample_rate * mel_hop_len / 1_000)

        # --- EfficientNet-B0 backbone ---
        eff = efficientnet_b0(weights=None)  # torchvision ≥ 0.15
        old_conv = eff.features[0][0]
        new_conv = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
        eff.features[0][0] = new_conv
        self.encoder = eff.features

        # we will implement our own masked pooling
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(
        self,
        wave: torch.Tensor,        # (B, 1, T)
        lengths: Optional[torch.Tensor] = None  # (B,)
    ) -> torch.Tensor:
        """Forward pass with masking support."""
        mel = self.mel(wave.squeeze(1))          # (B, n_mels, time)
        mel = self.log1p(mel).unsqueeze(1)       # (B, 1, n_mels, time)
        feats = self.encoder(mel)                # (B, C, H, W)

        if lengths is not None:
            # compute number of frames before padding for each example
            num_frames = torch.div(lengths, self.hop_len, rounding_mode="floor") + 1
            max_frames = feats.size(-1)

            mask = torch.arange(max_frames, device=feats.device).expand(len(num_frames), max_frames)
            mask = mask < num_frames.unsqueeze(1)           # (B, W)
            mask = mask.unsqueeze(1).unsqueeze(2)           # (B,1,1,W)

            # set padded positions to -inf so max pooling ignores them
            feats = feats.masked_fill(~mask, float("-inf"))

            pooled = torch.amax(feats, dim=(-2, -1))        # (B, C)
        else:
            # fall back to standard pooling
            pooled = self.pool(feats).flatten(1)            # (B, C)

        return pooled

    


class ViTAudioEncoder(nn.Module):
    """
    ViT-style Transformer encoder for audio spectrograms.

    This encoder maps audio patch embeddings to contextual representations using
    stacked Transformer layers. It supports optional output of all hidden states
    for teacher-student distillation.

    Args:
        embed_dim (int): Dimension of the input patch embeddings.
        num_layers (int): Number of Transformer encoder layers.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Expansion factor for MLP hidden dimension.
        dropout (float): Dropout rate.
        output_all_layers (bool): Whether to return outputs from all layers.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        output_all_layers: bool = False,
    ):
        super().__init__()
        self.output_all_layers = output_all_layers

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input of shape (B, P, E), where P is number of patches.

        Returns:
            torch.Tensor: If output_all_layers=False, output from final layer (B, P, E).
                          If output_all_layers=True, list of outputs from all layers.
        """
        if self.output_all_layers:
            out_all = []
            for mod in self.encoder.layers:
                x = mod(x)
                out_all.append(x)
            return out_all
        return self.encoder(x)
