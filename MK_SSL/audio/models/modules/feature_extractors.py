import torch
import torch.nn as nn

from torch import Tensor

import torchaudio.transforms as T

from typing import List, Tuple, Optional

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)



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
        self.conv_layers = conv_layers

        for out_channels, kernel_size, stride in self.conv_layers:
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
                        Transpose(1, 2), norm, Transpose(1, 2)
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

        if waveforms.dim() != 3 or waveforms.size(1) != 1:
            raise ValueError(f"Expected input shape (B, 1, T), but got {tuple(waveforms.shape)}")
        
        x = self.extractor(waveforms)       # (B, C, T')

        if lengths is not None:
            for module in self.extractor:
                if isinstance(module, nn.Conv1d):
                    lengths = ((lengths - module.kernel_size[0]) // module.stride[0]) + 1

        return x.transpose(1, 2), lengths  # (B, T', C)


    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Computes the output lengths (T') for given raw audio input lengths (in samples).
        """
        lengths = input_lengths.clone()
        for out_channels, kernel_size, stride in self.conv_layers:
            lengths = ((lengths - kernel_size) // stride) + 1
        return lengths



class FBANKFeatureExtractor(nn.Module):
    """Compute 80-dimensional log-Mel FBANK features.

    Accepts raw waveforms shaped ``(B, 1, T)`` or ``(B, T)`` and returns
    normalized FBANKs shaped ``(B, T', 80)``.

    Args:
        sample_rate (int, optional): Audio sample rate. Defaults to ``16000``.
        n_mels (int, optional): Number of Mel filterbanks. Defaults to ``80``.
        cmvn_eps (float, optional): Numerical stability term for CMVN.
            Defaults to ``1e-5``.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        cmvn_eps: float = 1e-5,
    ):
        super().__init__()
        self.fbank = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=n_mels,
            center=True,
            power=2.0,
        )
        self.log = T.AmplitudeToDB(stype="power", top_db=80)
        self.cmvn_eps = cmvn_eps

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Extract log-Mel FBANKs.

        Args:
            waveforms (torch.Tensor):
                Raw audio of shape ``(B, 1, T)`` or ``(B, T)``.

        Returns:
            torch.Tensor: Normalized FBANK features of shape ``(B, T', 80)``.
        """
        if waveforms.dim() == 3 and waveforms.size(1) == 1:
            waveforms = waveforms.squeeze(1)
        elif waveforms.dim() != 2:
            raise ValueError(
                f"Expected (B, 1, T) or (B, T); got {tuple(waveforms.shape)}"
            )

        with torch.no_grad():
            feats = self.fbank(waveforms)      # (B, 80, T')
            feats = self.log(feats)
            feats = feats.transpose(1, 2)      # (B, T', 80)

            mean = feats.mean(dim=1, keepdim=True)
            std = feats.std(dim=1, keepdim=True) + self.cmvn_eps
            feats = (feats - mean) / std

        return feats



class MFCCFeatureExtractor(nn.Module):
    """
    Extracts MFCC features from raw waveform.
    Based on torchaudio.transforms.MFCC.

    Args:
        sample_rate (int): Audio sampling rate (default: 16000).
        n_mfcc (int): Number of MFCCs to retain (default: 39, common for HuBERT).
        n_mels (int): Number of Mel filterbanks (default: 80).
        log_mels (bool): Whether to apply log to Mel spectrogram (default: True).
        dct_type (int): Type of DCT (default: 2).
        norm (str): Normalization type for DCT (default: "ortho").
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 39,
        n_mels: int = 80,
        log_mels: bool = True,
        dct_type: int = 2,
        norm: str = "ortho",
    ):
        super().__init__()
        self.mfcc = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": 400,       # Common value for speech
                "win_length": 400,  # Common value for speech
                "hop_length": 160,  # Common value for speech
                "n_mels": n_mels,
                "center": True,
                "power": 2.0,
                "mel_scale": "htk", # Common scale
            },
            log_mels=log_mels,
            dct_type=dct_type,
            norm=norm,
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms (torch.Tensor): Input tensor of shape (B, T).

        Returns:
            torch.Tensor: MFCC features of shape (B, T', n_mfcc).
        """
        with torch.no_grad():
            # MFCC transform outputs (B, n_mfcc, T_frames)
            feats = self.mfcc(waveforms)
            # Transpose to (B, T_frames, n_mfcc) to match common feature conventions
            feats = feats.transpose(1, 2)
        return feats



class SpectrogramPatchEmbedder(nn.Module):
    """
    CNN-based patch embedding for log-mel spectrograms.

    Converts input of shape (B, 1, F, T) to (B, P, E) where
    P = (F/16) * (T/16), E = embed_dim.

    Args:
        embed_dim (int): Output embedding dimension.
    """

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=(16, 16),
            stride=(16, 16),
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, F, T).

        Returns:
            Tuple[torch.Tensor, Tuple[int, int]]:
                - Flattened patches (B, P, E)
                - Patch grid size (T//16, F//16)
        """
        x = self.proj(x)  # (B, E, F', T')
        B, E, F, T = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, F * T, E)  # (B, P, E)
        return x, (F, T)
