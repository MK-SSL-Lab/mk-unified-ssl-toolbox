import torch
from torch import nn
import torchaudio.transforms as T



class LogMelSpectrogramTransform(nn.Module):
    """
    Converts a mono waveform (B, 1, T) to a normalized log-Mel spectrogram.

    Input shape : (B, 1,  T)   – batch, channel=1, samples  
    Output shape: (B, 1, 128, τ) – batch, channel=1, mel-bins, frames
    """

    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=128,
            window_fn=torch.hann_window,
        )
        self.db = T.AmplitudeToDB()

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav (Tensor): Input waveform of shape (B, 1, T)

        Returns:
            Tensor: Normalized log-Mel spectrogram, shape (B, 1, 128, T')
        """
        if wav.ndim != 3 or wav.size(1) != 1:
            raise ValueError(f"Expected input shape (B, 1, T); got {tuple(wav.shape)}")

        wav = wav.squeeze(1)                            # (B, T)
        mel = self.mel(wav)                             # (B, 128, T')
        mel = torch.clamp(mel, min=1e-10)               # prevent log(0)

        logmel = self.db(mel)                           # (B, 128, T')

        # normalize per-sample
        mean = logmel.mean(dim=(-2, -1), keepdim=True)
        std = logmel.std(dim=(-2, -1), keepdim=True) + 1e-5
        logmel = (logmel - mean) / std

        return logmel.unsqueeze(1)                      # (B, 1, 128, T')
