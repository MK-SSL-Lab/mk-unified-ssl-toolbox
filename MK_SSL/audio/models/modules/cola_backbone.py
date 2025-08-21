import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class COLABackbone(nn.Module):
    """
    Backbone model for downstream tasks using a pretrained COLA model.

    This class wraps the pretrained backbone and can return either:
      - Global embeddings (B, C) [default]
      - Frame-level embeddings (B, T, C) for CTC
    
    Args:
        pretrained_model (nn.Module): The pretrained COLA model (pretext phase).
        return_sequence (bool): If True, outputs frame-level features (B, T, C).
                                If False, outputs global embeddings (B, C).
    """

    def __init__(self, pretrained_model: nn.Module, return_sequence: bool = False):
        super().__init__()
        self.backbone = pretrained_model.backbone   # EfficientNetAudioEncoder
        self.encoder = self.backbone.encoder        # conv stack inside EfficientNet
        self.mel = self.backbone.mel
        self.log1p = self.backbone.log1p
        self.hop_len = self.backbone.hop_len
        self.return_sequence = return_sequence

    def forward(
        self, waveforms: Tensor, lengths: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            waveforms (Tensor): Input waveform tensor of shape (B,1,T).
            lengths (Optional[Tensor]): Valid lengths before padding.

        Returns:
            Tensor:
              - (B, C) if return_sequence == False
              - (B, T, C) if return_sequence == True
        """
        if not self.return_sequence:
            # Default COLA behavior → pooled embeddings
            return self.backbone(waveforms, lengths)

        # --- Frame-level mode (CTC) ---
        mel = self.mel(waveforms.squeeze(1))     # (B, n_mels, time)
        mel = self.log1p(mel).unsqueeze(1)       # (B, 1, n_mels, time)

        feats = self.encoder(mel)                # (B, C, H, W)
        feats = feats.mean(dim=2)                # collapse freq dim → (B, C, T)
        feats = feats.transpose(1, 2)            # (B, T, C)

        if lengths is not None:
            # compute valid number of frames per example
            num_frames = torch.div(lengths, self.hop_len, rounding_mode="floor") + 1
            max_frames = feats.size(1)

            mask = torch.arange(max_frames, device=feats.device).expand(len(num_frames), max_frames)
            mask = mask < num_frames.unsqueeze(1)   # (B, T)

            # Optionally you could return mask alongside features
            return feats, mask

        return feats