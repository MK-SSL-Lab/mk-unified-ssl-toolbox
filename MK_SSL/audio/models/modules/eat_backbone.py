# =========================
# EAT Backbone (Wav2Vec2-parity)
# Returns: (features, token_lengths) just like your Wav2Vec2Backbone
# =========================
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, List


class EATBackbone(nn.Module):
    """
    Backbone for downstream CTC evaluation using a pretrained **EAT** model.

    Mirrors Wav2Vec2Backbone behavior:
      - Input:  waveforms (B,1,T) and optional raw lengths (B,)
      - Output: contextualized frame features (B, T_out, D) and token lengths (B,)
    """

    def __init__(self, pretrained_model: nn.Module):
        super().__init__()
        # Use teacher path (EMA-smoothed) for stable features
        self.logmel_transform = pretrained_model.logmel_transform
        self.feature_extractor = pretrained_model.feature_extractor
        self.teacher_encoder = pretrained_model.teacher_encoder  # outputs all layers

    @torch.no_grad()
    def _teacher_avg(self, patches: Tensor) -> Tensor:
        """
        Run teacher encoder and average across layers.
        Args:
            patches: (B, P, E)
        Returns:
            (B, P, E)
        """
        layers: List[Tensor] = self.teacher_encoder(patches)   # list[L] of (B,P,E)
        return torch.stack(layers, dim=0).mean(dim=0)          # (B,P,E)

    def forward(
        self, waveforms: Tensor, lengths: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            waveforms: (B, 1, T) padded
            lengths:   (B,) raw lengths in samples (pre-pad). Optional.

        Returns:
            feats:      (B, T_out, D) contextualized frame features for CTC
            t_lengths:  (B,) valid lengths in **time tokens** (<= T_out)
        """
        if waveforms.dim() != 3 or waveforms.size(1) != 1:
            raise ValueError(f"Expected (B,1,T), got {tuple(waveforms.shape)}")

        # 1) Log-mel spectrogram: (B, 1, F_spec, T_spec)
        logmel = self.logmel_transform(waveforms)
        B, _, _, T_spec = logmel.shape

        # 2) Patchify → (B, P, E) and grid (F_g, T_g)
        patches, (F_g, T_g) = self.feature_extractor(logmel)  # P = F_g * T_g
        P, E = patches.size(1), patches.size(2)
        assert P == F_g * T_g, "Patch grid mismatch"

        # 3) Teacher encoder aggregation (no grad for stability)
        with torch.no_grad():
            reps = self._teacher_avg(patches)                 # (B, P, E)

        # 4) Reshape to time×freq, then pool frequency → time sequence
        reps_2d = reps.view(B, T_g, F_g, E)                   # (B, T_g, F_g, E)
        feats = reps_2d.mean(dim=2)                           # (B, T_g, E)  <-- CTC time axis

        # 5) Compute per-example output lengths in time tokens
        if lengths is None:
            t_lengths = torch.full((B,), T_g, dtype=torch.long, device=feats.device)
        else:
            max_samples = waveforms.size(-1)
            valid_spec = (lengths.to(torch.float32) * (T_spec / float(max_samples))).floor()
            valid_spec.clamp_(min=1)
            frames_per_token = T_spec / float(T_g)
            t_lengths = (valid_spec / frames_per_token).floor().to(torch.long)
            t_lengths = t_lengths.clamp_(min=1, max=T_g).to(device=feats.device)

        return feats, t_lengths
