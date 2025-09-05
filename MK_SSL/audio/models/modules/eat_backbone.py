import torch
import torch.nn as nn
from typing import Optional, Tuple, List

class EATBackbone(nn.Module):
    """
    EAT backbone for downstream CTC-style evaluation.

    Input:
        waveforms: (B, 1, T)   -- padded batch
        lengths:   (B,)        -- *unpadded* raw audio lengths in samples (optional)

    Output:
        feats:      (B, T_out, D)  -- time-major features for CTC (T_out == T_g)
        t_lengths:  (B,)           -- *unpadded* valid lengths in time tokens (<= T_out)

    Notes
    -----
    - Time lengths are computed per item from the raw sample lengths using a single
      exact integer ceil against the batch max, which avoids rounding drift and
      off-by-one undercounts that can trigger `output_lengths < label_lengths`.
    - If `lengths` is None, we assume no padding (i.e., every item uses the full T_out).
    """

    def __init__(self, pretrained_model: nn.Module):
        super().__init__()
        # Pull the exact components from your EAT model
        self.logmel_transform = pretrained_model.logmel_transform
        self.feature_extractor = pretrained_model.feature_extractor
        self.teacher_encoder  = pretrained_model.teacher_encoder  # returns list[L] of (B, P, E)

        # Cache embedding dim if present (optional)
        self.embed_dim = getattr(pretrained_model, "embed_dim", None)

        # Teacher is used in no-grad mode
        for p in self.teacher_encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _teacher_avg(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Run teacher encoder and average across layers.
        Args:
            patches: (B, P, E) flattened patches
        Returns:
            (B, P, E) averaged representations
        """
        layers: List[torch.Tensor] = self.teacher_encoder(patches)  # list[L] of (B,P,E)
        return torch.stack(layers, dim=0).mean(dim=0)

    def forward(
        self,
        waveforms: torch.Tensor,              # (B,1,T)
        lengths: Optional[torch.Tensor] = None  # (B,) in raw samples, unpadded
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if waveforms.dim() != 3 or waveforms.size(1) != 1:
            raise ValueError(f"Expected waveforms of shape (B,1,T), got {tuple(waveforms.shape)}")

        B, _, T_max = waveforms.shape

        # 1) Log-mel spectrogram: (B, 1, F_spec, T_spec)
        logmel = self.logmel_transform(waveforms)

        # 2) Patchify (time×freq grid) → flattened patches (B, P, E), with grid size (F_g, T_g)
        patches, (F_g, T_g) = self.feature_extractor(logmel)  # P = F_g * T_g
        P, E = patches.size(1), patches.size(2)
        assert P == F_g * T_g, "Patch grid mismatch: P != F_g * T_g"

        # 3) Teacher encoder average (no grad) to get contextual representations
        with torch.no_grad():
            reps = self._teacher_avg(patches)  # (B, P, E)

        # 4) Convert back to (time, freq) and pool frequency to get a pure time sequence
        reps_2d = reps.view(B, T_g, F_g, E)     # (B, T_g, F_g, E)
        feats   = reps_2d.mean(dim=2)           # (B, T_g, E)  <-- CTC time axis

        # 5) Compute *unpadded* per-item lengths in **time tokens**
        if lengths is None:
            # Assume no padding: each item uses the entire time grid
            t_lengths = torch.full((B,), T_g, dtype=torch.long, device=feats.device)
        else:
            if lengths.dim() != 1 or lengths.size(0) != B:
                raise ValueError(f"lengths must be 1-D of size B={B}, got {tuple(lengths.shape)}")

            # We know: along the batch, T_g scales ~linearly with raw samples up to T_max.
            # For item i with raw length L_i (samples), its token length is:
            #   ceil( L_i * T_g / T_max )
            # Do it with integer math to avoid floating rounding drift:
            num = lengths.to(torch.long) * T_g                     # (B,)
            den = torch.as_tensor(int(T_max), device=lengths.device)
            # integer ceil: ceil(num/den) == floor((num + den - 1)/den)
            t_lengths = torch.div(num + den - 1, den, rounding_mode='floor')
            t_lengths = t_lengths.clamp_(min=1, max=T_g).to(device=feats.device)

        return feats, t_lengths
