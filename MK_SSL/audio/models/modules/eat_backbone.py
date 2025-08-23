import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

# Optional: if you want static type-checking against the concrete class
# from MK_SSL.audio.models.eat import EAT  # noqa: F401


class EATBackbone(nn.Module):
    """
    Backbone model for downstream tasks using a pretrained **EAT** model.

    The backbone disables masking/decoding and returns a fixed-length,
    utterance-level representation suitable for classification or regression.

    Args:
        pretrained_model (nn.Module): A fully-trained :class:`EAT` model.
    """

    def __init__(self, pretrained_model: nn.Module):
        super().__init__()
        # Re-use only the parts needed for pure feature extraction
        self.logmel_transform = pretrained_model.logmel_transform
        self.feature_extractor = pretrained_model.feature_extractor
        self.teacher_encoder = pretrained_model.teacher_encoder  # EMA-smoothed weights

        # Optional: freeze everything (uncomment if you don’t want finetuning)
        # for p in self.parameters():
        #     p.requires_grad = False
    def forward(self, waveforms: Tensor, lengths: Optional[Tensor] = None):
        """
        Args:
            waveforms (Tensor): (B, 1, T) padded.
            lengths   (Optional[Tensor]): valid raw lengths (in samples) before padding.

        Returns:
            feats_time (Tensor): (B, T_g, E) time-major features (freq-pooled).
            t_lengths  (Tensor): (B,) per-item valid lengths in time tokens (<= T_g).
        """
        if waveforms.dim() != 3 or waveforms.size(1) != 1:
            raise ValueError(f"Expected input shape (B, 1, T), but got {tuple(waveforms.shape)}")

        # 1) Log-mel spectrogram: (B, 1, F_spec, T_spec)
        logmel = self.logmel_transform(waveforms)

        # 2) Patchify -> (B, P, E) and the grid sizes (F_g, T_g)
        patches, (F_g, T_g) = self.feature_extractor(logmel)   # P = F_g * T_g
        B, P, E = patches.shape
        assert P == F_g * T_g, "Feature extractor grid does not match flattened token count."

        # 3) Teacher encoder outputs: list of (B, P, E) -> average over layers -> (B, P, E)
        need_grad = any(p.requires_grad for p in self.teacher_encoder.parameters())
        with torch.set_grad_enabled(need_grad):
            layer_outputs = self.teacher_encoder(patches)      # List[L] of (B, P, E)
        reps = torch.stack(layer_outputs).mean(dim=0)          # (B, P, E)

        # 4) Reshape to time × freq, then pool over frequency to get a **time sequence**
        reps_2d = reps.view(B, T_g, F_g, E)                    # (B, T_g, F_g, E)
        feats_time = reps_2d.mean(dim=2)                       # (B, T_g, E)  <-- CTC time axis

        # 5) Compute per-example output lengths in **time tokens**
        #    We avoid hard-coding hop/stride. Instead:
        #    - T_spec = logmel.size(-1) is the spectrogram time steps of the padded batch.
        #    - T_g is the time tokens after patching.
        #    - For each item, scale by raw length vs padded max length.
        T_spec = logmel.size(-1)
        if lengths is None:
            t_lengths = torch.full((B,), T_g, dtype=torch.long, device=waveforms.device)
        else:
            max_L = waveforms.size(-1)
            # valid spectrogram frames per item (proportional to its raw length)
            valid_spec = torch.floor(lengths.to(torch.float32) * (T_spec / float(max_L)))
            valid_spec.clamp_(min=1)

            # map spectrogram frames -> token frames by proportional scaling
            frames_per_token = T_spec / float(T_g)
            t_lengths = torch.floor(valid_spec / frames_per_token).to(torch.long)
            t_lengths.clamp_(min=1, max=T_g)

        return feats_time, t_lengths

