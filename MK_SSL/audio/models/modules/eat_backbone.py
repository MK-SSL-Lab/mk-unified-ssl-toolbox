import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

class EATBackbone(nn.Module):
    """
    Backbone for downstream tasks using a pretrained **EAT** model.
    Returns frame-level reps (B, Pmax, E) *and* per-item valid lengths (B,).
    """

    def __init__(self, pretrained_model: nn.Module):
        super().__init__()
        # Re-use only the parts needed for pure feature extraction
        self.logmel_transform = pretrained_model.logmel_transform
        self.feature_extractor = pretrained_model.feature_extractor
        self.teacher_encoder = pretrained_model.teacher_encoder  # EMA-smoothed weights

        # Try to locate the Conv2d used for patchifying
        self._proj_conv: Optional[nn.Conv2d] = getattr(self.feature_extractor, "proj", None)
        if not isinstance(self._proj_conv, nn.Conv2d):
            self._proj_conv = None  # we'll fall back gracefully if absent

    @staticmethod
    def _conv_out_len(L_in: torch.Tensor, k: int, s: int, p: int, d: int) -> torch.Tensor:
        # PyTorch Conv formula: floor((L + 2p - d*(k-1) - 1)/s + 1)
        return torch.floor((L_in + 2*p - d*(k-1) - 1) / s + 1).to(torch.long).clamp(min=0)

    def _infer_mel_params(self):
        """
        Best-effort introspection of mel transform params.
        Falls back to None if not found; we then use a ratio fallback.
        """
        obj = self.logmel_transform
        def pick(names, default=None):
            for n in names:
                if hasattr(obj, n):
                    return getattr(obj, n)
            return default

        # Common names
        hop = pick(["hop_length", "hop", "hop_size", "hopsize"])
        n_fft = pick(["n_fft", "fft_size", "nfft"])
        win_length = pick(["win_length", "win_size"])
        center = pick(["center"], True)

        if n_fft is None and win_length is not None:
            n_fft = win_length
        # If at least hop & n_fft exist, we can compute exactly
        if hop is not None and n_fft is not None:
            hop = int(hop)
            n_fft = int(n_fft)
            pad = (n_fft // 2) if bool(center) else 0
            return hop, n_fft, pad
        return None

    def forward(self, waveforms: Tensor, lengths: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            waveforms: (B, 1, T_pad)
            lengths:   (B,) valid lengths in **samples** (unpadded). Required for per-item lengths.

        Returns:
            reps:        (B, P_max, E) frame-level embeddings
            valid_p:     (B,) number of valid frames per item after patching (used as CTC input_lengths)
        """
        if waveforms.dim() != 3 or waveforms.size(1) != 1:
            raise ValueError(f"Expected input shape (B, 1, T), but got {tuple(waveforms.shape)}")

        B = waveforms.size(0)
        device = waveforms.device

        # 1) Log-mel
        logmel = self.logmel_transform(waveforms)        # (B, 1, F_in, T_mel_full)
        F_in = logmel.size(-2)
        T_mel_full = logmel.size(-1)

        # 2) Patchify (get grid dims)
        patches, grid = self.feature_extractor(logmel)   # patches: (B, P_max, E), grid: (F_g, T_g)
        F_g, T_g = grid
        P_max = patches.size(1)
        E = patches.size(2)

        # 3) Teacher encoder (returns list of (B, P_max, E)); average layers
        need_grad = any(p.requires_grad for p in self.teacher_encoder.parameters())
        with torch.set_grad_enabled(need_grad):
            layer_outputs = self.teacher_encoder(patches)  # List[L] of (B, P_max, E)
        reps = torch.stack(layer_outputs).mean(dim=0)      # (B, P_max, E)

        # ----- Compute per-item valid lengths on the time grid -----
        # Get Conv2d patch params if available
        if self._proj_conv is not None:
            kF, kT = self._proj_conv.kernel_size
            sF, sT = self._proj_conv.stride
            # padding/dilation can be int or tuple
            pF, pT = (self._proj_conv.padding if isinstance(self._proj_conv.padding, tuple)
                      else (self._proj_conv.padding, self._proj_conv.padding))
            dF, dT = (self._proj_conv.dilation if isinstance(self._proj_conv.dilation, tuple)
                      else (self._proj_conv.dilation, self._proj_conv.dilation))
        else:
            # Sensible fallbacks for a ViT-style patch embedder:
            # kernel == patch size, stride == patch size, no dilation/padding
            # (Only used if we cannot introspect conv; safe for many embedders.)
            kF = getattr(self.feature_extractor, "patch_size_f", 16)
            kT = getattr(self.feature_extractor, "patch_size_t", 16)
            sF = getattr(self.feature_extractor, "stride_f", kF)
            sT = getattr(self.feature_extractor, "stride_t", kT)
            pF = getattr(self.feature_extractor, "padding_f", 0)
            pT = getattr(self.feature_extractor, "padding_t", 0)
            dF = getattr(self.feature_extractor, "dilation_f", 1)
            dT = getattr(self.feature_extractor, "dilation_t", 1)

        # Frequency grid is independent of utterance length
        F_g_exact = self._conv_out_len(torch.tensor([F_in], device=device, dtype=torch.float32),
                                       kF, sF, pF, dF)[0].item()
        assert F_g_exact == F_g, "Feature-extractor F-grid mismatch; check embedder params."

        # Time grid per item
        if lengths is None:
            # If lengths not provided, assume everything is valid (fully padded)
            T_g_i = torch.full((B,), T_g, dtype=torch.long, device=device)
        else:
            lengths = lengths.to(device=device, dtype=torch.long)

            # Try exact mel frame math; fallback to ratio
            mel_params = self._infer_mel_params()
            if mel_params is not None:
                hop, n_fft, pad = mel_params
                # frames = floor((L + 2*pad - n_fft) / hop) + 1, clamped to [0, T_mel_full]
                L_eff = lengths.to(torch.long) + 2*pad - n_fft
                T_mel_i = torch.floor_divide(torch.clamp(L_eff, min=0), hop) + 1
                T_mel_i = T_mel_i.clamp(min=0, max=T_mel_full)
            else:
                # Ratio fallback (very close; off-by-one at most)
                T_pad_samples = waveforms.size(-1)
                T_mel_i = torch.floor(lengths.to(torch.float32) * (T_mel_full / float(T_pad_samples))).to(torch.long)
                T_mel_i = T_mel_i.clamp(min=0, max=T_mel_full)

            # Map mel frames → patch time grid with the same conv formula
            T_g_i = self._conv_out_len(T_mel_i.to(torch.float32), kT, sT, pT, dT)
            T_g_i = T_g_i.clamp(min=0, max=T_g)

        # Valid patches per item = F_g * T_g_i
        valid_p = (T_g_i * F_g_exact).to(torch.long)

        # Safety: lengths must never exceed P_max, and P_max should equal F_g * T_g
        assert P_max == F_g * T_g, f"P_max ({P_max}) != F_g*T_g ({F_g}*{T_g})"
        valid_p = valid_p.clamp(min=0, max=P_max)

        return reps, valid_p
