import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


class EATBackbone(nn.Module):
    """
    Backbone for downstream tasks using a pretrained **EAT** model.

    Produces frame-level features (time steps) suitable for CTC:
      - Extracts log-mel -> spectrogram patches (2D grid: T_g x F_g)
      - Runs the **teacher** encoder (EMA) and averages across its layers
      - Collapses frequency by averaging over F_g to get (B, T_g, E)
      - Computes per-utterance valid lengths T_out from raw audio lengths

    Returns:
        feats       : Tensor, shape (B, T_out_max, E)
        out_lengths : Tensor, shape (B,)
    """

    def __init__(self, pretrained_model: nn.Module):
        super().__init__()
        # Reuse the exact modules from the pretrained EAT
        self.logmel_transform = pretrained_model.logmel_transform
        self.feature_extractor = pretrained_model.feature_extractor  # SpectrogramPatchEmbedder
        self.teacher_encoder = pretrained_model.teacher_encoder      # ViTAudioEncoder (EMA)
        self.embed_dim = getattr(pretrained_model, "embed_dim", None)

        # Try to locate the Conv2d used to patchify (to compute output lengths exactly)
        self._patch_conv = self._find_patch_conv2d(self.feature_extractor)

        # Cache mel/STFT parameters if present
        self._mel_params = self._infer_mel_params(self.logmel_transform)

    # ---------- helpers ----------

    @staticmethod
    def _find_patch_conv2d(module: nn.Module) -> Optional[nn.Conv2d]:
        # Common name: 'proj' (as in many patch embedders). Fall back to first Conv2d found.
        if hasattr(module, "proj") and isinstance(module.proj, nn.Conv2d):
            return module.proj
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                return m
        return None  # ratio fallback will be used for out_lengths

    @staticmethod
    def _infer_mel_params(logmel) -> dict:
        # Robust extraction of STFT/mel params (with safe defaults)
        params = {}
        # Try direct attributes
        for k in ("n_fft", "hop_length", "win_length", "center"):
            if hasattr(logmel, k):
                params[k] = getattr(logmel, k)
        # Try nested melkwargs dict if present
        mk = getattr(logmel, "melkwargs", None)
        if isinstance(mk, dict):
            params.setdefault("n_fft", mk.get("n_fft"))
            params.setdefault("hop_length", mk.get("hop_length"))
            params.setdefault("win_length", mk.get("win_length"))
            params.setdefault("center", mk.get("center"))
        # Safe defaults (typical for 16 kHz)
        params.setdefault("n_fft", 400)          # ~25 ms
        params.setdefault("hop_length", 160)     # ~10 ms
        params.setdefault("win_length", params["n_fft"])
        params.setdefault("center", True)
        return params

    @staticmethod
    def _stft_frames_from_samples(L: Tensor, n_fft: int, hop: int, center: bool) -> Tensor:
        """
        Estimate number of STFT frames produced for raw length L (samples).
        Torch STFT with center=True pads by n_fft//2 on both sides.
        Formula mirrors conv length:
            out = floor((L + 2*pad - n_fft) / hop) + 1
        """
        pad = (n_fft // 2) if center else 0
        out = torch.floor((L + 2 * pad - n_fft) / hop) + 1
        return torch.clamp(out, min=0).to(torch.long)

    @staticmethod
    def _conv_time_length(
        L_in: Tensor, kernel: int, stride: int, padding: int, dilation: int
    ) -> Tensor:
        """
        1D conv length formula for time dimension mirrored from PyTorch:
            out = floor((L_in + 2*padding - dilation*(kernel-1) - 1) / stride) + 1
        """
        num = L_in + 2 * padding - dilation * (kernel - 1) - 1
        out = torch.floor(num / stride) + 1
        return torch.clamp(out, min=0).to(torch.long)

    # ---------- forward ----------

    def forward(self, waveforms: Tensor, lengths: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            waveforms : (B, 1, T)
            lengths   : (B,) valid raw lengths in samples (before padding). Optional but recommended.

        Returns:
            feats       : (B, T_out_max, E)
            out_lengths : (B,)
        """
        if waveforms.dim() != 3 or waveforms.size(1) != 1:
            raise ValueError(f"Expected input shape (B, 1, T), got {tuple(waveforms.shape)}")

        # 1) Log-mel spectrogram  -> (B, 1, F_mel, T_mel_max)
        logmel = self.logmel_transform(waveforms)

        # 2) Patchify (2D)        -> tokens (B, P, E), grid (F_g, T_g)
        patches, (F_g, T_g) = self.feature_extractor(logmel)   # P = F_g * T_g
        B, P, E = patches.shape
        assert P == F_g * T_g, "Patch grid dims do not match sequence length"

        # 3) Teacher encoder (EMA). Enable grad only if params require it.
        need_grad = any(p.requires_grad for p in self.teacher_encoder.parameters())
        with torch.set_grad_enabled(need_grad):
            layer_outputs = self.teacher_encoder(patches)       # List[L] of (B, P, E)

        # 4) Average over teacher layers, then collapse frequency grid -> (B, T_g, E)
        teacher_avg = torch.stack(layer_outputs).mean(dim=0)    # (B, P, E)
        feats_time = teacher_avg.view(B, T_g, F_g, E).mean(dim=2)  # average over F_g

        # 5) Compute per-utterance output lengths in time steps (T_out)
        if lengths is not None:
            lengths = lengths.to(waveforms.device)
            n_fft = self._mel_params["n_fft"]
            hop = self._mel_params["hop_length"]
            center = self._mel_params["center"]

            # Mel frame lengths
            mel_L = self._stft_frames_from_samples(lengths, n_fft=n_fft, hop=hop, center=center)

            # If we can read patch Conv2d, compute exact grid time length; else ratio fallback.
            if self._patch_conv is not None:
                k_t = int(self._patch_conv.kernel_size[1])
                s_t = int(self._patch_conv.stride[1])
                p_t = int(self._patch_conv.padding[1])
                d_t = int(self._patch_conv.dilation[1])
                out_lengths = self._conv_time_length(mel_L, k_t, s_t, p_t, d_t)
            else:
                # Ratio fallback: scale mel frames to grid time using the observed max
                T_mel_max = logmel.size(-1)
                ratio = float(T_g) / float(T_mel_max) if T_mel_max > 0 else 0.0
                out_lengths = torch.floor(mel_L.to(torch.float32) * ratio).to(torch.long)

            # Safety clamps
            out_lengths = torch.clamp(out_lengths, min=0, max=T_g)
        else:
            # No lengths given -> assume everything is valid in the padded batch
            out_lengths = torch.full((B,), T_g, dtype=torch.long, device=feats_time.device)

        return feats_time, out_lengths
