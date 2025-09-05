import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, List


class EATBackbone(nn.Module):
    """
    Backbone for downstream CTC evaluation using a pretrained **EAT** model.

    Matches your Wav2Vec2Backbone contract:
      - Input:  waveforms (B, 1, T), optional raw lengths (B,) in samples
      - Output: (features, token_lengths)
            features:     (B, T_out, D)   # time sequence fed to EvaluateNet's FC
            token_lengths:(B,)            # valid time tokens per item (<= T_out)

    Notes
    -----
    * Uses the **teacher encoder** outputs (averaged across layers) as stable features.
      If you unfreeze the backbone (EvaluateNet with is_linear=False), gradients
      will flow; otherwise they're disabled.
    * Token lengths are computed per item:
        1) Try **exact** math from STFT + patcher (preferred; avoids CTC warnings).
        2) Fallback: **ceil** proportional mapping from spectrogram frames to tokens,
           clamped to T_out, which is safe against undercounts.
    """

    def __init__(self, pretrained_model: nn.Module):
        super().__init__()
        # Reuse EAT components
        self.logmel_transform = pretrained_model.logmel_transform
        self.feature_extractor = pretrained_model.feature_extractor
        self.teacher_encoder = pretrained_model.teacher_encoder  # returns all layers

    # ----------------------- helpers: params & math -----------------------
    @staticmethod
    def _stft_frames(L: torch.Tensor, n_fft: int, hop_length: int, win_length: int, center: bool) -> torch.Tensor:
        """
        Compute number of STFT frames for each raw length L (in samples), matching
        torchaudio/librosa convention. Returns LongTensor (B,).
        """
        L = L.to(dtype=torch.long)
        if center:
            pad = n_fft // 2
            L_eff = L + 2 * pad
        else:
            L_eff = L
        # floor((L_eff - win_length)/hop) + 1, with min 1
        num = (L_eff - win_length).to(dtype=torch.float32)
        frames = torch.floor(num / float(hop_length)).to(dtype=torch.long) + 1
        return torch.clamp(frames, min=1)

    @staticmethod
    def _try_get_stft_params(logmel_transform) -> Tuple[int, int, int, bool]:
        # Try common attribute names; fall back to typical 25ms/10ms @16k
        n_fft      = getattr(logmel_transform, "n_fft", 400)
        hop_length = getattr(logmel_transform, "hop_length", 160)
        win_length = getattr(logmel_transform, "win_length", n_fft)
        center     = getattr(logmel_transform, "center", True)
        return int(n_fft), int(hop_length), int(win_length), bool(center)

    @staticmethod
    def _try_get_time_patch_stride(feature_extractor) -> Optional[Tuple[int, int]]:
        """
        Best-effort introspection of time patch & stride from the patch embedder.
        Returns (t_patch, t_stride) if found, else None.
        """
        # Direct attribute names (common patterns)
        for attr_p, attr_s in [
            ("patch_size_t", "stride_t"),
            ("patch_time", "stride_time"),
            ("t_patch", "t_stride"),
        ]:
            if hasattr(feature_extractor, attr_p):
                t_patch = int(getattr(feature_extractor, attr_p))
                t_stride = int(getattr(feature_extractor, attr_s, t_patch))
                return t_patch, t_stride

        # If there's a Conv2d projector, read its kernel/stride
        for m in feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                # kernel_size: (F_patch, T_patch); stride: (F_stride, T_stride)
                t_patch = int(m.kernel_size[-1])
                t_stride = int(m.stride[-1])
                return t_patch, t_stride

        return None

    @staticmethod
    def _teacher_avg(encoder: nn.Module, patches: Tensor, need_grad: bool) -> Tensor:
        """
        Run teacher encoder, average across layers. Respects training/freeze state.
        """
        with torch.set_grad_enabled(need_grad):
            layers: List[Tensor] = encoder(patches)  # list[L] of (B,P,E)
            reps = torch.stack(layers, dim=0).mean(dim=0)  # (B,P,E)
        return reps

    # ----------------------------- forward ------------------------------
    def forward(
        self, waveforms: Tensor, lengths: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            waveforms: (B, 1, T) padded waveforms.
            lengths:   (B,) raw sample lengths before padding (optional).

        Returns:
            feats_time: (B, T_out, D)
            t_lengths:  (B,)
        """
        if waveforms.dim() != 3 or waveforms.size(1) != 1:
            raise ValueError(f"Expected input shape (B, 1, T), got {tuple(waveforms.shape)}")

        B = waveforms.size(0)

        # 1) Log-mel spectrogram → (B, 1, F_spec, T_spec)
        logmel = self.logmel_transform(waveforms)
        T_spec = logmel.size(-1)

        # 2) Patchify → (B, P, E) with grid (F_g, T_g)
        patches, (F_g, T_g) = self.feature_extractor(logmel)  # P == F_g * T_g
        P, E = patches.size(1), patches.size(2)
        if P != F_g * T_g:
            raise RuntimeError("Feature extractor grid does not match flattened token count.")

        # 3) Teacher encoder (avg over layers). Allow grads only if module is unfrozen.
        need_grad = any(p.requires_grad for p in self.teacher_encoder.parameters())
        reps = self._teacher_avg(self.teacher_encoder, patches, need_grad=need_grad)  # (B,P,E)

        # 4) Reshape to (B, T_g, F_g, E) then mean over freq → time sequence
        reps_2d = reps.view(B, T_g, F_g, E)     # (B, T_g, F_g, E)
        feats_time = reps_2d.mean(dim=2)        # (B, T_g, E)  <-- CTC time axis

        # 5) Per-item **token lengths** (exact if possible; safe fallback otherwise)
        device = feats_time.device

        # Defaults if lengths not provided: full length
        if lengths is None:
            t_lengths = torch.full((B,), T_g, dtype=torch.long, device=device)
            return feats_time, t_lengths

        # Try exact STFT → patch math
        n_fft, hop, win, center = self._try_get_stft_params(self.logmel_transform)
        tps = self._try_get_time_patch_stride(self.feature_extractor)

        # Per-item spectrogram frames from raw sample lengths
        T_spec_i = self._stft_frames(lengths.to(device), n_fft, hop, win, center)
        # Clamp to batch spectrogram length just in case
        T_spec_i = torch.clamp(T_spec_i, min=1, max=T_spec)

        if tps is not None:
            t_patch, t_stride = tps
            # tokens per item along time: floor((T_spec_i - t_patch) / t_stride) + 1
            # make sure we never go below 1
            num = (T_spec_i.to(torch.float32) - float(t_patch))
            t_lengths = torch.floor(num / float(t_stride)).to(torch.long) + 1
            t_lengths = torch.clamp(t_lengths, min=1, max=T_g).to(device=device, dtype=torch.long)
        else:
            # Fallback: **ceil** proportional mapping to avoid undercount
            # t_len = ceil(T_g * T_spec_i / T_spec)
            ratio = (T_spec_i.to(torch.float32) / float(T_spec))
            t_lengths = torch.ceil(ratio * float(T_g)).to(torch.long)
            t_lengths = torch.clamp(t_lengths, min=1, max=T_g).to(device=device, dtype=torch.long)

        # Final sanity: cannot exceed actual time dimension
        if t_lengths.max().item() > feats_time.size(1):
            t_lengths = torch.clamp(t_lengths, max=feats_time.size(1))

        return feats_time, t_lengths
