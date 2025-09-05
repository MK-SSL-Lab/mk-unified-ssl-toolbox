import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Union


class EATBackbone(nn.Module):
    """
    Backbone for downstream use of a pre-trained EAT model (student path).

    Pipeline:
        wave -> LogMelSpectrogramTransform -> SpectrogramPatchEmbedder -> ViT(student)

    This module applies NO masking and NO decoder. It produces patch-level
    token embeddings and the (optional) CLS embedding. It also returns the
    effective (unpadded) sequence length in **patch units** (time-patches),
    i.e., how many tokens are valid for each sample after your front-end.

    Args:
        pretrained_model: An instance of your EAT class exposing:
            - logmel_transform
            - feature_extractor
            - student_encoder (output_all_layers=False)
            - cls_token
        normalize: If True, L2-normalize returned embeddings.

    Inputs:
        waveforms: Float tensor (B, 1, T), raw mono audio.
        lengths: Optional Long/Int tensor (B,) with the TRUE waveform lengths
                 (in samples) before padding. If omitted, the final lengths
                 are assumed to be the full patch length for each item.
        return_cls: If True, also return the CLS embedding.

    Returns:
        If return_cls=False:
            token_embeddings: (B, P, E)  # CLS removed
            final_lengths:    (B,)       # valid tokens count (time-patches)
        If return_cls=True:
            token_embeddings: (B, P, E)
            final_lengths:    (B,)
            cls_embeddings:   (B, E)
    """

    def __init__(self, pretrained_model: nn.Module, normalize: bool = False):
        super().__init__()
        needed = ["logmel_transform", "feature_extractor", "student_encoder", "cls_token"]
        for n in needed:
            if not hasattr(pretrained_model, n):
                raise AttributeError(
                    f"`pretrained_model` lacks `{n}`; expected an instance of your EAT class."
                )
        self.logmel_transform = pretrained_model.logmel_transform
        self.feature_extractor = pretrained_model.feature_extractor
        self.student_encoder = pretrained_model.student_encoder
        self.cls_token = pretrained_model.cls_token

        self.normalize = normalize

    def _maybe_l2(self, x: Tensor) -> Tensor:
        if not self.normalize:
            return x
        return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

    def _wave_to_frame_lengths(self, lengths_samples: Tensor, total_frames: int) -> Tensor:
        """
        Convert true waveform lengths (in samples) to spectrogram frame counts,
        using the same STFT params as the LogMel transform.

        L_spec = floor((L_wave - win_length)/hop_length) + 1, clamped to [0, total_frames]
        """
        hop = getattr(self.logmel_transform, "hop_length")
        win = getattr(self.logmel_transform, "win_length")
        frames = torch.div((lengths_samples - win), hop, rounding_mode="floor") + 1
        frames = torch.clamp(frames, min=0, max=total_frames)
        return frames

    def forward(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
        return_cls: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """
        See class docstring for details.
        """
        if waveforms.dim() != 3 or waveforms.size(1) != 1:
            raise ValueError(f"Expected input shape (B, 1, T), got {tuple(waveforms.shape)}")

        # 1) Log-mel
        logmel = self.logmel_transform(waveforms)               # (B, n_mels, T_spec)
        if torch.isnan(logmel).any():
            raise ValueError("NaN in log-mel output")

        # 2) Patchify (returns patches and grid sizes)
        patches, (F_g, T_g) = self.feature_extractor(logmel)    # patches: (B, P, E)
        B, P, E = patches.shape  # P == F_g * T_g (rasterized order)

        # 3) Build effective (unpadded) final lengths in **patch units (time)**
        # We derive the valid count along time by mapping true waveform lengths
        # -> spectrogram frames -> time-patches. We scale into the time grid T_g.
        if lengths is not None:
            # frames valid in spectrogram (time axis)
            total_frames = logmel.size(-1)
            frame_lengths = self._wave_to_frame_lengths(lengths.to(device=logmel.device), total_frames)
            # proportional mapping frames -> time-patches (T_g)
            time_patch_lengths = torch.div(
                frame_lengths * T_g, total_frames, rounding_mode="floor"
            )
            # clip to [0, T_g]
            time_patch_lengths = torch.clamp(time_patch_lengths, min=0, max=T_g)
        else:
            # no padding info -> assume all time patches are valid
            time_patch_lengths = torch.full(
                (B,), T_g, dtype=torch.long, device=patches.device
            )

        # 4) Encode with student (prepend CLS, no masking)
        cls_tok = self.cls_token.expand(B, 1, E)                # (B, 1, E)
        x = torch.cat([cls_tok, patches], dim=1)                # (B, 1+P, E)
        out = self.student_encoder(x)                           # (B, 1+P, E)
        if torch.isnan(out).any():
            raise ValueError("NaN in student encoder output")

        cls_embeddings = out[:, 0]                              # (B, E)
        token_embeddings = out[:, 1:]                           # (B, P, E)

        token_embeddings = self._maybe_l2(token_embeddings)
        cls_embeddings = self._maybe_l2(cls_embeddings)

        # Return tokens + final (unpadded) length in patch units, and optional CLS
        if return_cls:
            return token_embeddings, time_patch_lengths, cls_embeddings
        else:
            return token_embeddings, time_patch_lengths
