import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Union


class EATBackbone(nn.Module):
    """
    Backbone for downstream use of a pre-trained EAT model (student path).

    Pipeline:
        wave -> LogMelSpectrogramTransform -> SpectrogramPatchEmbedder -> ViT(student)

    This applies NO masking and NO decoder. It returns:
      - token embeddings (CLS removed),
      - the final (unpadded) length in **time-patch units**,
      - and optionally the CLS embedding.

    Args:
        pretrained_model: An instance of your EAT class exposing:
            - logmel_transform
            - feature_extractor
            - student_encoder (output_all_layers=False)
            - cls_token
        normalize: If True, L2-normalize returned embeddings.

    Inputs:
        waveforms: (B, 1, T) raw mono audio (padded to a common T).
        lengths: Optional (B,) true audio lengths in **samples** before padding.
                 If omitted, all time-patches are considered valid.
        return_cls: If True, also return the CLS embedding.

    Returns:
        If return_cls == False:
            token_embeddings: (B, P, E)  # CLS removed, P = T_g * F_g
            final_lengths:    (B,)       # valid count along **time-patches** (0..T_g)
        If return_cls == True:
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

        # 1) Log-mel (same module as training)
        logmel = self.logmel_transform(waveforms)               # (B, n_mels, T_spec)
        if torch.isnan(logmel).any():
            raise ValueError("NaN in log-mel output")

        # 2) Patchify spectrogram (same embedder as training)
        patches, (F_g, T_g) = self.feature_extractor(logmel)    # patches: (B, P, E), P = F_g*T_g
        B, P, E = patches.shape

        # 3) Compute final unpadded length in **time-patch units** without using hop/win.
        #    We map true waveform lengths (samples) -> spectrogram frames (proportionally)
        #    -> time patches (proportionally). This avoids relying on transform internals.
        if lengths is not None:
            total_samples = waveforms.size(-1)                  # padded T (samples)
            total_frames = logmel.size(-1)                      # T_spec (frames)
            # valid spectrogram frames (proportional mapping from samples)
            frame_lengths = torch.div(
                lengths.to(device=logmel.device) * total_frames,
                total_samples,
                rounding_mode="floor",
            )
            frame_lengths = torch.clamp(frame_lengths, min=0, max=total_frames)
            # frames -> time-patches
            time_patch_lengths = torch.div(
                frame_lengths * T_g, total_frames, rounding_mode="floor"
            )
            final_lengths = torch.clamp(time_patch_lengths, min=0, max=T_g).to(dtype=torch.long)
        else:
            final_lengths = torch.full((B,), T_g, dtype=torch.long, device=patches.device)

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

        if return_cls:
            return token_embeddings, final_lengths, cls_embeddings
        else:
            return token_embeddings, final_lengths
