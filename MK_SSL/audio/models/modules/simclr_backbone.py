import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


class SimCLRBackbone(nn.Module):
    """
    Backbone for downstream use of a pretrained SimCLRSpeech model.

    Aligned with SimCLRSpeech.encode:
        wave -> FBANK(80) -> optional input_proj -> Transformer backbone
            -> masked mean pool over time -> projection_head -> embedding

    Returns a single utterance-level embedding per waveform plus the (sample) lengths.

    Args:
        pretrained_model (nn.Module): Your SimCLRSpeech instance exposing:
            - fbank
            - input_proj
            - backbone
            - projection_head
        normalize (bool): If True, L2-normalize the returned embeddings.

    Inputs:
        waveforms: (B, 1, T) raw mono audio, padded to a common T.
        lengths: Optional (B,) *true* audio lengths in **samples** (before padding).

    Returns:
        embeddings: (B, E) where E = projection_dim used in SimCLRSpeech
        final_lengths: (B,) in **samples** (same units as `lengths`)
    """

    def __init__(self, pretrained_model: nn.Module, normalize: bool = False):
        super().__init__()

        needed = ["fbank", "input_proj", "backbone", "projection_head"]
        for n in needed:
            if not hasattr(pretrained_model, n):
                raise AttributeError(
                    f"`pretrained_model` lacks `{n}`; expected a SimCLRSpeech instance."
                )

        self.fbank = pretrained_model.fbank
        self.input_proj = pretrained_model.input_proj
        self.backbone = pretrained_model.backbone
        self.projection_head = pretrained_model.projection_head

        # Useful references if needed downstream
        self.embed_dim = getattr(self.backbone, "embed_dim", None)
        self.projection_dim = getattr(pretrained_model, "projection_head", None)
        self.normalize = normalize

    def _maybe_l2(self, x: Tensor) -> Tensor:
        if not self.normalize:
            return x
        return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

    @torch.no_grad()
    def forward(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        See class docstring for shapes and semantics.
        """
        if waveforms.dim() != 3 or waveforms.size(1) != 1:
            raise ValueError(f"Expected input shape (B, 1, T), got {tuple(waveforms.shape)}")

        # 1) FBANK (B, T_spec, 80)
        feats = self.fbank(waveforms)
        if torch.isnan(feats).any():
            raise ValueError("NaN encountered in FBANK output")

        # 2) Optional input projection to match Transformer embed dim (B, T_spec, E)
        feats = self.input_proj(feats)

        # 3) Transformer encoder (B, T_spec, E)
        enc = self.backbone(feats)
        if torch.isnan(enc).any():
            raise ValueError("NaN encountered in Transformer output")

        # 4) Masked mean over time using provided sample-lengths → spec-frame lengths
        B, T_spec, E = enc.shape
        if lengths is not None:
            total_samples = waveforms.size(-1)          # padded T (samples)
            total_frames  = T_spec                      # FBANK frames
            # proportional mapping samples → frames (integer, clipped)
            frame_lengths = torch.div(
                lengths.to(device=enc.device) * total_frames,
                total_samples,
                rounding_mode="floor",
            )
            frame_lengths = torch.clamp(frame_lengths, min=0, max=total_frames).to(torch.long)

            # build mask (B, T_spec, 1)
            arange = torch.arange(T_spec, device=enc.device).unsqueeze(0)  # (1, T_spec)
            mask = (arange < frame_lengths.unsqueeze(1)).unsqueeze(-1).to(enc.dtype)  # (B,T_spec,1)

            denom = mask.sum(dim=1).clamp_min(1.0)       # (B,1)
            pooled = (enc * mask).sum(dim=1) / denom     # (B,E)
            final_lengths = lengths.to(device=enc.device, dtype=torch.long)
        else:
            pooled = enc.mean(dim=1)                      # (B,E)
            final_lengths = torch.full(
                (B,),
                waveforms.size(-1),
                dtype=torch.long,
                device=enc.device,
            )

        # 5) SimCLR projection head → utterance embedding (B, proj_dim)
        emb = self.projection_head(pooled)
        if torch.isnan(emb).any():
            raise ValueError("NaN encountered in projection head output")

        emb = self._maybe_l2(emb)                         # optional L2-normalization
        return emb, final_lengths
