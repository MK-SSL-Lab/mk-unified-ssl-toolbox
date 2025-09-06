import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


class HuBERTBackbone(nn.Module):
    """
    Encoder-only backbone for a pretrained HuBERT model (no masking/prediction head).

    Pipeline:
        wave -> ConvFeatureExtractor -> Linear + LayerNorm + Dropout
             -> TransformerEncoder
             -> length-aware mean pool over time -> (B, D) features

    Returns a single utterance-level representation per waveform plus lengths (samples).

    Args:
        pretrained_model (nn.Module): HuBERT instance exposing:
            - feature_extractor(waveforms, lengths) -> (feat_seq, frame_lengths)
            - feature_projection (nn.Linear)
            - post_extract_proj_norm (nn.LayerNorm)
            - post_extract_proj_dropout (nn.Dropout)
            - encoder(feat_seq, frame_lengths)
        normalize (bool): If True, L2-normalize returned features.
    """

    def __init__(self, pretrained_model: nn.Module, normalize: bool = False):
        super().__init__()

        needed = [
            "feature_extractor",
            "feature_projection",
            "post_extract_proj_norm",
            "post_extract_proj_dropout",
            "encoder",
        ]
        for n in needed:
            if not hasattr(pretrained_model, n):
                raise AttributeError(f"`pretrained_model` lacks `{n}`; expected a HuBERT instance.")

        self.feature_extractor = pretrained_model.feature_extractor
        self.feature_projection = pretrained_model.feature_projection
        self.post_extract_proj_norm = pretrained_model.post_extract_proj_norm
        self.post_extract_proj_dropout = pretrained_model.post_extract_proj_dropout
        self.encoder = pretrained_model.encoder

        # Useful reference
        self.encoder_embed_dim = getattr(self.encoder, "embed_dim", None)

        self.normalize = normalize

    def _maybe_l2(self, x: Tensor) -> Tensor:
        if not self.normalize:
            return x
        return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

    @torch.no_grad()
    def forward(
        self,
        waveforms: Tensor,                 # (B, 1, T)
        lengths: Optional[Tensor] = None,  # (B,) in samples
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            features:      (B, D) where D == encoder_embed_dim
            final_lengths: (B,) in samples (same units as input `lengths`)
        """
        if waveforms.dim() != 3 or waveforms.size(1) != 1:
            raise ValueError(f"Expected input shape (B, 1, T), got {tuple(waveforms.shape)}")

        B, _, T_pad = waveforms.shape
        device = waveforms.device

        # If lengths not provided, assume full padded length in samples
        if lengths is None:
            lengths = torch.full((B,), T_pad, dtype=torch.long, device=device)
        else:
            lengths = lengths.to(device=device, dtype=torch.long)

        # 1) Conv feature extractor → (feat_seq, frame_lengths)
        #    feat_seq: (B, T_feat, C_in), frame_lengths: (B,) in frames
        feat_seq, frame_lengths = self.feature_extractor(waveforms, lengths)
        if torch.isnan(feat_seq).any():
            raise ValueError("NaN encountered in ConvFeatureExtractor output")

        # 2) Project to encoder dim + norm + dropout
        feat_seq = self.feature_projection(feat_seq)
        feat_seq = self.post_extract_proj_norm(feat_seq)
        feat_seq = self.post_extract_proj_dropout(feat_seq)

        # 3) Transformer encoder (length-aware) → (B, T_enc, D)
        enc = self.encoder(feat_seq, frame_lengths)
        if torch.isnan(enc).any():
            raise ValueError("NaN encountered in Transformer encoder output")

        # 4) Length-aware mean pooling over time (use frame_lengths)
        T_enc = enc.size(1)
        arange = torch.arange(T_enc, device=device).unsqueeze(0)            # (1, T_enc)
        valid_mask = (arange < frame_lengths.unsqueeze(1)).unsqueeze(-1)    # (B, T_enc, 1)
        valid_mask = valid_mask.to(enc.dtype)

        denom = valid_mask.sum(dim=1).clamp_min(1.0)                        # (B, 1)
        pooled = (enc * valid_mask).sum(dim=1) / denom                      # (B, D)

        feats = self._maybe_l2(pooled)

        # Return original sample lengths for consistency with other backbones
        final_lengths = lengths
        return feats, final_lengths
