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

    def forward(self, waveforms: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            waveforms (Tensor): Input tensor with shape **(B, 1, T)**.
            lengths   (Optional[Tensor]): Valid lengths before padding (ignored).

        Returns:
            Tensor: Utterance-level embeddings of shape **(B, E)**.
        """
        if waveforms.dim() != 3 or waveforms.size(1) != 1:
            raise ValueError(
                f"Expected input shape (B, 1, T), but got {tuple(waveforms.shape)}"
            )

        # 1. Log-mel spectrogram
        logmel = self.logmel_transform(waveforms)            # (B, 1, F, T)

        # 2. Patchify
        patches, _ = self.feature_extractor(logmel)          # (B, P, E)

        need_grad = any(p.requires_grad for p in self.teacher_encoder.parameters())
        with torch.set_grad_enabled(need_grad):
            layer_outputs = self.teacher_encoder(patches)      # List[L] of (B, P, E)

        # 4. Average across layers
        reps = torch.stack(layer_outputs).mean(dim=0)

        return reps
