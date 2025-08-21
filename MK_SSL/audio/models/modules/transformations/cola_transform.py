import torch
from typing import Tuple
import random


class COLAAudioTransform:
    def __init__(self, segment_ms: int = 960, sample_rate: int = 16000):
        self.segment_len = int(segment_ms / 1000 * sample_rate)

    def _transform_single(self, waveform: torch.Tensor, orig_len: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Apply COLA transform to a single waveform [1, L].

        Returns:
            seg1, seg2: cropped waveform segments [1, segment_len]
            len1, len2: original unpadded length (same for both segments)
        """
        if waveform.dim() != 2 or waveform.shape[0] != 1:
            raise ValueError(
                f"Expected waveform shape [1, L], but got {tuple(waveform.shape)}."
            )

        total_len = waveform.size(1)
        if total_len < 2 * self.segment_len:
            raise ValueError(
                f"Waveform too short for two segments ({total_len} < {2 * self.segment_len})."
            )

        offset1 = random.randint(0, total_len - 2 * self.segment_len)
        offset2 = random.randint(offset1 + self.segment_len, total_len - self.segment_len)

        seg1 = waveform[:, offset1:offset1 + self.segment_len]
        seg2 = waveform[:, offset2:offset2 + self.segment_len]
        return seg1, seg2, orig_len, orig_len

    def __call__(self, batch_waveform: torch.Tensor, batch_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply COLA transform to a batch.

        Args:
            batch_waveform (Tensor): shape [B, 1, L]
            batch_lengths (Tensor): original unpadded lengths [B]

        Returns:
            segs0, segs1: tensors of shape [B, 1, segment_len]
            lens0, lens1: tensors of shape [B,], each = original length
        """
        if batch_waveform.dim() == 2:  # Single sample [1, L]
            s0, s1, l0, l1 = self._transform_single(batch_waveform, int(batch_lengths))
            return s0.unsqueeze(0), s1.unsqueeze(0), torch.tensor([l0]), torch.tensor([l1])

        elif batch_waveform.dim() == 3:  # Batch [B, 1, L]
            segs0, segs1, lens0, lens1 = [], [], [], []
            for waveform, orig_len in zip(batch_waveform, batch_lengths):
                s0, s1, l0, l1 = self._transform_single(waveform, int(orig_len))
                segs0.append(s0)
                segs1.append(s1)
                lens0.append(l0)
                lens1.append(l1)
            return (
                torch.stack(segs0),
                torch.stack(segs1),
                torch.tensor(lens0, dtype=torch.long),
                torch.tensor(lens1, dtype=torch.long),
            )

        else:
            raise ValueError(
                f"Expected input shape [1, L] or [B, 1, L], but got {tuple(batch_waveform.shape)}."
            )
