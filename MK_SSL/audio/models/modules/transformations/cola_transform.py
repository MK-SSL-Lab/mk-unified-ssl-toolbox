import torch
from typing import Tuple
import random


class COLAAudioTransform:
    def __init__(self, segment_ms: int = 960, sample_rate: int = 16000):
        self.segment_len = int(segment_ms / 1000 * sample_rate)

    def _transform_single(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply COLA transform to a single waveform [1, L].
        """
        if waveform.dim() != 2 or waveform.shape[0] != 1:
            raise ValueError(
                f"Expected waveform shape [1, L], but got {tuple(waveform.shape)}. "
                "Please ensure the dataset returns audio in [channels, time] format."
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
        return seg1, seg2

    def __call__(self, batch_waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply COLA transform to a batch [B, 1, L] or a single waveform [1, L].
        Returns:
            Tuple of (view0, view1) with shape [B, 1, segment_len] if input is batch.
        """
        if batch_waveform.dim() == 2:  # Single sample [1, L]
            return self._transform_single(batch_waveform)

        elif batch_waveform.dim() == 3:  # Batch [B, 1, L]
            segs0, segs1 = [], []
            for waveform in batch_waveform:  # Each waveform is [1, L]
                s0, s1 = self._transform_single(waveform)
                segs0.append(s0)
                segs1.append(s1)
            return torch.stack(segs0), torch.stack(segs1)

        else:
            raise ValueError(
                f"Expected input shape [1, L] or [B, 1, L], but got {tuple(batch_waveform.shape)}."
            )
