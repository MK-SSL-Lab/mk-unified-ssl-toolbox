import torch
from typing import Tuple
import random


class COLAAudioTransform:
    def __init__(self, segment_ms: int = 960, sample_rate: int = 16000):
        self.segment_len = int(segment_ms / 1000 * sample_rate)

    def __call__(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
       
        # Ensure [1, L] shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2 and waveform.shape[0] != 1:
            raise ValueError("Expected mono waveform [1, L] or [L], but got multi-channel.")

        total_len = waveform.size(1)
        if total_len < 2 * self.segment_len:
            raise ValueError("Waveform too short for two segments.")

        # Choose two non-overlapping random offsets
        offset1 = random.randint(0, total_len - 2 * self.segment_len)
        offset2 = random.randint(offset1 + self.segment_len, total_len - self.segment_len)

        seg1 = waveform[:, offset1:offset1 + self.segment_len]
        seg2 = waveform[:, offset2:offset2 + self.segment_len]

        return seg1, seg2

