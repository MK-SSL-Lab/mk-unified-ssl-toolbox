import torch
from typing import Tuple, Optional, Sequence
import random

class COLAAudioTransform:
    def __init__(self, segment_ms: int = 960, sample_rate: int = 16000):
        self.segment_len = int(segment_ms / 1000 * sample_rate)

    def _transform_single(
        self,
        waveform: torch.Tensor,         # [1, L_total] (may include right-pad zeros)
        true_len: Optional[int] = None  # number of *real* samples (orig_len)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Produce two fixed-length crops (each [1, segment_len]) strictly within the real audio.
        """
        if waveform.dim() != 2 or waveform.shape[0] != 1:
            raise ValueError(
                f"Expected waveform shape [1, L], got {tuple(waveform.shape)}."
            )

        total_len = waveform.size(1)
        L = int(true_len) if (true_len is not None) else total_len  # fall back if needed

        needed = 2 * self.segment_len
        if L < needed:
            raise ValueError(
                f"Waveform too short for two segments from real audio "
                f"({L} < {needed}). Consider filtering such files."
            )

        # Offsets sampled only from the *real* part [0, L)
        offset1 = random.randint(0, L - needed)
        offset2 = random.randint(offset1 + self.segment_len, L - self.segment_len)

        seg1 = waveform[:, offset1:offset1 + self.segment_len]
        seg2 = waveform[:, offset2:offset2 + self.segment_len]
        return seg1, seg2

    def __call__(
        self,
        batch_waveform: torch.Tensor,             # [B, 1, L_total] or [1, L_total]
        lengths: Optional[Sequence[int]] = None   # true lengths per item (orig_len)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        If input is a batch, 'lengths' must be a sequence of size B giving orig_len for each item.
        Returns (view0, view1) with shapes [B, 1, segment_len].
        """
        if batch_waveform.dim() == 2:  # single sample [1, L]
            true_len = None if lengths is None else int(lengths)
            return self._transform_single(batch_waveform, true_len)

        if batch_waveform.dim() == 3:  # batch [B, 1, L]
            B = batch_waveform.size(0)
            if lengths is None:
                # Fall back to total length, but this re-introduces pad risk.
                # Pass 'lengths' from the dataset to avoid that.
                lengths = [batch_waveform.size(2)] * B
            if len(lengths) != B:
                raise ValueError(f"'lengths' must have size B={B}, got {len(lengths)}")

            segs0, segs1 = [], []
            for i in range(B):
                s0, s1 = self._transform_single(batch_waveform[i], int(lengths[i]))
                segs0.append(s0)
                segs1.append(s1)
            return torch.stack(segs0, dim=0), torch.stack(segs1, dim=0)

        raise ValueError(
            f"Expected input shape [1, L] or [B, 1, L], got {tuple(batch_waveform.shape)}."
        )
