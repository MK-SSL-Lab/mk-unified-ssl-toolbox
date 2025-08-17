import torch
from typing import Tuple, Optional, Sequence
import random


class COLAAudioTransform:
    """Random 2×960 ms crops for COLA, padding-aware (optional).

    If `lengths` is provided, each sample's *true* length (in samples) is used
    to constrain the crop offsets so they never enter the padded tail.

    If `lengths` is omitted, behavior is unchanged vs. the original
    implementation (i.e., uses the tensor's time dimension as total length).

    Args:
        segment_ms: Segment duration in milliseconds. Default: 960.
        sample_rate: Sampling rate (Hz). Default: 16000.

    Call Args:
        batch_waveform: Tensor of shape [B, 1, Lpad] or [1, L].
        lengths: Optional per-sample true lengths (in samples). Shape [B] or
            Python sequence of ints. If given with a single sample, can be a
            scalar tensor/int.

    Returns:
        (seg1, seg2): Two tensors of shape [B, 1, segment_len] for batched
            input, or [1, segment_len] for single input.

    Raises:
        ValueError: On invalid shapes or insufficient true length.
    """

    def __init__(self, segment_ms: int = 960, sample_rate: int = 16000):
        self.segment_len = int(segment_ms / 1000 * sample_rate)

    def _transform_single(self, waveform: torch.Tensor, true_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # waveform: [1, Lpad]; true_len: usable (unpadded) samples
        if waveform.dim() != 2 or waveform.shape[0] != 1:
            raise ValueError(
                f"Expected waveform shape [1, L], but got {tuple(waveform.shape)}. "
                "Please ensure the dataset returns audio in [channels, time] format."
            )

        need = 2 * self.segment_len
        if true_len < need:
            raise ValueError(f"Waveform too short for two segments ({true_len} < {need}).")

        # First segment start ∈ [0, true_len - 2*seg_len]
        start1 = random.randint(0, true_len - need)
        # Second segment start ∈ [start1 + seg_len, true_len - seg_len]
        start2 = random.randint(start1 + self.segment_len, true_len - self.segment_len)

        seg1 = waveform[:, start1:start1 + self.segment_len]
        seg2 = waveform[:, start2:start2 + self.segment_len]
        return seg1, seg2

    @staticmethod
    def _coerce_lengths(lengths: Optional[Sequence[int] | torch.Tensor], fallback_len: int, batch: int) -> list[int]:
        # Turn various length inputs into a Python list[int] of size B
        if lengths is None:
            return [fallback_len] * batch

        if isinstance(lengths, torch.Tensor):
            if lengths.numel() == 1 and batch == 1:
                return [int(lengths.item())]
            lengths = lengths.view(-1).tolist()

        if not isinstance(lengths, (list, tuple)):
            # single int
            return [int(lengths)] * batch

        if len(lengths) != batch:
            raise ValueError(f"`lengths` size {len(lengths)} does not match batch size {batch}.")
        return [int(x) for x in lengths]

    def __call__(self, batch_waveform: torch.Tensor, lengths: Optional[Sequence[int] | torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Single sample: [1, L]
        if batch_waveform.dim() == 2 and batch_waveform.shape[0] == 1:
            true_len_list = self._coerce_lengths(lengths, fallback_len=batch_waveform.size(1), batch=1)
            return self._transform_single(batch_waveform, true_len_list[0])

        # Batched: [B, 1, Lpad]
        elif batch_waveform.dim() == 3 and batch_waveform.shape[1] == 1:
            B, _, Lpad = batch_waveform.shape
            true_len_list = self._coerce_lengths(lengths, fallback_len=Lpad, batch=B)

            segs0, segs1 = [], []
            for i in range(B):
                s0, s1 = self._transform_single(batch_waveform[i], true_len_list[i])
                segs0.append(s0)
                segs1.append(s1)
            return torch.stack(segs0, dim=0), torch.stack(segs1, dim=0)

        else:
            raise ValueError(
                f"Expected input shape [1, L] or [B, 1, L], but got {tuple(batch_waveform.shape)}."
            )
