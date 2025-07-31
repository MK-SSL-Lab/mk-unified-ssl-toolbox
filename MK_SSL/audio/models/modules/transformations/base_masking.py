import torch
import numpy as np
from typing import Tuple


class InverseBlockMasking:
    """
    Inverse block masking for 2D spectrogram patches.

    This strategy first masks all patches, then reveals a number of rectangular
    blocks until the desired keep ratio is achieved.

    Args:
        shape (Tuple[int, int]): Shape of the patch grid (T, F).
        mask_ratio (float): Percentage of patches to mask (e.g., 0.8 = 80% masked).
        block_size (Tuple[int, int]): Size of blocks to reveal.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        mask_ratio: float = 0.8,
        block_size: Tuple[int, int] = (5, 5),
    ):
        self.T, self.F = shape
        self.mask_ratio = mask_ratio
        self.block_H, self.block_W = block_size
        self.total_patches = self.T * self.F
        self.keep_patches = int(self.total_patches * (1 - mask_ratio))

    def __call__(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Mask of shape (T, F) with 0 for masked, 1 for visible.
        """
        mask = np.zeros((self.T, self.F), dtype=np.int32)
        count = 0

        while count < self.keep_patches:
            i = np.random.randint(0, self.T - self.block_H + 1)
            j = np.random.randint(0, self.F - self.block_W + 1)
            block = mask[i:i + self.block_H, j:j + self.block_W]
            added = np.sum(block == 0)
            mask[i:i + self.block_H, j:j + self.block_W] = 1
            count += added

        return torch.tensor(mask, dtype=torch.bool)
