import torch
import torch.nn as nn
from torch import Tensor
from typing import List


class Wav2Vec2Loss(nn.Module):
    """Contrastive-plus-diversity loss used for wav2vec 2.0 pre-training.

    This implements
        L = L_m + α · L_d             (Eq. 2, Baevski et al., 2020)

    where
        – L_m is a cross-entropy / NT-Xent contrastive loss (Eq. 3),
        – L_d is the codebook diversity (entropy) loss (Eq. 4).

    Args:
        temperature:   κ in the paper (default 0.1).
        num_distractors: number of negative samples K (default 100).
        alpha:         weight α for the diversity term (default 0.1).
    """

    def __init__(
        self,
        temperature: float = 0.1,
        num_distractors: int = 100,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.num_distractors = num_distractors
        self.alpha = alpha
        self.similarity = nn.CosineSimilarity(dim=-1)

    # Forward
    def forward(
        self,
        context: Tensor,            # (B, T, D)    – Transformer outputs   (c_t)
        quantized: Tensor,          # (B, T, D)    – Quantized targets     (q_t)
        codevector_probs: Tensor,   # (G, V)       – Avg. p_{g,v}
        time_mask_indices: Tensor,  # (B, T) bool  – Masked positions
    ) -> Tensor:
        """Compute total loss L = L_m + α·L_d."""

        # Select only masked positions (positive samples)
        target_context   = context[time_mask_indices]     # (M, D)
        target_quantized = quantized[time_mask_indices]   # (M, D)

        # How many masked positions per sequence – used for same-utterance negatives
        targets_per_seq = [
            int(time_mask_indices[i].sum()) for i in range(time_mask_indices.size(0))
        ]

        negatives = self._sample_negatives(target_quantized, targets_per_seq)  # (M, K, D)

        contrastive_loss = self._contrastive_loss(
            target_context,                  # c_t
            target_quantized,                # positive q_t
            negatives,                       # K negatives
        )
        diversity_loss   = self._diversity_loss(codevector_probs)

        return contrastive_loss + self.alpha * diversity_loss

    # Contrastive loss  L_m
    def _contrastive_loss(
        self,
        targets:   Tensor,   # (M, D)   – context c_t
        positives: Tensor,   # (M, D)   – matching q_t
        negatives: Tensor,   # (M, K, D)
    ) -> Tensor:
        """Contrastive loss – Eq. 3."""
        # Positive similarities
        pos_sim = torch.exp(self.similarity(targets, positives) / self.temperature)      # (M,)

        # Negative similarities (K per example)
        neg_sim = torch.exp(
            self.similarity(targets.unsqueeze(1), negatives) / self.temperature          # (M, K)
        ).sum(dim=1)                                                                     # (M,)

        total_sim = pos_sim + neg_sim                                                    # denom
        return -torch.log(pos_sim / total_sim).mean()

    # Diversity loss  L_d
    def _diversity_loss(self, probs: Tensor) -> Tensor:
        """Entropy-based diversity loss – Eq. 4."""
        entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=-1)  # (G,)
        G, V = probs.shape
        return entropy.sum() / (G * V)

    # Negative sampling (same utterance)
    def _sample_negatives(
        self, positives: Tensor, targets_per_seq: List[int]
    ) -> Tensor:
        """Sample K negatives for every masked position from same sequence.

        Args:
            positives: Flattened (M, D) tensor of positives.
            targets_per_seq: List with #masked positions for each utterance.

        Returns:
            Tensor of shape (M, K, D) with negative samples.
        """
        negatives = []
        start = 0
        D = positives.size(-1)

        for count in targets_per_seq:
            if count <= 1:
                # Edge-case: a sequence has ≤1 masked pos → fill zeros
                negatives.append(positives.new_zeros((count, self.num_distractors, D)))
                start += count
                continue

            # indices for the current utterance in `positives`
            current_pos = positives[start : start + count]          # (count, D)

            # For each masked position, draw K indices ∈ [0, count-1] (with replacement)
            idx = torch.randint(
                0, count, (count, self.num_distractors), device=positives.device
            )                                                       # (count, K)
            neg = current_pos[idx]                                  # (count, K, D)
            negatives.append(neg)
            start += count

        return torch.cat(negatives, dim=0)   # (M, K, D)
