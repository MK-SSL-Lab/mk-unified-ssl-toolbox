import torch
import torch.nn as nn
from torch import Tensor
from typing import List


class Wav2Vec2Loss(nn.Module):
    r"""Contrastive‑plus‑diversity loss from *wav2vec 2.0: A Framework for
    Self‑Supervised Learning of Speech Representations* (Baevski et al., 2020).

        L = L_m + α · L_d                                      (Eq. 2)

    where
        • L_m – NT‑Xent (cross‑entropy) contrastive loss       (Eq. 3)
        • L_d – code‑book diversity loss (= –entropy)          (Eq. 4)

    Args
    ----
    temperature : float
        κ in the paper (default 0.1).
    num_distractors : int
        Number of negative samples K (default 100).
    alpha : float
        Weight α applied to the diversity term (default 0.1).
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
        context: Tensor,            # (B, T, D) – transformer outputs c_t
        quantized: Tensor,          # (B, T, D) – quantised targets  q_t
        codevector_probs: Tensor,   # (G, V)    – running mean p̄_{g,v}
        time_mask_indices: Tensor,  # (B, T) bool – which positions are masked
    ) -> Tensor:
        """Compute total loss L = L_m + α·L_d."""
        # 1) gather only masked positions (the “positive” pairs)
        target_ctx   = context[time_mask_indices]          # (M, D)
        target_quant = quantized[time_mask_indices]        # (M, D)

        # 2) how many masked positions per sequence? (for same‑utt negatives)
        per_seq = time_mask_indices.sum(dim=1).tolist()

        # 3) same‑utterance negatives
        negatives = self._sample_negatives(target_quant, per_seq)  # (M, K, D)

        # 4) losses
        contrast = self._contrastive_loss(target_ctx, target_quant, negatives)
        diversity = self._diversity_loss(codevector_probs)

        return contrast + self.alpha * diversity


    # Contrastive loss  L_m
    def _contrastive_loss(
        self,
        targets:   Tensor,          # (M, D) – c_t
        positives: Tensor,          # (M, D) – matching q_t
        negatives: Tensor,          # (M, K, D)
    ) -> Tensor:
        """NT‑Xent contrastive loss (Eq. 3)."""
        T = self.temperature

        # Positive similarities
        pos_sim = torch.exp(self.similarity(targets, positives) / T)      # (M,)

        # Negative similarities (K per example)
        neg_sim = torch.exp(
            self.similarity(targets.unsqueeze(1), negatives) / T          # (M, K)
        ).sum(dim=1)                                                      # (M,)

        return -torch.log(pos_sim / (pos_sim + neg_sim)).mean()


    # Diversity loss  L_d
    def _diversity_loss(self, probs: Tensor) -> Tensor:
        """Negative entropy (Eq. 4) → encourages code‑book utilisation."""
        # NB: *no* leading minus ‑‑ matches the sign in the paper
        neg_entropy = torch.sum(probs * torch.log(probs + 1e-7), dim=-1)   # (G,) ≤ 0
        G, V = probs.shape
        return neg_entropy.sum() / (G * V)    # value ≤ 0; minimisation ⇒ ↑ entropy

    # Negative‑sample helper
    def _sample_negatives(
        self,
        positives: Tensor,          # (M, D)
        targets_per_seq: List[int], # [len(seq₀_masked), len(seq₁_masked), …]
    ) -> Tensor:
        """Sample K negatives for each masked position from the *same* utterance."""
        negatives = []
        start = 0
        D = positives.size(-1)

        for count in targets_per_seq:
            if count <= 1:
                # edge‑case: utterance has ≤1 masked position
                negatives.append(positives.new_zeros((count, self.num_distractors, D)))
                start += count
                continue

            # masked positions belonging to the current utterance
            current = positives[start : start + count]                     # (count, D)

            # draw K indices ∈ [0, count‑1] with replacement for each position
            idx = torch.randint(
                0, count, (count, self.num_distractors), device=positives.device
            )                                                              # (count, K)
            negatives.append(current[idx])                                 # (count, K, D)
            start += count

        return torch.cat(negatives, dim=0)   # (M, K, D)
