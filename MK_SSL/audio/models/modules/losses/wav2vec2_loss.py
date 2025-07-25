import torch
import torch.nn as nn
from torch import Tensor
from typing import List
import random

class Wav2Vec2Loss(nn.Module):
    """Contrastive + Diversity loss for wav2vec 2.0 pretraining.

    Implements the contrastive prediction task (Lm) and the codebook
    diversity loss (Ld) from Baevski et al. (2020).

    Args:
        temperature (float): Temperature scaling for contrastive loss.
        num_distractors (int): Number of negative distractor samples per positive sample.
        alpha (float): Weight for the diversity loss term.
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

    def forward(
        self,
        context: Tensor,
        quantized: Tensor,
        codevector_probs: Tensor,
        time_mask_indices: Tensor,
    ) -> Tensor:
        """Compute total loss.

        Args:
            context (Tensor): Contextualized encoder output of shape (B, T, D).
            quantized (Tensor): Quantized targets of shape (B, T, D).
            codevector_probs (Tensor): Codebook probabilities of shape (G, V).
            time_mask_indices (Tensor): Boolean mask for time positions of shape (B, T).

        Returns:
            Tensor: Combined loss scalar.
        """
        target_context = context[time_mask_indices]
        target_quantized = quantized[time_mask_indices]

        targets_per_batch = [int(time_mask_indices[i].sum()) for i in range(time_mask_indices.size(0))]

        negatives = self._sample_negatives(target_quantized, targets_per_batch)
        negatives = torch.cat([target_quantized.unsqueeze(1), negatives], dim=1)

        contrastive_loss = self._contrastive_loss(target_context, target_quantized, negatives)
        diversity_loss = self._diversity_loss(codevector_probs)

        return contrastive_loss + self.alpha * diversity_loss

    def _contrastive_loss(self, targets: Tensor, positives: Tensor, negatives: Tensor) -> Tensor:
        """Compute contrastive loss."""
        pos_sim = torch.exp(self.similarity(targets, positives) / self.temperature)
        neg_sim = torch.exp(self.similarity(targets.unsqueeze(1), negatives) / self.temperature).sum(dim=1)
        return -torch.log(pos_sim / neg_sim).mean()

    def _diversity_loss(self, probs: Tensor) -> Tensor:
        """Compute diversity loss encouraging equal codebook usage."""
        entropy = torch.sum(probs * torch.log(probs + 1e-7), dim=-1)
        G, V = probs.shape
        return torch.sum(entropy) / (G * V)

    def _sample_negatives(self, positives: Tensor, targets_per_batch: List[int]) -> Tensor:
        """Sample negative examples for contrastive loss."""
        negatives = []
        start = 0
        D = positives.size(-1)

        for count in targets_per_batch:
            if count <= 1:

                negatives.append(positives.new_zeros((count, self.num_distractors, D)))
                start += count
                continue

            idx_range = torch.arange(count, device=positives.device)
            mask = torch.eye(count, device=positives.device).bool()
            candidate_indices = idx_range.repeat(count).view(count, -1)[~mask].view(count, -1)
            candidate_indices += start

            rand_rows = torch.arange(count).repeat_interleave(self.num_distractors)
            rand_cols = []
            for _ in range(count):
                if count - 1 >= self.num_distractors:
                    cols = random.sample(range(count - 1), self.num_distractors)
                else:
                    cols = random.choices(range(count - 1), k=self.num_distractors)
                rand_cols.extend(cols)
            rand_cols = torch.tensor(rand_cols, device=positives.device)

            sampled = candidate_indices[rand_rows, rand_cols]
            # Reshape to [count, num_distractors, D]
            negatives.append(positives[sampled].view(count, self.num_distractors, D))
            start += count

        return torch.cat(negatives, dim=0)  # shape [total_masked, num_distractors, D]