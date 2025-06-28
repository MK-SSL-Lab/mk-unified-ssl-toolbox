import random

import torch
import torch.nn as nn
from torch import Tensor

from typing import List



class Wav2Vec2Loss(nn.Module):
    def __init__(
        self,
        loss_temperature: float = 0.1,
        num_distractors: int = 100,
        quantizer_num_groups: int = None,
        quantizer_num_entries_per_codebook: int = None,
        loss_alpha: float = 0.1,
    ):
        super().__init__()
        self.temperature = loss_temperature
        self.num_distractors = num_distractors
        self.quantizer_num_groups = quantizer_num_groups
        self.quantizer_num_entries_per_codebook = quantizer_num_entries_per_codebook
        self.alpha = loss_alpha

        self.similarity = nn.CosineSimilarity(dim=-1)

    def forward(
        self,
        context: Tensor,
        quantized: Tensor,
        perplexity: Tensor,
        time_mask_indices: Tensor,
    ) -> Tensor:
        """
        Args:
            context (Tensor): Encoder output. Shape: (B, T, D)
            quantized (Tensor): Quantized targets. Shape: (B, T, D)
            perplexity (Tensor): Code usage distribution. Shape: (G, V)
            time_mask_indices (Tensor): Mask positions. Shape: (B, T)

        Returns:
            Tensor: Combined loss scalar.
        """
        target_context = context[time_mask_indices]
        target_quantized = quantized[time_mask_indices]

        targets_per_batch = [int(time_mask_indices[i].sum()) for i in range(time_mask_indices.size(0))]

        negatives = self._sample_negatives(target_quantized, targets_per_batch)
        negatives = torch.cat([target_quantized.unsqueeze(1), negatives], dim=1)

        contrastive_loss = self._contrastive_loss(target_context, target_quantized, negatives)
        diversity_loss = self._diversity_loss(perplexity)

        return contrastive_loss + self.alpha * diversity_loss


    def _contrastive_loss(
        self,
        targets: Tensor,
        positives: Tensor,
        negatives: Tensor,
    ) -> Tensor:
        """
        Computes contrastive loss.

        Args:
            targets (Tensor): Anchor representations. Shape: (N, D)
            positives (Tensor): Positive samples. Shape: (N, D)
            negatives (Tensor): Negative samples. Shape: (N, K, D)

        Returns:
            Tensor: Scalar loss.
        """
        pos_sim = torch.exp(self.similarity(targets, positives) / self.temperature)
        neg_sim = torch.exp(self.similarity(targets.unsqueeze(1), negatives) / self.temperature).sum(dim=1)
        return -torch.log(pos_sim / neg_sim).mean()

    def _diversity_loss(self, probs: Tensor) -> Tensor:
        """
        Computes diversity loss encouraging codebook usage.

        Args:
            probs (Tensor): Codebook probabilities. Shape: (G, V)

        Returns:
            Tensor: Scalar loss.
        """
        entropy = torch.sum(probs * torch.log(probs + 1e-7), dim=-1)
        return torch.sum(entropy) / (self.num_groups * self.num_codevectors)

    def _sample_negatives(self, positives: Tensor, targets_per_batch: List[int]) -> Tensor:
        """
        Samples K negative examples for each positive.

        Args:
            positives (Tensor): Flattened positive samples. Shape: (N, D)
            targets_per_sample (List[int]): Number of targets per sample.

        Returns:
            Tensor: Negative samples. Shape: (N, K, D)
        """
        negatives = []
        start = 0

        for count in targets_per_batch:
            idx_range = torch.arange(count, device=positives.device)
            mask = torch.eye(count, device=positives.device).bool()
            candidate_indices = idx_range.repeat(count).view(count, -1)[~mask].view(count, -1)
            candidate_indices += start

            rand_rows = torch.arange(count).repeat_interleave(self.num_distractors)
            rand_cols = torch.tensor(
                [random.sample(range(count - 1), self.num_distractors) for _ in range(count)],
                device=positives.device
            ).view(-1)

            sampled = candidate_indices[rand_rows, rand_cols]
            negatives.append(positives[sampled])
            start += count

        return torch.cat(negatives).view(positives.size(0), self.num_distractors, -1)