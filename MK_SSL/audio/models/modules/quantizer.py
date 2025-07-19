import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class GumbelVectorQuantizer(nn.Module):
    """Gumbel Vector Quantizer for wav2vec 2.0 pretraining.

    This module discretizes latent representations using Gumbel-softmax sampling,
    producing both quantized vectors and codebook usage statistics for the diversity loss.

    Args:
        dim (int): Input dimension to quantizer (should match feature dim from CNN).
        num_entries_per_codebook (int): Number of codebook entries per group (e.g., 320).
        code_vector_size (int): Dimension of the output code vectors.
        temp (float): Initial temperature for Gumbel-softmax.
        num_groups (int): Number of groups to split channels into.
        combine_groups (bool): If True, output is reshaped to (B, T, dim).
    """

    def __init__(
        self,
        dim: int,
        num_entries_per_codebook: int,
        code_vector_size: int,
        temp: float = 2.0,
        num_groups: int = 2,
        combine_groups: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_entries_per_codebook = num_entries_per_codebook
        self.groups = num_groups
        self.combine_groups = combine_groups
        self.code_vector_size = code_vector_size
        self.gumbel_temp = temp

        self.gumbel_logits_proj = nn.Linear(dim, num_groups * num_entries_per_codebook)

        self.codebook = nn.Parameter(
            torch.FloatTensor(1, num_groups, num_entries_per_codebook, dim // num_groups)
        )
        nn.init.uniform_(self.codebook, -1.0 / dim, 1.0 / dim)

        self.codevector_proj = nn.Linear(dim, code_vector_size)

    @staticmethod
    def _compute_avg_probs(probs: torch.Tensor, lengths: torch.Tensor) -> Tensor:
        """Compute average codebook usage probabilities across all valid frames.

        Args:
            probs (Tensor): Softmax probabilities of shape (B, L, G, V).
            lengths (Tensor): Valid lengths of shape (B,).

        Returns:
            Tensor: Averaged probabilities of shape (G, V).
        """
        mask = torch.arange(probs.size(1), device=probs.device).unsqueeze(0) < lengths.unsqueeze(-1)
        probs = probs[mask]  # Keep only valid timesteps
        return probs.mean(dim=0)  # (G, V)

    @staticmethod
    def _compute_perplexity(avg_probs: torch.Tensor) -> Tensor:
        """Compute codebook perplexity.

        Args:
            avg_probs (Tensor): Averaged probabilities of shape (G, V).

        Returns:
            Tensor: Codebook perplexity scalar.
        """
        entropy = -(avg_probs * (avg_probs + 1e-7).log()).sum(dim=-1).mean()
        return entropy.exp()

    def forward(self, hidden_states: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass through quantizer.

        Args:
            hidden_states (Tensor): Input latent features of shape (B, L, D).
            lengths (Tensor): Valid lengths of shape (B,).

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - Quantized and projected vectors of shape (B, L, code_vector_size).
                - Average codebook probabilities (G, V).
                - Codebook perplexity (scalar).
        """
        B, L, _ = hidden_states.shape

        logits = self.gumbel_logits_proj(hidden_states)  # (B, L, G*V)
        logits = logits.view(B, L, self.groups, self.num_entries_per_codebook)  # (B, L, G, V)

        gumbel_out = F.gumbel_softmax(logits.float(), tau=self.gumbel_temp, hard=True).type_as(logits)  # (B, L, G, V)
        soft_probs = torch.softmax(logits.float(), dim=-1)  # (B, L, G, V)

        avg_probs = self._compute_avg_probs(soft_probs, lengths)  # (G, V)
        perplexity = self._compute_perplexity(avg_probs)  # scalar

        gumbel_out = gumbel_out.unsqueeze(-1)  # (B, L, G, V, 1)
        code_vectors = (gumbel_out * self.codebook).sum(dim=-2)  # (B, L, G, D/G)
        code_vectors = code_vectors.contiguous().view(B, L, self.dim)  # (B, L, D)

        projected_vectors = self.codevector_proj(code_vectors)  # (B, L, code_vector_size)

        return projected_vectors, avg_probs, perplexity
