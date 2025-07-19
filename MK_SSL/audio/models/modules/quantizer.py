import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class GumbelVectorQuantizer(nn.Module):
    """
    Gumbel Vector Quantizer used in wav2vec 2.0 pretraining.

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

        # Project input features to G * V logits
        self.gumbel_logits_proj = nn.Linear(dim, num_groups * num_entries_per_codebook)

        # Codebook of shape (1, G, V, D/G)
        self.codebook = nn.Parameter(
            torch.FloatTensor(1, num_groups, num_entries_per_codebook, dim // num_groups)
        )
        nn.init.uniform_(self.codebook, -1.0 / dim, 1.0 / dim)

        self.codevector_proj = nn.Linear(dim, code_vector_size)

    @staticmethod
    def _compute_perplexity(probs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs (torch.Tensor): shape (B, L, G, V)
            lengths (torch.Tensor): shape (B)
        Returns:
            torch.Tensor: shape (G, V)
        """
        mask = torch.arange(probs.size(1), device=probs.device).unsqueeze(0) < lengths.unsqueeze(-1)
        probs = probs[mask]  # Keep only valid timesteps
        num_values = probs.size(0)
        perplexity = probs.sum(0) / num_values
        return perplexity

    def forward(self, hidden_states: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            hidden_states (torch.Tensor): shape (B, L, D)
            lengths (torch.Tensor): shape (B)
        Returns:
            tuple:
                projected_vectors (torch.Tensor): shape (B, L, code_vector_size)
                perplexity (torch.Tensor): shape (G, V)
        """
        batch_size, length, _ = hidden_states.shape

        logits = self.gumbel_logits_proj(hidden_states)  # (B, L, G*V)
        logits = logits.view(batch_size, length, self.groups, self.num_entries_per_codebook)  # (B, L, G, V)

        gumbel_out = F.gumbel_softmax(logits.float(), tau=self.gumbel_temp, hard=True).type_as(logits)  # (B, L, G, V)

        soft_probs = torch.softmax(logits.float(), dim=-1)  # (B, L, G, V)
        perplexity = self._compute_perplexity(soft_probs, lengths)

        gumbel_out = gumbel_out.unsqueeze(-1)  # (B, L, G, V, 1)
        code_vectors = torch.sum(gumbel_out * self.codebook, dim=-2)  # (B, L, G, D/G)
        code_vectors = code_vectors.contiguous().view(batch_size, length, self.dim)  # (B, L, D)

        projected_vectors = self.codevector_proj(code_vectors)  # (B, L, code_vector_size)

        return projected_vectors, perplexity
