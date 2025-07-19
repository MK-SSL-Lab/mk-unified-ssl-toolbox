import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class GumbelVectorQuantizer(nn.Module):
    """
    Gumbel Vector Quantizer used in wav2vec 2.0 pretraining.

    This quantizer learns discrete latent codes using Gumbel-softmax sampling.

    Args:
        dim (int): Input dimension to quantizer (should match feature dim from CNN).
        num_entries_per_codebook (int): Total number of codebook entries (e.g., 320).
        temp (float): Initial temperature for Gumbel-softmax.
        groups (int): Number of groups to split channels into.
        combine_groups (bool): If True, output is reshaped to (B, T, dim). If False, returned as-is.
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

        self.gumbel_logits_proj = nn.Linear(self.dim, num_groups * self.num_entries_per_codebook)

        self.codebook = nn.Parameter(
            torch.FloatTensor(1, num_groups, self.num_entries_per_codebook, self.dim // num_groups)
        )

        nn.init.uniform_(self.codebook, -1.0 / self.dim, 1.0 / self.dim)

        self.codevector_proj = nn.Linear(self.dim, self.code_vector_size)


        self.gumbel_temp = temp


    @staticmethod
    def _compute_perplexity(probs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs (torch.Tensor): shape (B, L, G, V)
            lengths (torch.Tensor): shape (B)

        Returns:
            torch.Tensor: shape (G, V)
        """
        where_calculate_probs = torch.arange(probs.size(1), device=probs.device).unsqueeze(0) < lengths.unsqueeze(-1)
        probs = probs[where_calculate_probs == 1]
        num_values = probs.size(0)
        perplexity = probs.sum(0) / num_values
        return perplexity

    def forward(self, hidden_states: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            hidden_states (torch.Tensor): shape (B, L, D1)
            lengths (torch.Tensor): shape (B)

        Returns:
            tuple:
                code_vectors (torch.Tensor): shape (B, L, code_vector_size)
                perplexity (torch.Tensor): shape (G, V)
        """
        batch_size, length, _ = hidden_states.shape

        logits = self.gumbel_logits_proj(batch_size) # (B, L, G*V)
        logits = logits.view(batch_size, length, self.num_groups, self.num_vectors)  # (B, L, G, V)

        # Sample from Gumbel-softmax
        gumbel_out = nn.functional.gumbel_softmax(
            logits.float(), tau=self.temperature, hard=True
        ).type_as(logits)  # (B, L, G, V)

        soft_probs = torch.softmax(logits.float(), dim=-1)  # for perplexity
        perplexity = self._compute_perplexity(soft_probs, lengths)

        # Compute quantized code vectors from one-hot indices
        gumbel_out = gumbel_out.unsqueeze(-1)  # (B, L, G, V, 1)
        codebook = self.codebook  # (1, G, V, D/G)
        code_vectors = torch.sum(gumbel_out * codebook, dim=-2)  # (B, L, G, D/G)

        # Reshape to (B, L, D) where D = dim
        code_vectors = code_vectors.contiguous().view(batch_size, length, self.dim)

        # Final projection to (B, L, code_vector_size)
        projected_vectors = self.codevector_proj(code_vectors)

        return projected_vectors, perplexity



