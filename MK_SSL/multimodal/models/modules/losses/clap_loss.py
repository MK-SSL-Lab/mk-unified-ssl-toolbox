import torch
import torch.nn as nn
import torch.nn.functional as F


class CLAPLoss(nn.Module):
    """
    Symmetric Contrastive Loss for CLAP (Contrastive Language-Audio Pretraining).
    Implements the symmetric cross-entropy loss from Section 2.1 of the CLAP paper.

    The loss is defined as:
        L = 0.5 * (L_text_to_audio + L_audio_to_text)
    where:
        L_k = -(1/N) * sum_i log softmax(similarity_matrix)[i, i]

    Args:
        temperature_eps (float): Small value to check for valid temperature.
    """

    def __init__(self, temperature_eps: float = 1e-8, **kwargs):
        super().__init__()
        self.temperature_eps = temperature_eps

    def forward(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            similarity_matrix (torch.Tensor): Cosine similarity matrix of shape (B, B),
                where diagonal entries are positive pairs and others are negatives.

        Returns:
            torch.Tensor: Scalar contrastive loss value.
        """
        if similarity_matrix.size(0) != similarity_matrix.size(1):
            raise ValueError(f"Expected square similarity matrix, got {similarity_matrix.shape}")

        batch_size = similarity_matrix.size(0)

        # Text-to-Audio direction
        logits_text = F.log_softmax(similarity_matrix, dim=1)  # (B, B)
        loss_text = -torch.diag(logits_text).mean()

        # Audio-to-Text direction
        logits_audio = F.log_softmax(similarity_matrix.T, dim=1)  # (B, B)
        loss_audio = -torch.diag(logits_audio).mean()

        loss = 0.5 * (loss_text + loss_audio)
        return loss
