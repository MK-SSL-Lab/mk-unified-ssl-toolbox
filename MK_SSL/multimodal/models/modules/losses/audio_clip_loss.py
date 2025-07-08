import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioCLIPLoss(nn.Module):
    """
    Symmetric Contrastive Loss for AudioCLIP.

    Computes symmetric cross-entropy loss over the cosine similarity matrices
    between modality pairs (audio-text, audio-image, text-image) as used in AudioCLIP.

    For each modality pair with similarity matrix S:
        L = 0.5 * (CE(S, target) + CE(S.T, target))

    Only non-None similarity matrices are included in the final loss.

    Args:
        temperature_eps (float): Small epsilon to ensure numerical stability.
    """

    def __init__(self, temperature_eps: float = 1e-8, **kwargs):
        super().__init__()
        self.temperature_eps = temperature_eps

    def forward(
        self,
        sim_text_audio: torch.Tensor,
        sim_text_image: torch.Tensor,
        sim_audio_image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sim_text_audio (Tensor, optional): Similarity matrix between text and audio (B, B).
            sim_text_image (Tensor, optional): Similarity matrix between text and image (B, B).
            sim_audio_image (Tensor, optional): Similarity matrix between audio and image (B, B).

        Returns:
            Tensor: Scalar contrastive loss value (averaged over available pairs).
        """
        loss = 0.0
        count = 0

        def compute_symmetric_ce(sim_matrix: torch.Tensor) -> torch.Tensor:
            if sim_matrix.size(0) != sim_matrix.size(1):
                raise ValueError(f"Expected square similarity matrix, got {sim_matrix.shape}")
            targets = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
            loss_1 = F.cross_entropy(sim_matrix, targets)
            loss_2 = F.cross_entropy(sim_matrix.T, targets)
            return 0.5 * (loss_1 + loss_2)

        if sim_text_audio is not None:
            loss += compute_symmetric_ce(sim_text_audio)
            count += 1

        if sim_text_image is not None:
            loss += compute_symmetric_ce(sim_text_image)
            count += 1

        if sim_audio_image is not None:
            loss += compute_symmetric_ce(sim_audio_image)
            count += 1

        if count == 0:
            raise ValueError("At least one similarity matrix must be provided.")

        return loss / count
