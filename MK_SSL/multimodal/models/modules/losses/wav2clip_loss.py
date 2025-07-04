import torch
import torch.nn as nn
import torch.nn.functional as F


class Wav2ClipLoss(nn.Module):
    """
    Contrastive CX Loss for Wav2CLIP: L(f(Image), Audio) + L(Image, g(Audio)).

    Args:
        embedding_dim (int): Input embedding dimension.
        projection_dim (int): Output dimension after projection.
        temperature (float): Temperature scaling for logits.
    """

    def __init__(self, embedding_dim: int = 512, projection_dim: int = 512, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

        # f: projection from image to shared space
        self.image_proj = nn.Linear(embedding_dim, projection_dim)

        # g: projection from audio to shared space
        self.audio_proj = nn.Linear(embedding_dim, projection_dim)

    def forward(self, image_embeddings: torch.Tensor, audio_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute CX Loss:
        - L(f(Image), Audio) : project image, compare to raw audio
        - L(Image, g(Audio)) : raw image, compare to projected audio

        Args:
            image_embeddings (torch.Tensor): Shape (B, D)
            audio_embeddings (torch.Tensor): Shape (B, D)

        Returns:
            torch.Tensor: Scalar loss
        """
        # Apply projections to only one side per loss term
        image_proj = F.normalize(self.image_proj(image_embeddings), dim=-1)  # f(Image)
        audio_proj = F.normalize(self.audio_proj(audio_embeddings), dim=-1)  # g(Audio)

        raw_image = F.normalize(image_embeddings, dim=-1)
        raw_audio = F.normalize(audio_embeddings, dim=-1)

        targets = torch.arange(image_embeddings.size(0), device=image_embeddings.device)

        # L(f(Image), Audio): image is projected, audio is raw
        logits_image_to_audio = torch.matmul(image_proj, raw_audio.T) / self.temperature
        loss_image_to_audio = self.cross_entropy(logits_image_to_audio, targets)

        # L(Image, g(Audio)): image is raw, audio is projected
        logits_audio_to_image = torch.matmul(raw_image, audio_proj.T) / self.temperature
        loss_audio_to_image = self.cross_entropy(logits_audio_to_image, targets)

        return (loss_image_to_audio + loss_audio_to_image) / 2
