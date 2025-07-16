import torch
import torch.nn as nn
from torch import Tensor

class MAEBackbone(nn.Module):
    """
    Backbone model for downstream tasks using the encoder from a pretrained MAE model.

    This class applies patch embedding, positional encoding, and the transformer encoder,
    returning mean-pooled token embeddings suitable for classification or regression.

    Args:
        pretrained_model (nn.Module): The pretrained MAE model (pretext phase).
    """

    def __init__(self, pretrained_model: nn.Module):
        super().__init__()
        self.patch_embed = pretrained_model.patch_embed
        self.pos_embed = pretrained_model.pos_embed_enc
        self.encoder = pretrained_model.encoder.encoder.backbone  # Unwrap MAEVisionTransformer

        # Optional: freeze weights (up to user)
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, images: Tensor) -> Tensor:
        """
        Args:
            images (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Image embeddings (B, D)
        """
        x = self.patch_embed(images)     # (B, N, D)
        x = self.pos_embed(x)            # (B, N, D)
        x = self.encoder(x)              # (B, N, D)
        return x.mean(dim=1)             # mean pooling over tokens → (B, D)
