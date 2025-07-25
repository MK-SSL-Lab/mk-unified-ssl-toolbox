import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class MAEBackbone(nn.Module):
    """
    Backbone for downstream tasks using the encoder from a pretrained MAE model.

    This class applies patch embedding, positional encoding, and the transformer encoder,
    returning mean-pooled token embeddings suitable for classification or regression tasks.

    Attributes:
        patch_embed (PatchEmbed): Patch embedding module from the MAE model.
        pos_embed (PosEmbed2D): Positional embedding module for encoder input.
        encoder (nn.Module): Encoder module (ViT backbone) from the MAE model.
    """

    def __init__(self, pretrained_model: nn.Module, freeze: bool = False) -> None:
        """
        Initialize MAEBackbone.

        Args:
            pretrained_model (nn.Module): Pretrained MAE model (from pretext phase).
            freeze (bool): Whether to freeze backbone parameters. Defaults to False.
        """
        super().__init__()
        self.patch_embed = pretrained_model.patch_embed
        self.pos_embed = pretrained_model.pos_embed_enc
        self.encoder = pretrained_model.encoder  # Uses our MAEEncoder

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, images: Tensor) -> Tensor:
        """
        Forward pass for extracting embeddings.

        Args:
            images (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Mean-pooled embeddings of shape (B, D).
        """
        x = self.patch_embed(images)  # (B, N, D)
        x = self.pos_embed(x)         # (B, N, D)
        x = self.encoder(x)           # (B, N, D)
        return x.mean(dim=1)          # mean pooling over tokens → (B, D)
