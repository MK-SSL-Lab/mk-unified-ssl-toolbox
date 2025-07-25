import torch
import torch.nn as nn
from typing import Tuple

from MK_SSL.vision.models.modules.mae_blocks import PatchEmbed, MAEEncoder, MAEDecoder
from MK_SSL.vision.models.modules.pos_embed import PosEmbed2D
from MK_SSL.vision.models.modules.losses.mae_loss import MAELoss
from MK_SSL.vision.models.utils.registry import register_method


class MAE(nn.Module):
    """
    Masked Autoencoder (MAE) following He et al. (2021).

    This model consists of:
      - A Vision Transformer (ViT) encoder that processes only the visible patches.
      - A lightweight Transformer decoder that reconstructs the original image patches
        from encoded visible tokens and mask tokens.
      - A random masking mechanism that hides a portion of image patches.

    Attributes:
        patch_embed (PatchEmbed): Module to split an image into non-overlapping patches and embed them.
        mask_ratio (float): Ratio of patches to mask (e.g., 0.75 means 75% of patches are masked).
        encoder (MAEEncoder): Transformer encoder processing visible tokens.
        decoder (MAEDecoder): Transformer decoder reconstructing the full image.
        pos_embed_enc (PosEmbed2D): Sine-cosine positional embedding for encoder tokens.
        pos_embed_dec (PosEmbed2D): Sine-cosine positional embedding for decoder tokens.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_dim: int = 512,
        decoder_depth: int = 8,
        decoder_heads: int = 8,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
        **kwargs,
    ) -> None:
        """
        Initializes the MAE model.

        Args:
            image_size (int, optional): Input image size (assumes square). Defaults to 224.
            patch_size (int, optional): Patch size (square patches). Defaults to 16.
            in_chans (int, optional): Number of input channels (e.g., 3 for RGB). Defaults to 3.
            embed_dim (int, optional): Embedding dimension for encoder tokens. Defaults to 768.
            depth (int, optional): Number of transformer blocks in the encoder. Defaults to 12.
            num_heads (int, optional): Number of attention heads in the encoder. Defaults to 12.
            decoder_dim (int, optional): Embedding dimension for decoder tokens. Defaults to 512.
            decoder_depth (int, optional): Number of transformer blocks in the decoder. Defaults to 8.
            decoder_heads (int, optional): Number of attention heads in the decoder. Defaults to 8.
            mlp_ratio (float, optional): MLP expansion ratio in transformer blocks. Defaults to 4.0.
            mask_ratio (float, optional): Ratio of patches to mask. Defaults to 0.75.
        """
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_chans, embed_dim)
        self.mask_ratio = mask_ratio

        self.encoder = MAEEncoder(embed_dim, depth, num_heads, mlp_ratio)
        self.decoder = MAEDecoder(
            embed_dim,
            decoder_dim,
            decoder_depth,
            decoder_heads,
            mlp_ratio,
            patch_size,
            in_chans,
        )

        self.pos_embed_enc = PosEmbed2D(embed_dim, self.patch_embed.grid_size)
        self.pos_embed_dec = PosEmbed2D(decoder_dim, self.patch_embed.grid_size)
        self.decoder.pos_embed = self.pos_embed_dec.pos_embed

    def random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies random masking to a sequence of patch embeddings.

        Args:
            x (torch.Tensor): Patch embeddings of shape (B, N, D),
                where B = batch size, N = number of patches, D = embedding dimension.
            mask_ratio (float): Ratio of patches to mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - x_masked (torch.Tensor): Visible (unmasked) tokens, shape (B, N_visible, D).
                - mask (torch.Tensor): Binary mask of shape (B, N), 1 for masked patches, 0 for visible.
                - ids_restore (torch.Tensor): Indices to restore original patch order, shape (B, N).
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MAE.

        Args:
            imgs (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - pred (torch.Tensor): Predicted patches, shape (B, N, patch_dim).
                - mask (torch.Tensor): Binary mask indicating which patches were masked, shape (B, N).
        """
        x = self.patch_embed(imgs)
        x = self.pos_embed_enc(x)
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        x = self.encoder(x)
        pred = self.decoder(x, ids_restore)
        return pred, mask


register_method(
    name="mae",
    model_cls=MAE,
    loss=MAELoss,
    transformation=None,
    logs=lambda model, loss: (
        "\n"
        "---------------- MAE Configuration ----------------\n"
        f"Image Size                        : {getattr(model.patch_embed, 'image_size', 'N/A')}\n"
        f"Patch Size                        : {model.patch_embed.patch_size} x {model.patch_embed.patch_size}\n"
        f"Number of Patches                 : {model.patch_embed.num_patches}\n"
        f"Input Channels                    : 3\n"
        f"Encoder Embedding Dimension       : "
        f"{getattr(model.encoder.blocks[0].self_attn, 'embed_dim', 'N/A') if hasattr(model.encoder.blocks[0], 'self_attn') else 'N/A'}\n"
        f"Encoder Depth                     : {len(model.encoder.blocks)}\n"
        f"Encoder Heads                     : "
        f"{getattr(model.encoder.blocks[0].self_attn, 'num_heads', 'N/A') if hasattr(model.encoder.blocks[0], 'self_attn') else 'N/A'}\n"
        f"Decoder Dimension                 : {model.decoder.pred.in_features}\n"
        f"Decoder Depth                     : {len(model.decoder.blocks)}\n"
        f"Decoder Attention Heads           : "
        f"{getattr(model.decoder.blocks[0].self_attn, 'num_heads', 'N/A') if hasattr(model.decoder.blocks[0], 'self_attn') else 'N/A'}\n"
        f"Decoder MLP Ratio                 : "
        f"{model.decoder.blocks[0].linear1.out_features / model.decoder.blocks[0].linear1.in_features if hasattr(model.decoder.blocks[0], 'linear1') else 'N/A'}\n"
        f"Mask Ratio                        : {model.mask_ratio}\n"
        f"Loss                              : Pixel Reconstruction (MAELoss)\n"
    ),
)
