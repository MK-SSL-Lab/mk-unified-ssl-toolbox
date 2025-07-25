import torch
import math
import torch.nn as nn
from einops import rearrange, repeat



class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.

    This module splits an input image into non-overlapping patches and projects
    each patch into a vector (embedding) using a convolutional layer.

    Args:
        img_size (int): Height and width of the input image (must be square). Default is 224.
        patch_size (int): Height and width of each patch (must divide img_size evenly). Default is 16.
        in_chans (int): Number of input channels (e.g., 3 for RGB images). Default is 3.
        embed_dim (int): Dimension of the patch embeddings. Default is 768.

    Inputs:
        x (Tensor): Input image of shape (B, C, H, W)

    Returns:
        Tensor: Patch embeddings of shape (B, num_patches, embed_dim),
                where num_patches = (img_size // patch_size) ** 2
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.in_chans = in_chans

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PositionalEncoding2D(nn.Module):
    """
    2D Learnable Positional Encoding for Vision Transformers.

    This module adds a learnable 2D positional embedding to a sequence of patch embeddings.

    Args:
        dim (int): Dimension of the embeddings.
        grid_size (int): Number of patches per side (e.g., 14 for 14x14 patch grid).

    Inputs:
        x (Tensor): Input tensor of shape (B, N, dim), where N = grid_size ** 2

    Returns:
        Tensor: Output tensor with positional encoding added, same shape as input.
    """
    def __init__(self, dim, grid_size):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size ** 2, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return x + self.pos_embed


class MAEEncoder(nn.Module):
    """
    Wrapper for the encoder component of a Masked Autoencoder (MAE).

    This class wraps any transformer-style encoder module to be used in the MAE framework.

    Args:
        encoder (nn.Module): A transformer encoder module (e.g., ViT encoder block).

    Inputs:
        x (Tensor): Input patch embeddings of shape (B, N_visible, dim)

    Returns:
        Tensor: Encoded visible patches of shape (B, N_visible, dim_out)
    """
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)





class MAEDecoder(nn.Module):

    """
    Masked Autoencoder (MAE) Decoder Module.

    This module reconstructs the full set of image patches (including masked ones)
    from the encoded visible patches using a transformer-based decoder architecture.

    Args:
        dim (int): Input embedding dimension from the encoder.
        decoder_dim (int): Dimension of decoder embeddings. Default is 512.
        depth (int): Number of transformer blocks in the decoder. Default is 8.
        num_heads (int): Number of attention heads in each transformer block. Default is 8.
        mlp_ratio (float): MLP hidden dimension expansion ratio. Default is 4.0.

    Inputs:
        x_visible (Tensor): Encoded visible tokens of shape (B, N_visible, dim).
        ids_restore (Tensor): Indices to restore the original full sequence, shape (B, N_total).

    Returns:
        Tensor: Reconstructed full sequence of shape (B, N_total, decoder_dim).
    """



    def __init__(self, dim, decoder_dim=512, depth=8, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.pos_embed = None  # added later externally
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder_dim = decoder_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.decoder_embed = nn.Linear(dim, decoder_dim, bias=True)
        self.decoder_blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=decoder_dim,
                nhead=num_heads,
                dim_feedforward=int(decoder_dim * mlp_ratio),
                batch_first=True
            ) for _ in range(depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)

    def forward(self, x_visible, ids_restore):
        # Project encoder features to decoder dimension
        x_visible = self.decoder_embed(x_visible)  # (B, N_visible, D)
        B, N_visible, D = x_visible.shape
        N_total = ids_restore.shape[1]  # total patches

        # Prepare full sequence of mask tokens
        x_full = self.mask_token.repeat(B, N_total, 1)

        # Place visible tokens into their correct positions
        ids_keep = ids_restore[:, :N_visible]  # indices of visible tokens
        ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, D)
        x_full.scatter_(1, ids_keep_expanded, x_visible)

        # Add position embeddings
        x_full = x_full + self.pos_embed  # (B, N_total, D)

        # Pass through decoder transformer blocks
        x_full = self.decoder_blocks(x_full)
        x_full = self.decoder_norm(x_full)

        # Project back to patch pixel space
        x_full = self.decoder_pred(x_full)  # Add this layer (Linear D->patch_dim)

        return x_full



class Patchify(nn.Module):

    """
    Converts input images into a sequence of flattened patches.

    Args:
        patch_size (int): Size of each square patch (patch will be of shape patch_size x patch_size).

    Inputs:
        imgs (Tensor): Input tensor of shape (B, C, H, W), where H and W must be divisible by patch_size.

    Returns:
        Tensor: Patchified image of shape (B, N, patch_dim) where
                N = (H * W) / (patch_size ** 2) and
                patch_dim = patch_size * patch_size * C.
    """



    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, imgs):
        # (B, C, H, W) -> (B, N, patch_size*patch_size*C)
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H == W and H % p == 0
        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).flatten(1, 2).flatten(2)
        return x


class Unpatchify(nn.Module):

    """
    Reconstructs full images from a sequence of flattened patches.

    Args:
        patch_size (int): Size of each square patch.
        img_size (int): Size (height/width) of the original square image.

    Inputs:
        x (Tensor): Sequence of flattened patches of shape (B, N, patch_dim),
                    where patch_dim = patch_size * patch_size * C.

    Returns:
        Tensor: Reconstructed image of shape (B, C, H, W), where H = W = img_size.
    """


    def __init__(self, patch_size, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size

    def forward(self, x):
        # (B, N, patch_dim) -> (B, C, H, W)
        p = self.patch_size
        B, N, patch_dim = x.shape
        C = patch_dim // (p * p)
        h = w = self.img_size // p
        x = x.reshape(B, h, w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, self.img_size, self.img_size)
        return x




