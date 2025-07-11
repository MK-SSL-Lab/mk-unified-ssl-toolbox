import torch
import torch.nn as nn
from einops import rearrange

class MAEDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,           # encoder output dim
        decoder_dim: int = 512,         # internal dim of decoder
        depth: int = 8,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        input_dim: tuple = (128, 1024),  # (freq, time)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.num_patches = (input_dim[0] // patch_size) * (input_dim[1] // patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        self.proj = nn.Linear(embed_dim, decoder_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))

        self.blocks = nn.Sequential(*[
            ViTBlock(decoder_dim, heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)

        patch_dim = patch_size * patch_size  # if input is image-like (e.g., spectrogram patch)
        self.head = nn.Linear(decoder_dim, patch_dim)