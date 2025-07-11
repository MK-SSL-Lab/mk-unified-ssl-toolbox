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

    def forward(self, x_visible, ids_restore):
        """
        Args:
            x_visible (B, N_visible, D_enc): encoded visible tokens
            ids_restore (B, N_total): indices to restore masked tokens to original positions
        Returns:
            x_rec (B, N_total, patch_dim): reconstructed patch vectors
        """
        x_proj = self.proj(x_visible)

        B, N, D = x_proj.shape
        N_total = ids_restore.shape[1]

        # Prepare mask tokens and concatenate with encoded ones
        mask_tokens = self.mask_token.expand(B, N_total - N, D)
        x_ = torch.cat([x_proj, mask_tokens], dim=1)

        # Unshuffle to original order
        idx = ids_restore.unsqueeze(-1).expand(-1, -1, D)
        x_ = torch.gather(x_, dim=1, index=idx)  # (B, N_total, D)

        # Add positional embedding
        x_ = x_ + self.pos_embed

        # Decode
        x_ = self.blocks(x_)
        x_ = self.norm(x_)
        x_rec = self.head(x_)  # (B, N_total, patch_dim)

        return x_rec
