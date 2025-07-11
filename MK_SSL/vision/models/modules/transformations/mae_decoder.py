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
              ids_restore (B, N_total): indices to restore full sequence
              ids_keep (B, N_visible): indices of the visible tokens
          Returns:
        """
        x_proj = self.proj(x_visible)
        B, N_visible, D = x_proj.shape
        N_total = ids_restore.shape[1]


        # 1. Create a full-length tensor filled with mask tokens
        x_full = self.mask_token.repeat(B, N_total, 1)
    
        # 2. Insert visible tokens back into their original positions
        # scatter_ works in-place
        ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, D)
        x_full.scatter_(dim=1, index=ids_keep_expanded, src=x_proj)
    
        # 3. Add positional embeddings
        x_full = x_full + self.pos_embed
    
        # 4. Decode the full sequence
        x_decoded = self.blocks(x_full)
        x_decoded = self.norm(x_decoded)
        x_rec = self.head(x_decoded)
    
        return x_rec
