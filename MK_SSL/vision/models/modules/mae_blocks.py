import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MAEEncoder(nn.Module):
    """ViT Encoder for MAE."""
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, int(embed_dim * mlp_ratio), batch_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

class MAEDecoder(nn.Module):
    """Lightweight Transformer Decoder for MAE."""
    def __init__(self, embed_dim=768, decoder_dim=512, depth=8, num_heads=8, mlp_ratio=4., patch_size=16, in_chans=3):
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(decoder_dim, num_heads, int(decoder_dim * mlp_ratio), batch_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_size * patch_size * in_chans, bias=True)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pos_embed = None  # Set externally

    def forward(self, x, ids_restore):
        x = self.decoder_embed(x)
        B, N, D = x.shape
        N_total = ids_restore.shape[1]

        mask_tokens = self.mask_token.expand(B, N_total - N, -1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x_ = x_ + self.pos_embed[:, :N_total, :]

        for blk in self.blocks:
            x_ = blk(x_)
        x_ = self.norm(x_)
        return self.pred(x_)
