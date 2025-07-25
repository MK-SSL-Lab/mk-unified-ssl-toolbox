import torch
import torch.nn as nn

from MK_SSL.vision.models.modules.mae_blocks import PatchEmbed, MAEEncoder
from MK_SSL.vision.models.modules.mae_blocks import MAEDecoder
from MK_SSL.vision.models.modules.pos_embed import PosEmbed2D

class MAE(nn.Module):
    """Masked Autoencoder following He et al. (2021)."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, decoder_dim=512, decoder_depth=8,
                 decoder_heads=8, mlp_ratio=4., mask_ratio=0.75):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.mask_ratio = mask_ratio

        self.encoder = MAEEncoder(embed_dim, depth, num_heads, mlp_ratio)
        self.decoder = MAEDecoder(embed_dim, decoder_dim, decoder_depth, decoder_heads,
                                  mlp_ratio, patch_size, in_chans)

        self.pos_embed_enc = PosEmbed2D(embed_dim, self.patch_embed.grid_size)
        self.pos_embed_dec = PosEmbed2D(decoder_dim, self.patch_embed.grid_size)
        self.decoder.pos_embed = self.pos_embed_dec.pos_embed

    def random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward(self, imgs):
        x = self.patch_embed(imgs)
        x = self.pos_embed_enc(x)
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        x = self.encoder(x)
        pred = self.decoder(x, ids_restore)
        return pred, mask
