
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from .mae_blocks import PatchEmbed, PositionalEncoding2D, MAEEncoder, MAEDecoder, Patchify, Unpatchify


class MAE(nn.Module):
    def __init__(
        self,
        encoder,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        decoder_dim=512,
        decoder_depth=8,
        decoder_heads=8,
        mlp_ratio=4.0,
        mask_ratio=0.75
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed_enc = PositionalEncoding2D(embed_dim, img_size // patch_size)
        self.encoder = MAEEncoder(encoder)

        self.decoder = MAEDecoder(
            dim=embed_dim,
            decoder_dim=decoder_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
            mlp_ratio=mlp_ratio
        )
        self.decoder.pos_embed = PositionalEncoding2D(decoder_dim, img_size // patch_size).pos_embed

        self.patchify = Patchify(patch_size)
        self.unpatchify = Unpatchify(patch_size, img_size)

        self.mask_ratio = mask_ratio
        self.head = nn.Linear(decoder_dim, patch_size * patch_size * in_chans, bias=True)

    def random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)  # noise in [0, 1)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        return x_masked, ids_restore, ids_keep

    def forward(self, imgs):
        x = self.patch_embed(imgs)  # (B, N, D)
        x = self.pos_embed_enc(x)

        x_masked, ids_restore, ids_keep = self.random_masking(x, self.mask_ratio)
        x_encoded = self.encoder(x_masked)
        x_decoded = self.decoder(x_encoded, ids_restore)

        pred = self.head(x_decoded)  # (B, N, patch_dim)
        pred_img = self.unpatchify(pred)

        return pred, pred_img, ids_restore

    def encode(self, imgs):
        x = self.patch_embed(imgs)
        x = self.pos_embed_enc(x)
        x_masked, ids_restore, ids_keep = self.random_masking(x, self.mask_ratio)
        x_encoded = self.encoder(x_masked)
        return x_encoded
