
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Type

from MK_SSL.vision.models.modules import PatchEmbed, PositionalEncoding2D, MAEEncoder, MAEDecoder, Patchify, Unpatchify
from MK_SSL.vision.models.modules import MAEVisionTransformer

class MAE(nn.Module):
    def __init__(
        self,
        backbone: nn.modules =None,
        variant:str = "vit-b",
        img_size: int =224,
        patch_size: int =16,
        in_chans: int =3,
        embed_dim:int =768,
        decoder_dim:int =512,
        decoder_depth:int =8,
        decoder_heads:int =8,
        mlp_ratio:int =4.0,
        mask_ratio:int =0.75,
        encoder_dropout : Optional[int] = 0,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed_enc = PositionalEncoding2D(embed_dim, img_size // patch_size)

        if backbone is None:
            backbone = MAEVisionTransformer(
                variant=variant,
                image_size=img_size,
                in_chans=in_chans,
                dropout=encoder_dropout,
                use_cls_token=False,
                num_classes=0
            )

        self.encoder = MAEEncoder(backbone)

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
