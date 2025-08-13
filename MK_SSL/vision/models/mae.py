import torch
import torch.nn as nn
from typing import Tuple, Optional

from MK_SSL.vision.models.modules.mae_blocks import PatchEmbed, MAEEncoder, MAEDecoder
from MK_SSL.vision.models.modules.pos_embed import PosEmbed2D
from MK_SSL.vision.models.modules.losses.mae_loss import MAELoss
from MK_SSL.vision.models.utils.registry import register_method

class MAE(nn.Module):
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
        backbone: Optional[nn.Module] = None,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_chans, embed_dim)
        self.mask_ratio = mask_ratio
        self.encoder = backbone or MAEEncoder(embed_dim, depth, num_heads, mlp_ratio)
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

        # register decoder fixed pos-embed as buffer inside decoder
        dec_pos = self.pos_embed_dec.pos_embed
        self.decoder.register_buffer("pos_embed", dec_pos, persistent=False)
        self.to(device)

    def random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        assert 0 < len_keep < N, f"len_keep={len_keep}, N={N}, mask_ratio={mask_ratio}"

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones([B, N], device=x.device, dtype=x.dtype)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        f"Image Size                        : {getattr(model.patch_embed, 'img_size', 'N/A')}\n"
        f"Patch Size                        : {model.patch_embed.patch_size} x {model.patch_embed.patch_size}\n"
        f"Number of Patches                 : {model.patch_embed.num_patches}\n"
        f"Input Channels                    : 3\n"
        f"Encoder Embedding Dimension       : {getattr(model.encoder, 'blocks', [None])[0].attn.qkv.in_features if hasattr(model.encoder, 'blocks') else 'Custom Backbone'}\n"
        f"Encoder Depth                     : {len(model.encoder.blocks) if hasattr(model.encoder, 'blocks') else 'Custom Backbone'}\n"
        f"Encoder Heads                     : {model.encoder.blocks[0].attn.num_heads if hasattr(model.encoder, 'blocks') else 'Custom Backbone'}\n"
        f"Decoder Dimension                 : {model.decoder.pred.in_features}\n"
        f"Decoder Depth                     : {len(model.decoder.blocks)}\n"
        f"Decoder Attention Heads           : {model.decoder.blocks[0].attn.num_heads}\n"
        f"Decoder MLP Ratio                 : {model.decoder.blocks[0].mlp.fc1.out_features / model.decoder.blocks[0].mlp.fc1.in_features}\n"
        f"Mask Ratio                        : {model.mask_ratio}\n"
        f"Loss                              : Pixel Reconstruction (MAELoss)\n"
    ),
)
