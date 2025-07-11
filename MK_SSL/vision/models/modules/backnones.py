import torch
import torch.nn as nn
from einops import rearrange

from MK_SSL.vision.models.modules import PatchEmbed


class TransformerBlock(nn.Module):
    """
    A single block of the Vision Transformer.

    Args:
        dim (int): Dimensionality of input and output.
        heads (int): Number of attention heads.
        mlp_dim (int): Dimensionality of hidden layer in the MLP.
        dropout (float): Dropout rate.
    """
    def __init__(self, dim, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x



class MAEVisionTransformer(nn.Module):
    """
    MAE Vision Transformer backbone supporting different model variants.

    Args:
        variant (str): One of 'vit-b', 'vit-l', 'vit-h' specifying model size.
        image_size (int): Input image size (assumes square).
        in_chans (int): Number of input image channels.
        dropout (float): Dropout probability.
        use_cls_token (bool): Whether to use a class token.
        num_classes (int): If > 0, adds a linear classification head.
    """
    def __init__(
        self,
        variant="vit-b",
        image_size=224,
        in_chans=3,
        dropout=0.,
        use_cls_token=False,
        num_classes=0
    ):
        super().__init__()

        model_config = self._get_config(self.variant)


        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=model_config["patch_size"],
            in_chans=in_chans,
            embed_dim=model_config["dim"]
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + int(use_cls_token), model_config["dim"]))
        self.dropout = nn.Dropout(dropout)
        self.use_cls_token = use_cls_token

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, model_config["dim"]))

        self.blocks = nn.ModuleList([
            TransformerBlock(model_config["dim"], model_config["heads"], model_config["mlp_dim"], dropout)
            for _ in range(model_config["depth"])
        ])
        self.norm = nn.LayerNorm(model_config["dim"])

        self.head = nn.Linear(model_config["dim"], num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        """
        Forward pass of the Vision Transformer.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            Tensor: Output embeddings or classification scores
        """
        x = self.patch_embed(x)

        if self.use_cls_token:
            B = x.size(0)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.use_cls_token:
            return self.head(x[:, 0])
        return x  # (B, N, D)
    
    def _get_config(self, variant):
        presets = {
            "vit-b": dict(patch_size=16, dim=768, depth=12, heads=12, mlp_dim=3072),
            "vit-l": dict(patch_size=16, dim=1024, depth=24, heads=16, mlp_dim=4096),
            "vit-h": dict(patch_size=14, dim=1280, depth=32, heads=16, mlp_dim=5120)
        }

        if variant not in presets:
            raise ValueError(f"Invalid variant: {variant}")
        return presets[variant]