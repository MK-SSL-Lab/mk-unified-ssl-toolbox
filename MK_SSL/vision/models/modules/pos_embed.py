import torch
import torch.nn as nn

class PosEmbed2D(nn.Module):
    """
    2D Sine-Cosine Positional Embedding (Fixed).
    Args:
        dim (int): Embedding dimension.
        grid_size (int): Size of the patch grid.
        cls_token (bool): Whether to include a class token embedding.
    """

    def __init__(self, dim: int, grid_size: int, cls_token: bool = False):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        self.cls_token = cls_token

        pos_embed = self._build_2d_sincos_pos_embed(dim, grid_size, cls_token)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _build_2d_sincos_pos_embed(self, embed_dim: int, grid_size: int, cls_token: bool):
        grid_h = torch.arange(grid_size, dtype=torch.float32)
        grid_w = torch.arange(grid_size, dtype=torch.float32)
        grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
        grid = torch.stack(grid, dim=0)  # (2, grid_h, grid_w)
        pos_embed = self._get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

        if cls_token:
            pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)
        return pos_embed.unsqueeze(0)

    def _get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0
        emb_h = self._get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
        emb_w = self._get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
        return torch.cat([emb_h, emb_w], dim=1)

    def _get_1d_sincos_pos_embed(self, embed_dim, pos):
        omega = torch.arange(embed_dim // 2, dtype=torch.float32)
        omega /= embed_dim / 2
        omega = 1. / (10000 ** omega)
        pos = pos.flatten().unsqueeze(1)
        out = pos * omega.unsqueeze(0)
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        return torch.cat([emb_sin, emb_cos], dim=1)

    def forward(self, x: torch.Tensor):
        """
        Add fixed positional embedding.
        Args:
            x (Tensor): shape (B, N, D)
        """
        return x + self.pos_embed[:, :x.size(1), :]
