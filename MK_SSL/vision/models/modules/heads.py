import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ProjectionHead(nn.Module):
    """Base class for all projection and prediction MLP heads.

    Args:
        blocks (List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]]):
            Each tuple defines a block as (in_features, out_features, batch_norm, activation).
    """

    def __init__(self, blocks: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]]):
        super().__init__()
        layers = []
        for in_dim, out_dim, bn_layer, act_layer in blocks:
            use_bias = not bool(bn_layer)
            layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
            if bn_layer:
                layers.append(bn_layer)
            if act_layer:
                layers.append(act_layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SimCLRProjectionHead(ProjectionHead):
    """Projection head used in SimCLR (v1 and v2)."""

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        num_layers: int = 2,
        batch_norm: bool = True,
        **kwargs,
    ):
        layers: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]] = []

        layers.append((input_dim, hidden_dim,
                       nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                       nn.ReLU(inplace=True)))
        for _ in range(2, num_layers):
            layers.append((hidden_dim, hidden_dim,
                           nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                           nn.ReLU(inplace=True)))
        layers.append((hidden_dim, output_dim,
                       nn.BatchNorm1d(output_dim) if batch_norm else None,
                       None))
        super().__init__(layers)


class BarlowTwinsProjectionHead(ProjectionHead):
    """Projection head used for Barlow Twins."""

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 8192, output_dim: int = 8192):
        super().__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)),
            (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)),
            (hidden_dim, output_dim, None, None),
        ])


class BYOLProjectionHead(ProjectionHead):
    """Projection head used for BYOL."""

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 4096, output_dim: int = 256):
        super().__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class BYOLPredictionHead(ProjectionHead):
    """Prediction head used for BYOL."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 4096, output_dim: int = 256):
        super().__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class SimSiamProjectionHead(ProjectionHead):
    """Projection head used for SimSiam."""

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 2048):
        super().__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, nn.BatchNorm1d(output_dim, affine=False), None),
        ])


class SimSiamPredictionHead(ProjectionHead):
    """Prediction head used for SimSiam."""

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, output_dim: int = 2048):
        super().__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class SwAVProjectionHead(ProjectionHead):
    """Projection head used for SwAV."""

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 128):
        super().__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class DINOProjectionHead(nn.Module):
    """Projection head for DINO: supports GELU, optional BN, and weight-normalized output.

    Args:
        input_dim (int): Input dimension from the backbone.
        output_dim (int): Final projection output size. Default = 256.
        use_bn (bool): Whether to use batch norm between MLP layers.
        norm_last_layer (bool): Whether to freeze norm of the final layer.
        num_layers (int): Number of layers in the MLP. Default = 3.
        hidden_dim (int): Hidden dimension size. Default = 2048.
        bottleneck_dim (int): Projection bottleneck before final norm. Default = 256.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        use_bn: bool = False,
        norm_last_layer: bool = True,
        num_layers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        layers = []

        if num_layers == 1:
            self.mlp = nn.Linear(input_dim, bottleneck_dim)
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())

            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())

            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, output_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1.0)

        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
