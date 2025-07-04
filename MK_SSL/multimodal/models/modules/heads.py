import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class ProjectionHead(nn.Module):
    """
    Base class for all projection and prediction heads.

    Args:
        blocks:
            List of tuples, each denoting one block of the projection head MLP.
            Each tuple reads (in_features, out_features, batch_norm_layer,
            non_linearity_layer).
    """

    def __init__(
        self, blocks: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]]
    ):
        super().__init__()

        layers = []
        for input_dim, output_dim, batch_norm, non_linearity in blocks:
            use_bias = not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Wav2ClipProjectionHead(ProjectionHead):
    """
    Optional projection MLP head to transform encoder outputs.

    Args:
        input_dim (int): Dimension of input features.
        output_dim (int): Dimension of output features.
        hidden_dim (int, optional): Hidden layer size. If None, uses a single linear layer.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None):
        if hidden_dim is None:
            blocks = [
                (input_dim, output_dim, None, None)
            ]
        else:
            blocks = [
                (input_dim, hidden_dim, None, nn.ReLU(inplace=True)),
                (hidden_dim, output_dim, None, None),
            ]
        super().__init__(blocks)
