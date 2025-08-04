import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class ProjectionHead(nn.Sequential):
    """
    Projection MLP head for transforming encoder outputs.

    Args:
        blocks: List of tuples, each defining one layer of the MLP.
                Each tuple is (in_features, out_features, batch_norm_layer, activation_layer).
    """

    def __init__(
        self, blocks: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]]
    ):
        layers = []
        for input_dim, output_dim, batch_norm, non_linearity in blocks:
            use_bias = not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        super().__init__(*layers)



class Wav2ClipProjectionHead(ProjectionHead):
    """
    Optional projection MLP head to transform encoder outputs.

    Args:
        input_dim (int): Dimension of input features.
        output_dim (int): Dimension of output features.
        hidden_dim (int, optional): Hidden layer size. If None, uses a single linear layer.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None):
        self.input_dim = input_dim
        self.output_dim = output_dim
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
