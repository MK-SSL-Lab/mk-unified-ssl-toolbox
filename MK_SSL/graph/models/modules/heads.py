import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class ProjectionHead(nn.Module):
    """
    Description:
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

    def forward(self, x: torch.Tensor):
        return self.layers(x)



class GraphCLProjectionHead(ProjectionHead):
    """
    Projection head for GraphCL.

    Paper spec: a small MLP projection head, typically **two layers**,
    with the contrastive loss applied on the projected embeddings `z`
    (not on encoder features `h`).

    * If ``num_layers == 1``  ➜  Linear → (LayerNorm)
    * If ``num_layers  >  1`` ➜  (Linear → (LayerNorm) → ReLU)×(num_layers-1)
                               →  Linear → (LayerNorm)

    Notes:
        - No activation on the final layer (loss runs on z directly).
        - LayerNorm is used here for (B, D) tensors for consistency with your style.
          You may switch to BatchNorm1d if you prefer BN semantics.

    Args:
        input_dim (int):  Input feature dimension.
        hidden_dim (int): Hidden layer dimension (used when num_layers > 1).
        output_dim (int): Output projection dimension.
        num_layers (int): Number of linear layers (paper-default: 2).
        batch_norm (bool): Apply LayerNorm after each linear layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        batch_norm: bool = True,
        **kwargs,
    ):
        layers: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]] = []

        if num_layers == 1:
            # Linear → (LayerNorm)
            layers.append(
                (
                    input_dim,
                    output_dim,
                    nn.LayerNorm(output_dim) if batch_norm else None,
                    None,
                )
            )
        else:
            # (Linear → (LayerNorm) → ReLU) repeated (num_layers - 1) times
            layers.append(
                (
                    input_dim,
                    hidden_dim,
                    nn.LayerNorm(hidden_dim) if batch_norm else None,
                    nn.ReLU(inplace=True),
                )
            )
            for _ in range(2, num_layers):
                layers.append(
                    (
                        hidden_dim,
                        hidden_dim,
                        nn.LayerNorm(hidden_dim) if batch_norm else None,
                        nn.ReLU(inplace=True),
                    )
                )
            # Final Linear → (LayerNorm)
            layers.append(
                (
                    hidden_dim,
                    output_dim,
                    nn.LayerNorm(output_dim) if batch_norm else None,
                    None,
                )
            )

        super().__init__(layers)
