import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class ProjectionHead(nn.Module):
    """
    Base class for projection/prediction heads.

    Args:
        blocks: List of (in_features, out_features, batch_norm_layer, non_linearity_layer).
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


class GraphCLProjectionHead(ProjectionHead):
    """
    GraphCL projection head.

    Paper spec: small MLP (typically two layers). Loss runs on z (no activation on final layer).

    This version aligns closer to the paper/official repo:
      • Hidden layers: (Linear → (LayerNorm) → ReLU) × (num_layers - 1)
      • Final layer:   Linear (NO normalization, NO activation)

    If num_layers == 1:
      • Linear (NO normalization)
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
        def _ln(d: int) -> Optional[nn.Module]:
            return nn.LayerNorm(d) if batch_norm else None

        def _clone_activation_like(src: nn.Module) -> nn.Module:
            # Minimal, robust clone for common activations (e.g., ReLU(inplace))
            cls = src.__class__
            kwargs_ = {}
            if hasattr(src, "inplace"):
                kwargs_["inplace"] = bool(getattr(src, "inplace"))
            return cls(**kwargs_)

        layers: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]] = []

        if num_layers <= 1:
            # Final layer only — no norm per GraphCL convention
            layers.append((input_dim, output_dim, None, None))
        else:
            # Hidden blocks (with optional LayerNorm + ReLU)
            layers.append((input_dim, hidden_dim, _ln(hidden_dim), nn.ReLU(inplace=True)))
            for _ in range(2, num_layers):
                layers.append((hidden_dim, hidden_dim, _ln(hidden_dim), nn.ReLU(inplace=True)))
            # Final projection — NO normalization or activation
            layers.append((hidden_dim, output_dim, None, None))

        super().__init__(layers)
