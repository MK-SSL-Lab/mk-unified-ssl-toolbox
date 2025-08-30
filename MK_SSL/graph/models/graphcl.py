import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Any

from MK_SSL.graph.models.modules.heads import GraphCLProjectionHead
from MK_SSL.graph.models.modules.backbones import GNNGraphEncoder
from MK_SSL.graph.models.modules.losses.graphcl_loss import NTXentGraphLoss
from MK_SSL.graph.models.modules.transformations.graphcl_transform import GraphCLGraphTransform
from MK_SSL.graph.models.utils.registry import register_method


class GraphCL(nn.Module):
    """Graph Contrastive Learning (GraphCL).

    Based on:
        You et al., "Graph Contrastive Learning with Augmentations", NeurIPS 2020.

    Two independently augmented views per graph → shared encoder → 2-layer MLP head →
    NT-Xent on projected embeddings.

    Args:
        feature_size: Output dimension of the backbone encoder features.
        backbone: Optional external encoder. If None, uses GNNGraphEncoder.
        projection_dim: Output projection dimension.
        projection_num_layers: Number of layers in the projection head (paper: 2).
        projection_batch_norm: If True, apply norm layers inside the projection head.
        **kwargs: Forwarded to GNNGraphEncoder when backbone is None (must include in_dim).
    """

    def __init__(
        self,
        feature_size: int = 512,
        backbone: Optional[nn.Module] = None,
        in_dim: int = None,
        projection_dim: int = 128,
        projection_num_layers: int = 2,
        projection_batch_norm: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.projection_dim = projection_dim
        self.projection_num_layers = projection_num_layers
        self.projection_batch_norm = projection_batch_norm

        # Shared GNN encoder (architecture-agnostic per paper).
        if backbone is None:
            if not in_dim:
                raise ValueError(
                    "GNNGraphEncoder requires `in_dim`. Pass it via GraphCL(..., in_dim=..., ...)"
                )
            self.backbone = GNNGraphEncoder(in_dim=in_dim, out_dim=self.feature_size, **kwargs)
        else:
            self.backbone = backbone

        # 2-layer MLP projection head by default (paper-aligned).
        self.projection_head = GraphCLProjectionHead(
            input_dim=self.feature_size,
            hidden_dim=self.feature_size,
            output_dim=self.projection_dim,
            num_layers=self.projection_num_layers,
            batch_norm=self.projection_batch_norm,
        )

    def _encode_view(self, view: Any) -> torch.Tensor:
        """Encodes a single augmented view and applies the projection head.

        Accepts:
            - PyG Batch/Data with attributes (x, edge_index, batch)
            - Tuple (x, edge_index, batch)
            - Already pooled 2D features (B, D==feature_size)

        Returns:
            z: Projected embeddings with shape (B, projection_dim).
        """
        # PyG Batch/Data path
        if hasattr(view, "x") and hasattr(view, "edge_index") and hasattr(view, "batch"):
            h = self.backbone(view.x, view.edge_index, view.batch)
        # Explicit triplet (x, edge_index, batch)
        elif isinstance(view, (tuple, list)) and len(view) == 3:
            x, edge_index, batch = view
            h = self.backbone(x, edge_index, batch)
        # Already graph-level features (B, D)
        elif isinstance(view, torch.Tensor) and view.dim() == 2 and view.size(1) == self.feature_size:
            h = view
        else:
            # Last-resort: attempt calling backbone(view) for custom encoders
            h = self.backbone(view)

        if h.dim() > 2:
            h = h.view(h.size(0), -1)
        z = self.projection_head(h)
        return z

    def forward(
        self,
        x0: Union[torch.Tensor, Any],
        x1: Optional[Union[torch.Tensor, Any]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            x0: First augmented graph batch/view.
            x1: Second augmented graph batch/view. If None, returns a single embedding.

        Returns:
            z0 or (z0, z1): Projected embedding(s), shape (B, projection_dim).
        """
        if x0 is None:
            raise ValueError("x0 must be provided for GraphCL.forward")
        if (x1 is not None) and (type(x1) is not type(x0)) and not (
            hasattr(x0, "x") and hasattr(x1, "x")
        ):
            raise ValueError(f"x1 must have the same type as x0 (got {type(x0)} vs {type(x1)})")

        z0 = self._encode_view(x0)
        if x1 is None:
            return z0
        z1 = self._encode_view(x1)
        return z0, z1


register_method(
    name="graphcl",
    model_cls=GraphCL,
    loss=NTXentGraphLoss,               # NT-Xent on projected z
    transformation=GraphCLGraphTransform,  # NodeDrop / EdgePerturb / AttrMask / Subgraph
    default_params={
        "augmentation_defaults": {
            "node_drop_ratio": 0.2,
            "edge_perturb_ratio": 0.2,
            "attr_mask_ratio": 0.2,
            "subgraph_ratio": 0.2,
        },
    },
    logs=lambda model, loss: (
        "\n"
        "---------------- GraphCL Configuration ----------------\n"
        f"Input Type                       : Two augmented graph views per sample\n"
        f"Backbone Architecture            : {model.backbone.__class__.__name__}\n"
        f"Backbone Output Dimension (D)    : {model.feature_size}\n"
        f"Projection Head Dimension        : {model.projection_dim}\n"
        f"Projection Head Layers           : {model.projection_num_layers}\n"
        "Loss                             : InfoNCE (NT-Xent, in-batch negatives)\n"
        "Augmentations                    : NodeDrop | EdgePerturb | AttrMask | Subgraph\n"
        "Augmentation Default Ratios      : drop=0.2 | edge=0.2 | mask=0.2 | subgraph=0.2\n"
    ),
)
