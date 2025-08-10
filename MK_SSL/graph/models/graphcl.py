import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Any

from MK_SSL.graph.models.modules.heads import GraphProjectionHead
from MK_SSL.graph.models.modules.backbones import GNNGraphEncoder
from MK_SSL.graph.models.modules.losses import NTXent_loss
from MK_SSL.graph.models.modules.transformations import GraphCLGraphTransform
from MK_SSL.graph.models.utils.registry import register_method


class GraphCL(nn.Module):
    """Graph Contrastive Learning (GraphCL).

    Based on:
        You, Y., Chen, T., Sui, Y., Chen, T., Wang, Z., & Shen, Y. (2020).
        Graph Contrastive Learning with Augmentations. NeurIPS.

    This model consumes **two independently augmented views** of each input graph
    and learns graph-level embeddings by maximizing agreement between paired views
    via an NT-Xent (InfoNCE) objective computed on a projection space.

    Workflow:
        1) For each graph, sample two augmentations (node dropping, edge
           perturbation, attribute masking, subgraph) to obtain two correlated views.
        2) Encode each view with a **shared** GNN backbone to produce graph features.
        3) Project features with a **two-layer MLP** head to the contrastive space.
        4) Apply InfoNCE (NT-Xent) with in-batch negatives on the projected embeddings.

    Attributes:
        feature_size: Output dimension of the backbone encoder features.
        projection_dim: Output dimension of the projection head.
        projection_num_layers: Number of layers in the projection head
            (defaults to 2 to match the paper).
        projection_batch_norm: Whether to apply normalization inside the projection head.
        backbone: Shared GNN encoder that outputs graph-level features.
        projection_head: MLP mapping features to the contrastive space.

    Args:
        feature_size: Dimension of the backbone output features. Defaults to 512.
        backbone: Graph feature extractor. If ``None``, ``GNNGraphEncoder`` is used.
        projection_dim: Projection head output dimension. Defaults to 128.
        projection_num_layers: Number of layers in the projection head. Defaults to 2.
        projection_batch_norm: If ``True``, applies normalization inside the projection head.
            Defaults to True.
        **kwargs: Extra keyword arguments forwarded to ``GNNGraphEncoder`` when
            ``backbone`` is ``None`` (e.g., number of layers, hidden size, pooling).

    Inputs:
        x0: First augmented graph batch (e.g., PyG ``Batch`` or DGLGraph``).
        x1: Second augmented graph batch of the same type as ``x0`` (optional).

    Returns:
        torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
            If ``x1`` is provided, returns ``(z0, z1)`` where each is shape ``(B, projection_dim)``.
            Otherwise returns ``z0`` only.

    Raises:
        ValueError: If ``x0`` is missing or if ``x1`` is provided with an incompatible type.
    """

    def __init__(
        self,
        feature_size: int = 512,
        backbone: Optional[nn.Module] = None,
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

        # Shared GNN encoder (architecture-agnostic as per paper).
        self.backbone = backbone if backbone is not None else GNNGraphEncoder(
            out_dim=self.feature_size, **kwargs
        )

        # 2-layer MLP projection head by default (paper-aligned).
        self.projection_head = GraphProjectionHead(
            input_dim=self.feature_size,
            hidden_dim=self.feature_size,
            output_dim=self.projection_dim,
            num_layers=self.projection_num_layers,
            batch_norm=self.projection_batch_norm,
        )

    def forward(
        self,
        x0: Union[torch.Tensor, Any],
        x1: Optional[Union[torch.Tensor, Any]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for GraphCL.

        Args:
            x0: First augmented graph batch.
            x1: Second augmented graph batch. If ``None``, returns a single embedding.

        Returns:
            Projected embedding(s) suitable for InfoNCE.

        Raises:
            ValueError: If ``x0`` is None or ``x1`` type mismatches ``x0``.
        """
        if x0 is None:
            raise ValueError("x0 must be provided for GraphCL.forward")
        if (x1 is not None) and (type(x1) is not type(x0)):
            raise ValueError(f"x1 must have the same type as x0 (got {type(x0)} vs {type(x1)})")

        # Encode first view (graph-level representation).
        h0 = self.backbone(x0)  # expected (B, D)
        if h0.dim() > 2:
            h0 = h0.view(h0.size(0), -1)
        z0 = self.projection_head(h0)

        if x1 is None:
            return z0

        # Encode second view with the *shared* encoder.
        h1 = self.backbone(x1)  # expected (B, D)
        if h1.dim() > 2:
            h1 = h1.view(h1.size(0), -1)
        z1 = self.projection_head(h1)

        return z0, z1


register_method(
    name="graphcl",
    model_cls=GraphCL,
    loss=NTXent_loss,  # should implement NT-Xent (cosine sim + temperature) on projected z
    transformation=GraphCLGraphTransform,  # implement NodeDrop / EdgePerturb / AttrMask / Subgraph
    default_params={
        # Paper-aligned augmentation defaults (set inside the transform implementation).
        # Listed here for clarity/documentation:
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
