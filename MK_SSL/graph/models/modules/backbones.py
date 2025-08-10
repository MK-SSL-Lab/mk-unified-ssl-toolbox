import torch
import torch.nn as nn
from typing import Optional, List, Literal, Union, Tuple

# Optional PyG imports (required for actual execution)
try:
    from torch_geometric.nn import GINConv, GCNConv, global_mean_pool, global_add_pool, global_max_pool
    PYG_AVAILABLE = True
except Exception:  # pragma: no cover
    GINConv = GCNConv = global_mean_pool = global_add_pool = global_max_pool = None
    PYG_AVAILABLE = False


class MLP(nn.Module):
    """
    Simple MLP used inside GIN convolutions and as pre/post-encoders.

    Args:
        dims (List[int]): Layer dimensions including input and output (e.g., [in, h, out]).
        use_norm (bool): If True, applies LayerNorm after each Linear.
        activation (nn.Module): Non-linearity to apply between Linear and Norm.
        dropout (float): Dropout probability applied after activation.
        last_activation (bool): If True, also apply activation (and norm/dropout) after the last layer.
    """
    def __init__(
        self,
        dims: List[int],
        use_norm: bool = True,
        activation: nn.Module = nn.ReLU(inplace=True),
        dropout: float = 0.0,
        last_activation: bool = False,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        L = len(dims) - 1
        for i in range(L):
            in_dim, out_dim = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim, bias=not use_norm))
            if use_norm:
                layers.append(nn.LayerNorm(out_dim))
            # Apply activation/dropout on all hidden layers; optionally also on last
            if i < L - 1 or last_activation:
                if activation is not None:
                    layers.append(activation.__class__(**getattr(activation, "__dict__", {})))
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GNNGraphEncoder(nn.Module):
    """
    Default GraphCL backbone (paper-aligned): a shared GNN encoder producing graph-level features.

    Paper notes:
        - The framework does **not** constrain the GNN architecture.
        - They use standard backbones: GCN for semi-supervised experiments, GIN for
          unsupervised/transfer settings. This encoder lets you pick either, with GIN as default.

    Workflow:
        x, edge_index, batch → [GNN stack] → node embeddings → global READOUT (pool) → graph feature h.

    Args:
        in_dim (int):      Node feature dimension.
        hidden_dim (int):  Hidden size for GNN layers.
        out_dim (int):     Output feature dimension (graph-level embedding).
        num_layers (int):  Number of GNN layers (≥ 2 recommended).
        conv_type (Literal["gin","gcn"]): Convolution type. Defaults to "gin".
        readout (Literal["mean","sum","max"]): Graph READOUT function. Defaults to "mean".
        dropout (float):   Dropout after each layer.
        use_norm (bool):   Apply LayerNorm after each layer output.
        pre_mlp (Optional[List[int]]): Optional pre-MLP dims (e.g., [in_dim, hidden_dim]).
        post_mlp (Optional[List[int]]): Optional post-MLP dims (e.g., [hidden_dim, out_dim]).
        activation (nn.Module): Non-linearity (default: ReLU).

    Inputs:
        x (Tensor):          Node features of shape (num_nodes, in_dim).
        edge_index (Tensor): COO graph edges of shape (2, num_edges) [PyG].
        batch (Tensor):      Graph-id for each node, shape (num_nodes,). Required for READOUT.

    Returns:
        Tensor: Graph-level embedding h of shape (num_graphs, out_dim).
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 512,
        num_layers: int = 3,
        conv_type: Literal["gin", "gcn"] = "gin",
        readout: Literal["mean", "sum", "max"] = "mean",
        dropout: float = 0.0,
        use_norm: bool = True,
        pre_mlp: Optional[List[int]] = None,
        post_mlp: Optional[List[int]] = None,
        activation: nn.Module = nn.ReLU(inplace=True),
    ):
        super().__init__()
        if not PYG_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "torch_geometric is required for GNNGraphEncoder. "
                "Please install torch-geometric and torch-scatter/torch-sparse."
            )
        if num_layers < 1:
            raise ValueError("num_layers must be ≥ 1")

        self.readout = readout
        self.dropout = dropout
        self.use_norm = use_norm
        self.activation = activation

        # Optional input adaptor
        if pre_mlp:
            if pre_mlp[0] != in_dim:
                raise ValueError(f"pre_mlp first dim must equal in_dim ({in_dim}), got {pre_mlp[0]}")
            self.pre = MLP(pre_mlp, use_norm=use_norm, activation=activation, dropout=dropout)
            conv_in = pre_mlp[-1]
        else:
            self.pre = nn.Identity()
            conv_in = in_dim

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Build GNN layers
        last_dim = conv_in
        for layer_idx in range(num_layers):
            if conv_type == "gin":
                # GINConv expects an MLP as "nn"
                mlp = MLP([last_dim, hidden_dim, hidden_dim], use_norm=use_norm, activation=activation, dropout=dropout)
                conv = GINConv(nn=mlp, train_eps=True)
                next_dim = hidden_dim
            elif conv_type == "gcn":
                conv = GCNConv(last_dim, hidden_dim, cached=False, normalize=True)
                next_dim = hidden_dim
            else:
                raise ValueError(f"Unsupported conv_type: {conv_type}")

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(next_dim) if use_norm else nn.Identity())
            last_dim = next_dim

        self.proj = (
            MLP([last_dim, out_dim], use_norm=use_norm, activation=activation, dropout=dropout, last_activation=False)
            if post_mlp is None
            else MLP(post_mlp, use_norm=use_norm, activation=activation, dropout=dropout, last_activation=False)
        )

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.readout == "mean":
            return global_mean_pool(x, batch)
        if self.readout == "sum":
            return global_add_pool(x, batch)
        if self.readout == "max":
            return global_max_pool(x, batch)
        raise ValueError(f"Unknown readout: {self.readout}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features, shape (N, in_dim).
            edge_index: Edge indices (2, E).
            batch: Node-to-graph indices (N,).

        Returns:
            Graph-level embedding h, shape (B, out_dim).
        """
        x = self.pre(x)  # optional input adaptation

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = self.activation(x)
            if self.dropout > 0.0:
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        h = self._pool(x, batch)  # READOUT → graph feature
        h = self.proj(h)          # final projection to out_dim
        return h

    def embed_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Node-level embeddings after the last GNN layer (before READOUT).

        Args:
            x: Node features (N, in_dim).
            edge_index: Edge indices (2, E).

        Returns:
            Tensor: Node embeddings (N, hidden_dim).
        """
        x = self.pre(x)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = self.activation(x)
            if self.dropout > 0.0:
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x
