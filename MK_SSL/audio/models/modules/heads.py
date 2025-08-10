import torch
import torch.nn as nn
from typing import List, Optional, Tuple


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



class COLAProjectionHead(ProjectionHead):
    """
    Projection head for COLA (Contrastive Learning of Audio).

    * If ``num_layers == 1``  ➜  Linear → LayerNorm → **tanh**
    * If ``num_layers  >  1`` ➜  (Linear → LayerNorm → ReLU)×(num_layers‑1)
                               →  Linear → LayerNorm → **tanh**

    Args:
        input_dim (int):  Input feature dimension (e.g. 1280).
        hidden_dim (int): Hidden layer dimension (only used when num_layers > 1).
        output_dim (int): Output projection dimension (e.g. 512).
        num_layers (int): Number of linear layers.
        batch_norm (bool): Use LayerNorm after each linear layer.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 1280,
        output_dim: int = 512,
        num_layers: int = 1,
        batch_norm: bool = True,
        **kwargs,
    ):
        layers: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]] = []

        if num_layers == 1:
            # Linear → (LayerNorm) → tanh
            layers.append(
                (
                    input_dim,
                    output_dim,
                    nn.LayerNorm(output_dim) if batch_norm else None,
                    nn.Tanh(),
                )
            )
        else:
            # (Linear → (LayerNorm) → ReLU) repeated
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
            # Final Linear → (LayerNorm) → tanh
            layers.append(
                (
                    hidden_dim,
                    output_dim,
                    nn.LayerNorm(output_dim) if batch_norm else None,
                    nn.Tanh(),
                )
            )

        super().__init__(layers)



class InputSpeechSimCLRProjectionHead(ProjectionHead):
    """
    Projects input features (e.g., FBANK with dim=80) to Transformer embed_dim.

    Args:
        input_dim (int): Input feature size (e.g., 80).
        output_dim (int): Embedding dimension (e.g., 256).
        use_layer_norm (bool): Whether to apply LayerNorm.
        dropout (float): Dropout rate.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        norm = nn.LayerNorm(output_dim) if use_layer_norm else None
        super().__init__([(input_dim, output_dim, norm, None)])
        self.post_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.post_dropout(super().forward(x))



class SpeechSimCLRProjectionHead(ProjectionHead):
    """
    Description:
        Initialize a new SpeechSimCLRProjectionHead instance.
        Uses LayerNorm instead of BatchNorm as recommended for speech models.

    Args:
        input_dim: Number of input dimensions.
        hidden_dim: Number of hidden dimensions.
        output_dim: Number of output dimensions.
        num_layers: Number of hidden layers (2 for v1, 3+ for v2).
        batch_norm: Whether or not to use LayerNorm (applied across features).
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 768,
        output_dim: int = 128,
        num_layers: int = 2,
        batch_norm: bool = True,
        **kwargs,
    ):
        layers: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]] = []

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

        layers.append(
            (
                hidden_dim,
                output_dim,
                nn.LayerNorm(output_dim) if batch_norm else None,
                None,
            )
        )

        super().__init__(layers)



class Wav2Vec2FeatureProjectionHead(ProjectionHead):
    """Maps ConvFeatureExtractor output to Transformer embed_dim for wav2vec 2.0."""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        norm = nn.LayerNorm(output_dim) if use_layer_norm else None
        super().__init__([(input_dim, output_dim, norm, None)])
        self.post_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # (B, T, C_in) → (B, T, C_out)
        x = super().forward(x)
        return self.post_dropout(x)


class HuBERTProjectionHead(ProjectionHead):
    """
    Projection head for HuBERT.
    Maps encoder output to a smaller dimension (e.g., 256) before prediction head.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        norm = nn.LayerNorm(output_dim) if use_layer_norm else None
        super().__init__([(input_dim, output_dim, norm, None)])
        self.post_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.post_dropout(super().forward(x))