class COLAProjectionHead(ProjectionHead):
    """
    Description:
        Initialize a new COLAProjectionHead instance.
        Implements the projection head used in the COLA paper, consisting of
        one or more linear layers with optional LayerNorm and non-linearity.
        
        LayerNorm is preferred over BatchNorm for stability in audio.

    Args:
        input_dim (int): Number of input dimensions (typically 1280 from EfficientNet-B0).
        hidden_dim (int): Number of hidden dimensions (typically same as input_dim).
        output_dim (int): Output dimension of the projection space (e.g., 512).
        num_layers (int): Number of linear layers (1 for minimal COLA-style head).
        batch_norm (bool): Whether to use LayerNorm after each linear layer.
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
            # Simple linear projection with optional norm
            layers.append(
                (
                    input_dim,
                    output_dim,
                    nn.LayerNorm(output_dim) if batch_norm else None,
                    None,
                )
            )
        else:
            # First hidden layer
            layers.append(
                (
                    input_dim,
                    hidden_dim,
                    nn.LayerNorm(hidden_dim) if batch_norm else None,
                    nn.ReLU(inplace=True),
                )
            )
            # Intermediate hidden layers
            for _ in range(2, num_layers):
                layers.append(
                    (
                        hidden_dim,
                        hidden_dim,
                        nn.LayerNorm(hidden_dim) if batch_norm else None,
                        nn.ReLU(inplace=True),
                    )
                )
            # Final projection layer
            layers.append(
                (
                    hidden_dim,
                    output_dim,
                    nn.LayerNorm(output_dim) if batch_norm else None,
                    None,
                )
            )

        super().__init__(layers)
