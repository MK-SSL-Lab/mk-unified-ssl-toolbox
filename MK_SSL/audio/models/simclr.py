import torch
import torch.nn as nn
 
from MK_SSL.audio.models.modules.heads import SpeechSimCLRProjectionHead



class SimCLRSpeech(nn.Module):
    """
    SimCLR for Speech: A framework for contrastive learning of speech representations.
    Based on: https://arxiv.org/abs/2010.13991 (Speech SimCLR)
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_size: int,
        projection_dim: int = 128,
        projection_num_layers: int = 2,
        projection_batch_norm: bool = True,
        **kwargs
    ):
        """
        Args:
            backbone (nn.Module): Backbone model (e.g., Transformer) to extract speech features.
            feature_size (int): Dimensionality of the backbone output features.
            projection_dim (int): Output dimension of the projection head.
            projection_num_layers (int): Number of layers in the projection head.
            projection_batch_norm (bool): Whether to use BatchNorm/LayerNorm in the projection head.
        """
        super().__init__()
        self.feature_size = feature_size
        self.projection_dim = projection_dim
        self.projection_num_layers = projection_num_layers
        self.projection_batch_norm = projection_batch_norm
        self.backbone = backbone
        self.projection_head = SpeechSimCLRProjectionHead(
            input_dim=self.feature_size,
            hidden_dim=self.feature_size,
            output_dim=self.projection_dim,
            num_layers=self.projection_num_layers,
            batch_norm=self.projection_batch_norm,
        )
        self.encoder = nn.Sequential(self.backbone, self.projection_head)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor = None):
        """
        Forward pass for SimCLR.
        
        Args:
            x0 (torch.Tensor): First augmented input tensor of shape (B, T, F).
            x1 (torch.Tensor, optional): Second augmented input tensor of shape (B, T, F). Defaults to None.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Projected representations.
        """
        f0 = self.backbone(x0)           # (B, T, D)
        f0_pooled = f0.mean(dim=1)       # (B, D)
        out0 = self.projection_head(f0_pooled)

        if x1 is None:
            return out0

        f1 = self.backbone(x1)
        f1_pooled = f1.mean(dim=1)
        out1 = self.projection_head(f1_pooled)

        return out0, out1

