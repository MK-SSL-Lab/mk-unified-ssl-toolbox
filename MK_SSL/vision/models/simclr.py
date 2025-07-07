import torch
import torch.nn as nn

from MK_SSL.vision.models.modules.heads import SimCLRProjectionHead
from MK_SSL.vision.models.modules.losses import NT_Xent
from MK_SSL.vision.models.modules.transformations import SimCLRViewTransform
from MK_SSL.vision.models.utils import register_method  

class SimCLR(nn.Module):
    """
    SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
    Link: https://arxiv.org/abs/2002.05709
    Implementation: https://github.com/google-research/simclr
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
            backbone (nn.Module): Backbone to extract features.
            feature_size (int): Feature size.
            projection_dim (int): Projection head output dimension.
            projection_num_layers (int): Number of layers in the projection head.
            projection_batch_norm (bool): Whether to use batch norm in the projection head.
        """
        super().__init__()
        self.feature_size = feature_size
        self.projection_dim = projection_dim
        self.projection_num_layers = projection_num_layers
        self.projection_batch_norm = projection_batch_norm
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(
            input_dim=self.feature_size,
            hidden_dim=self.feature_size,
            output_dim=self.projection_dim,
            num_layers=self.projection_num_layers,
            batch_norm=self.projection_batch_norm,
        )
        self.encoder = nn.Sequential(self.backbone, self.projection_head)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor = None):
        f0 = self.backbone(x0).flatten(start_dim=1)
        out0 = self.projection_head(f0)

        if x1 is None:
            return out0

        f1 = self.backbone(x1).flatten(start_dim=1)
        out1 = self.projection_head(f1)

        return out0, out1


register_method(
    name= "simclr",
    model_cls= SimCLR,
    loss= NT_Xent,
    transformation= SimCLRViewTransform,
    logs=lambda model, loss: (
        "\n"
        "---------------- SimCLR Configuration ----------------\n"
        f"Projection Dimension                  : {model.projection_dim}\n"
        f"Projection number of layers           : {model.projection_num_layers}\n"
        f"Projection batch normalization        : {model.projection_batch_norm}\n"
        "Loss                                  : NT_Xent Loss\n"
        "Transformation                        : SimCLRViewTransform"
    )
)