import torch
import torch.nn as nn

from MK_SSL.vision.models.modules.heads import BarlowTwinsProjectionHead
from MK_SSL.vision.models.utils import register_method
from MK_SSL.vision.models.modules.losses import BarlowTwinsLoss
from MK_SSL.vision.models.modules.transformations import SimCLRViewTransform
class BarlowTwins(nn.Module):
    """
    Barlow Twins
    Link: https://arxiv.org/abs/2104.02057
    Implementation: https://arxiv.org/abs/2103.03230
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_size: int,
        projection_dim: int = 8192,
        hidden_dim: int = 8192,
        **kwargs
    ):
        """
        Args:
            backbone: Backbone network.
            feature_size: Number of features.
            projection_dim: Dimension of projection head output.
            hidden_dim: Dimension of hidden layer in projection head.
        """
        super().__init__()
        self.backbone = backbone
        self.feature_size = feature_size
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.projection_head = BarlowTwinsProjectionHead(
            input_dim=self.feature_size,
            hidden_dim=self.hidden_dim,
            output_dim=self.projection_dim,
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
    name= "barlowtwins",
    model_cls= BarlowTwins,
    loss= BarlowTwinsLoss,
    transformation= SimCLRViewTransform,
    logs=lambda model, loss: (
        "\n"
        "---------------- BarlowTwins Configuration ----------------\n"
        f"Projection Dimension         : {model.projection_dim}\n"
        f"Projection Hidden Dimension  : {model.hidden_dim}\n"
        "Loss                         : BarlowTwins Loss\n"
        "Transformation               : SimCLRViewTransform\n"
        "Transformation prime         : SimCLRViewTransform"
    )
)