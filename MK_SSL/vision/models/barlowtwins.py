import torch
import torch.nn as nn
import torchvision.models as models

from MK_SSL.vision.models.modules.heads import BarlowTwinsProjectionHead
from MK_SSL.vision.models.modules.losses import BarlowTwinsLoss
from MK_SSL.vision.models.modules.transformations import SimCLRViewTransform
from MK_SSL.vision.models.utils import register_method


class BarlowTwins(nn.Module):
    """Barlow Twins: Redundancy Reduction for Self-Supervised Learning.

    Reference:
        - Paper: https://arxiv.org/abs/2103.03230
        - Code: https://github.com/facebookresearch/barlowtwins

    Args:
        backbone (nn.Module, optional): Encoder network. Defaults to ResNet-50.
        feature_size (int, optional): Dimensionality of backbone output. Defaults to 2048.
        projection_dim (int): Output size of the projection head. Defaults to 8192.
        hidden_dim (int): Hidden size inside the projection head. Defaults to 8192.
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        backbone: nn.Module = None,
        feature_size: int = 2048,
        projection_dim: int = 8192,
        hidden_dim: int = 8192,
        **kwargs,
    ):
        super().__init__()

        # Default backbone = ResNet-50
        if backbone is None:
            base_model = models.resnet50(weights=None)
            backbone = nn.Sequential(*list(base_model.children())[:-1])
            feature_size = 2048

        self.encoder_backbone = backbone
        self.feat_dim = feature_size
        self.proj_dim = projection_dim
        self.hidden_dim = hidden_dim

        self.projector = BarlowTwinsProjectionHead(
            input_dim=self.feat_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.proj_dim,
        )

        self.encoder = nn.Sequential(self.encoder_backbone, self.projector)

    def forward(self, view_1: torch.Tensor, view_2: torch.Tensor = None):
        """Forward pass of Barlow Twins for one or two views."""
        features_1 = self.encoder_backbone(view_1).flatten(start_dim=1)
        output_1 = self.projector(features_1)

        if view_2 is None:
            return output_1

        features_2 = self.encoder_backbone(view_2).flatten(start_dim=1)
        output_2 = self.projector(features_2)

        return output_1, output_2


register_method(
    name="barlowtwins",
    model_cls=BarlowTwins,
    loss=BarlowTwinsLoss,
    transformation=SimCLRViewTransform,
    logs=lambda model, loss_fn: (
        "\n"
        "---------------- BarlowTwins Configuration ----------------\n"
        f"Projection Dimension         : {model.proj_dim}\n"
        f"Projection Hidden Dimension  : {model.hidden_dim}\n"
        "Loss                         : BarlowTwins Loss\n"
        "Transformation               : SimCLRViewTransform\n"
        "Transformation prime         : SimCLRViewTransform"
    )
)
