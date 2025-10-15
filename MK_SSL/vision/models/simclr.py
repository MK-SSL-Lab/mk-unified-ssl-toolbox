import torch
import torch.nn as nn
import torchvision.models as models

from MK_SSL.vision.models.modules.heads import SimCLRProjectionHead
from MK_SSL.vision.models.modules.losses import NT_Xent
from MK_SSL.vision.models.modules.transformations import SimCLRViewTransform
from MK_SSL.vision.models.utils import register_method


class SimCLR(nn.Module):
    """SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.

    Reference:
        - Paper: https://arxiv.org/abs/2002.05709
        - Code: https://github.com/google-research/simclr

    Args:
        backbone (nn.Module, optional): Backbone CNN encoder. Defaults to ResNet-50.
        feature_size (int, optional): Output feature size of the backbone. Defaults to 2048.
        projection_dim (int): Output dimension of the projection head. Defaults to 128.
        projection_num_layers (int): Number of MLP layers in the projection head. Defaults to 2.
        projection_batch_norm (bool): Whether to use batch normalization. Defaults to True.
        **kwargs: Additional optional arguments.
    """

    def __init__(
        self,
        backbone: nn.Module = None,
        feature_size: int = 2048,
        projection_dim: int = 128,
        projection_num_layers: int = 2,
        projection_batch_norm: bool = True,
        **kwargs,
    ):
        super().__init__()

        if backbone is None:
            # Default to ResNet-50 without classification head
            base_model = models.resnet50(weights=None)
            backbone = nn.Sequential(*list(base_model.children())[:-1])
            feature_size = 2048

        self.encoder_cnn = backbone
        self.feature_dim = feature_size
        self.proj_dim = projection_dim
        self.proj_layers = projection_num_layers
        self.use_bn = projection_batch_norm

        self.projection_mlp = SimCLRProjectionHead(
            input_dim=self.feature_dim,
            hidden_dim=self.feature_dim,
            output_dim=self.proj_dim,
            num_layers=self.proj_layers,
            batch_norm=self.use_bn,
        )

        self.encoder = nn.Sequential(self.encoder_cnn, self.projection_mlp)

    def forward(self, view1: torch.Tensor, view2: torch.Tensor = None):
        """Encodes two views through encoder and projection head."""
        emb1 = self.encoder_cnn(view1).flatten(start_dim=1)
        proj1 = self.projection_mlp(emb1)

        if view2 is None:
            return proj1

        emb2 = self.encoder_cnn(view2).flatten(start_dim=1)
        proj2 = self.projection_mlp(emb2)

        return proj1, proj2


register_method(
    name="simclr",
    model_cls=SimCLR,
    loss=NT_Xent,
    transformation=SimCLRViewTransform,
    logs=lambda model, loss_fn: (
        "\n"
        "---------------- SimCLR Configuration ----------------\n"
        f"Projection Dimension                  : {model.proj_dim}\n"
        f"Projection number of layers           : {model.proj_layers}\n"
        f"Projection batch normalization        : {model.use_bn}\n"
        "Loss                                  : NT_Xent Loss\n"
        "Transformation                        : SimCLRViewTransform"
    )
)
