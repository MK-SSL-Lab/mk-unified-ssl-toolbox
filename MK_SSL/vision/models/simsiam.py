import torch
import torch.nn as nn
import torchvision.models as models

from MK_SSL.vision.models.modules.heads import (
    SimSiamPredictionHead,
    SimSiamProjectionHead,
)
from MK_SSL.vision.models.modules.losses import NegativeCosineSimilarity
from MK_SSL.vision.models.modules.transformations import SimCLRViewTransform
from MK_SSL.vision.models.utils import register_method


class SimSiam(nn.Module):
    """SimSiam: A Simple Siamese Approach for Unsupervised Representation Learning.

    Reference:
        - Paper: https://arxiv.org/abs/2011.10566
        - Code: https://github.com/facebookresearch/simsiam

    Args:
        backbone (nn.Module, optional): Backbone encoder. If None, defaults to ResNet-50.
        feature_size (int, optional): Output feature size from the backbone. Defaults to 2048.
        projection_dim (int): Output dimension of the projection head. Defaults to 2048.
        projection_hidden_dim (int): Hidden layer size in the projection head. Defaults to 2048.
        prediction_hidden_dim (int): Hidden layer size in the prediction head. Defaults to 512.
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        backbone: nn.Module = None,
        feature_size: int = 2048,
        projection_dim: int = 2048,
        projection_hidden_dim: int = 2048,
        prediction_hidden_dim: int = 512,
        **kwargs,
    ):
        super().__init__()

        if backbone is None:
            # Default backbone is ResNet-50 with classification head removed
            resnet = models.resnet50(weights=None)
            backbone = nn.Sequential(*list(resnet.children())[:-1])
            feature_size = 2048

        self.encoder = backbone
        self.feat_dim = feature_size
        self.proj_dim = projection_dim
        self.proj_hidden_dim = projection_hidden_dim
        self.pred_hidden_dim = prediction_hidden_dim

        self.projector = SimSiamProjectionHead(
            input_dim=self.feat_dim,
            hidden_dim=self.proj_hidden_dim,
            output_dim=self.proj_dim,
        )

        self.predictor = SimSiamPredictionHead(
            input_dim=self.proj_dim,
            hidden_dim=self.pred_hidden_dim,
            output_dim=self.proj_dim,
        )

    def forward(self, input_a: torch.Tensor, input_b: torch.Tensor = None):
        """Forward pass through SimSiam encoder, projector, and predictor."""
        feat_a = self.encoder(input_a).flatten(start_dim=1)
        z_a = self.projector(feat_a)
        p_a = self.predictor(z_a)

        if input_b is None:
            return (z_a, p_a)

        feat_b = self.encoder(input_b).flatten(start_dim=1)
        z_b = self.projector(feat_b)
        p_b = self.predictor(z_b)

        return (z_a, p_a), (z_b, p_b)


register_method(
    name="simsiam",
    model_cls=SimSiam,
    loss=NegativeCosineSimilarity,
    transformation=SimCLRViewTransform,
    logs=lambda model, loss_fn: (
        "\n"
        "---------------- SimSiam Configuration ----------------\n"
        f"Projection Dimension           : {model.proj_dim}\n"
        f"Projection Hidden Dimension    : {model.proj_hidden_dim}\n"
        f"Prediction Hidden Dimension    : {model.pred_hidden_dim}\n"
        "Loss                           : Negative Cosine Simililarity\n"
        "Transformation                 : SimCLRViewTransform"
    )
)
