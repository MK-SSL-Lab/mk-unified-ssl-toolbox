import torch
import torch.nn as nn
import torchvision.models as models
import copy

from MK_SSL.vision.models.modules.heads import BYOLPredictionHead, BYOLProjectionHead
from MK_SSL.vision.models.modules.losses import BYOLLoss
from MK_SSL.vision.models.modules.transformations import SimCLRViewTransform
from MK_SSL.vision.models.utils import register_method


class BYOL(nn.Module):
    """BYOL: Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning.

    Reference:
        - Paper: https://arxiv.org/abs/2006.07733
        - Code: https://github.com/deepmind/deepmind-research/tree/master/byol

    Args:
        backbone (nn.Module, optional): Encoder backbone. Defaults to ResNet-50.
        feature_size (int, optional): Feature dimension of backbone output. Defaults to 2048.
        projection_dim (int): Output dimension of the projection head. Defaults to 256.
        hidden_dim (int): Hidden layer size in projection/prediction heads. Defaults to 4096.
        moving_average_decay (float): EMA decay for the target encoder. Defaults to 0.99.
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        backbone: nn.Module = None,
        feature_size: int = 2048,
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        moving_average_decay: float = 0.99,
        **kwargs,
    ):
        super().__init__()

        if backbone is None:
            base_model = models.resnet50(weights=None)
            backbone = nn.Sequential(*list(base_model.children())[:-1])
            feature_size = 2048

        self.encoder_backbone = backbone
        self.feat_dim = feature_size
        self.proj_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.momentum = moving_average_decay

        # Online network
        self.projector = BYOLProjectionHead(self.feat_dim, self.hidden_dim, self.proj_dim)
        self.predictor = BYOLPredictionHead(self.proj_dim, self.hidden_dim, self.proj_dim)
        self.online_encoder = nn.Sequential(self.encoder_backbone, self.projector)

        # Target network
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self._initialize_target_encoder()

    @torch.no_grad()
    def _initialize_target_encoder(self):
        """Initializes target encoder weights from online encoder."""
        for param_online, param_target in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_target.data.copy_(param_online.data)
            param_target.requires_grad = False

    @torch.no_grad()
    def _update_target_encoder(self):
        """Momentum update for the target encoder."""
        for param_online, param_target in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_target.data = (
                self.momentum * param_target.data
                + (1.0 - self.momentum) * param_online.data
            )

    def forward(self, view_1: torch.Tensor, view_2: torch.Tensor):
        """Forward pass for both views through online and target encoders."""
        # Online pathway
        proj_1_online = self.online_encoder(view_1)
        proj_2_online = self.online_encoder(view_2)
        pred_1_online = self.predictor(proj_1_online)
        pred_2_online = self.predictor(proj_2_online)

        # Target pathway (no gradient updates)
        with torch.no_grad():
            self._update_target_encoder()
            proj_1_target = self.target_encoder(view_1)
            proj_2_target = self.target_encoder(view_2)

        return (pred_1_online, proj_1_target), (pred_2_online, proj_2_target)


register_method(
    name="byol",
    model_cls=BYOL,
    loss=BYOLLoss,
    transformation=SimCLRViewTransform,
    logs=lambda model, loss_fn: (
        "\n"
        "---------------- BYOL Configuration ----------------\n"
        f"Projection Dimension         : {model.proj_dim}\n"
        f"Projection Hidden Dimension  : {model.hidden_dim}\n"
        f"Moving average decay         : {model.momentum}\n"
        "Loss                         : BYOL Loss\n"
        "Transformation               : SimCLRViewTransform\n"
        "Transformation prime         : SimCLRViewTransform"
    )
)
