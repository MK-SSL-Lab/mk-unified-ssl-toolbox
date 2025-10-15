import copy
import torch
import torch.nn as nn
import torchvision.models as models

from MK_SSL.vision.models.modules.heads import DINOProjectionHead
from MK_SSL.vision.models.modules.losses import DINOLoss
from MK_SSL.vision.models.modules.transformations import SimCLRViewTransform
from MK_SSL.vision.models.utils import register_method


class DINO(nn.Module):
    """DINO: Self-Supervised Vision Representation Learning via Knowledge Distillation.

    Reference:
        - Paper: https://arxiv.org/abs/2104.14294
        - Code: https://github.com/facebookresearch/dino

    Args:
        backbone (nn.Module, optional): Encoder network. If None, defaults to ResNet-50.
        feature_size (int, optional): Feature size from the backbone. Defaults to 2048.
        projection_dim (int): Output dimension of the projection head. Defaults to 256.
        hidden_dim (int): Hidden layer dimension in the projection head. Defaults to 2048.
        bottleneck_dim (int): Dimension of the bottleneck layer. Defaults to 256.
        temp_student (float): Temperature for the student network. Defaults to 0.1.
        temp_teacher (float): Temperature for the teacher network. Defaults to 0.5.
        projection_num_layers (int): Number of layers in the projection head. Defaults to 3.
        norm_last_layer (bool): Whether to normalize the last projection layer. Defaults to True.
        momentum_teacher (float): EMA momentum for teacher update. Defaults to 0.996.
        num_crops (int): Number of augmented crops. Defaults to 6.
        use_bn_in_head (bool): Whether to use batch norm in projection head. Defaults to False.
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        backbone: nn.Module = None,
        feature_size: int = 2048,
        projection_dim: int = 256,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        temp_student: float = 0.1,
        temp_teacher: float = 0.5,
        projection_num_layers: int = 3,
        norm_last_layer: bool = True,
        momentum_teacher: float = 0.996,
        num_crops: int = 6,
        use_bn_in_head: bool = False,
        **kwargs,
    ):
        super().__init__()

        # Default to ResNet-50 backbone if none is provided
        if backbone is None:
            base_model = models.resnet50(weights=None)
            backbone = nn.Sequential(*list(base_model.children())[:-1])
            feature_size = 2048

        self.encoder_backbone = backbone
        self.feat_dim = feature_size
        self.proj_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.temp_student = temp_student
        self.temp_teacher = temp_teacher
        self.norm_last_layer = norm_last_layer
        self.use_bn = use_bn_in_head
        self.momentum_teacher = momentum_teacher
        self.num_crops = num_crops
        self.num_layers = projection_num_layers

        # -----------------------------
        # Student network components
        # -----------------------------
        self.student_head = DINOProjectionHead(
            input_dim=self.feat_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.proj_dim,
            bottleneck_dim=self.bottleneck_dim,
            use_bn=self.use_bn,
            norm_last_layer=self.norm_last_layer,
            num_layers=self.num_layers,
        )
        self.student = nn.Sequential(self.encoder_backbone, self.student_head)

        # -----------------------------
        # Teacher network components
        # -----------------------------
        self.teacher_head = DINOProjectionHead(
            input_dim=self.feat_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.proj_dim,
            bottleneck_dim=self.bottleneck_dim,
            use_bn=self.use_bn,
            norm_last_layer=self.norm_last_layer,
            num_layers=self.num_layers,
        )
        self.teacher = nn.Sequential(copy.deepcopy(self.encoder_backbone), self.teacher_head)

        self._initialize_teacher()

    @torch.no_grad()
    def _initialize_teacher(self):
        """Copies student parameters to initialize teacher (no gradients)."""
        for p_student, p_teacher in zip(self.student.parameters(), self.teacher.parameters()):
            p_teacher.data.copy_(p_student.data)
            p_teacher.requires_grad = False

    @torch.no_grad()
    def _momentum_update_teacher(self):
        """Applies momentum update to teacher parameters."""
        for p_student, p_teacher in zip(self.student.parameters(), self.teacher.parameters()):
            p_teacher.data = (
                self.momentum_teacher * p_teacher.data
                + (1.0 - self.momentum_teacher) * p_student.data
            )

    def forward(self, global_view_1: torch.Tensor, global_view_2: torch.Tensor, local_views: list):
        """Forward pass through student and teacher networks."""
        # Student forward (all crops)
        z1_student = self.student(global_view_1)
        z2_student = self.student(global_view_2)

        zc_student = [self.student(view) for view in local_views]

        # Teacher forward (only global views)
        with torch.no_grad():
            self._momentum_update_teacher()
            z1_teacher = self.teacher(global_view_1)
            z2_teacher = self.teacher(global_view_2)

        z_student_all = [z1_student, z2_student] + zc_student
        z_teacher_all = [z1_teacher, z2_teacher]

        return z_student_all, z_teacher_all


register_method(
    name="dino",
    model_cls=DINO,
    loss=DINOLoss,
    transformation=SimCLRViewTransform,
    logs=lambda model, loss_fn: (
        "\n"
        "---------------- DINO Configuration ----------------\n"
        f"Projection Dimension                          : {model.proj_dim}\n"
        f"Projection Hidden Dimension                   : {model.hidden_dim}\n"
        f"Bottleneck Dimension                          : {model.bottleneck_dim}\n"
        f"Student Temp                                  : {model.temp_student}\n"
        f"Teacher Temp                                  : {model.temp_teacher}\n"
        f"Last layer normalization                      : {model.norm_last_layer}\n"
        f"Center Momentum                               : {loss_fn.center_momentum}\n"
        f"Teacher Momentum                              : {model.momentum_teacher}\n"
        f"Number of crops                               : {model.num_crops}\n"
        f"Use batch norm in projection head             : {model.use_bn}\n"
        "Loss                                          : DINO Loss\n"
        "Transformation global_1                       : SimCLRViewTransform\n"
        "Transformation global_2                       : SimCLRViewTransform\n"
        "Transformation local                          : SimCLRViewTransform"
    ),
)
