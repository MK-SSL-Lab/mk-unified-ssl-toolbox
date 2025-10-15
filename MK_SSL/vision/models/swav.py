import torch
import torch.nn as nn


import torchvision.models as models

from MK_SSL.vision.models.modules.heads import SwAVProjectionHead
from MK_SSL.vision.models.modules.losses import SwAVLoss
from MK_SSL.vision.models.modules.transformations import SimCLRViewTransform
from MK_SSL.vision.models.utils import register_method

class SwAV(nn.Module):
    """SwAV: Self-supervised learning via clustering and contrastive learning.

    Reference:
        - Paper: https://arxiv.org/abs/2006.09882
        - Code: https://github.com/facebookresearch/swav

    Args:
        backbone (nn.Module): The encoder network.
        feature_size (int): Size of the features from the encoder.
        projection_dim (int): Output dimension of the projection head. Default is 128.
        hidden_dim (int): Hidden layer dimension in the projection head. Default is 2048.
        epsilon (float): Temperature parameter for Sinkhorn. Default is 0.05.
        sinkhorn_iterations (int): Number of iterations in Sinkhorn. Default is 3.
        num_prototypes (int): Number of prototype vectors. Default is 3000.
        queue_length (int): Length of the embedding queue. Default is 64.
        use_the_queue (bool): Whether to use the queue for SwAV training. Default is True.
        num_crops (int): Number of augmented views. Default is 6.
        **kwargs: Additional arguments.
    """
    def __init__(
        self,
        backbone: nn.Module = None,
        feature_size: int = 2048,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        epsilon: float = 0.05,
        sinkhorn_iterations: int = 3,
        num_prototypes: int = 3000,
        queue_length: int = 64,
        use_the_queue: bool = True,
        num_crops: int = 6,
        **kwargs,
    ):
        super().__init__()

        if backbone is None:
            # Load ResNet-50 backbone without final classification layer
            resnet = models.resnet50(weights=None)
            modules = list(resnet.children())[:-1]  # Remove final FC layer
            backbone = nn.Sequential(*modules)
            feature_size = 2048  # Output of ResNet-50 before FC layer

        self.encoder_backbone = backbone
        self.feat_dim = feature_size
        self.proj_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.sinkhorn_eps = epsilon
        self.sinkhorn_iters = sinkhorn_iterations
        self.num_proto = num_prototypes
        self.queue_size = queue_length
        self.enable_queue = use_the_queue
        self.total_crops = num_crops

        self.register_buffer("embedding_queue", torch.zeros(2, self.queue_size, self.proj_dim))

        self.head = SwAVProjectionHead(self.feat_dim, self.hidden_dim, self.proj_dim)
        self.network = nn.Sequential(self.encoder_backbone, self.head)

        self.prototypes = nn.Linear(self.proj_dim, self.num_proto, bias=False)
        self._initialize_weights()

    @torch.no_grad()
    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    @torch.no_grad()
    def sinkhorn(self, scores: torch.Tensor) -> torch.Tensor:
        """Perform Sinkhorn-Knopp algorithm on scores."""
        Q = torch.exp(scores / self.sinkhorn_eps).t()
        B = Q.shape[1]
        K = Q.shape[0]
        Q /= torch.sum(Q)

        for _ in range(self.sinkhorn_iters):
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B
        return Q.t()

    def forward(self, anchor1: torch.Tensor, anchor2: torch.Tensor, views: list):
        batch_size = anchor1.size(0)

        # Normalize prototype weights
        with torch.no_grad():
            proto_weights = nn.functional.normalize(self.prototypes.weight.data.clone(), dim=1, p=2)
            self.prototypes.weight.copy_(proto_weights)

        # Encode and normalize anchor inputs
        z1 = nn.functional.normalize(self.network(anchor1), dim=1, p=2)
        z2 = nn.functional.normalize(self.network(anchor2), dim=1, p=2)

        z1_detach, z2_detach = z1.detach(), z2.detach()
        logits1, logits2 = self.prototypes(z1_detach), self.prototypes(z2_detach)

        logits1_detach, logits2_detach = logits1.detach(), logits2.detach()

        # Use queue if enabled
        with torch.no_grad():
            if self.embedding_queue is not None and self.enable_queue:
                queue_logits1 = torch.mm(self.embedding_queue[0], self.prototypes.weight.t())
                queue_logits2 = torch.mm(self.embedding_queue[1], self.prototypes.weight.t())

                logits1_detach = torch.cat([queue_logits1, logits1_detach], dim=0)
                logits2_detach = torch.cat([queue_logits2, logits2_detach], dim=0)

                self.embedding_queue[0, batch_size:] = self.embedding_queue[0, :-batch_size].clone()
                self.embedding_queue[0, :batch_size] = z1_detach

                self.embedding_queue[1, batch_size:] = self.embedding_queue[1, :-batch_size].clone()
                self.embedding_queue[1, :batch_size] = z2_detach

            q1 = self.sinkhorn(logits1_detach)[:batch_size]
            q2 = self.sinkhorn(logits2_detach)[:batch_size]

        # Process multi-crop views
        crop_embeddings, crop_logits = [], []
        for view in views:
            proj = nn.functional.normalize(self.network(view), dim=1, p=2).detach()
            crop_embeddings.append(proj)
            crop_logits.append(self.prototypes(proj))

        return (logits1, logits2, crop_logits), (q1, q2)


register_method(
    name="swav",
    model_cls=SwAV,
    loss=SwAVLoss,
    transformation=SimCLRViewTransform,
    logs=lambda model, loss_fn: (
        "\n"
        "---------------- SwAV Configuration ----------------\n"
        f"Projection Dimension         : {model.proj_dim}\n"
        f"Projection Hidden Dimension  : {model.hidden_dim}\n"
        f"Number of crops              : {model.total_crops}\n"
        "Loss                         : SwAV Loss\n"
        "Transformation global        : SimCLRViewTransform\n"
        "Transformation local         : SimCLRViewTransform"
    )
)
