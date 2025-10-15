import copy
import torch
import torch.nn as nn
import torchvision.models as models

from MK_SSL.vision.models.modules.heads import SimCLRProjectionHead, BYOLPredictionHead
from MK_SSL.vision.models.modules.losses import InfoNCE_MoCoV3
from MK_SSL.vision.models.modules.transformations import SimCLRViewTransform
from MK_SSL.vision.models.utils import register_method


class MoCoV3(nn.Module):
    """MoCo v3: Momentum Contrastive Learning with Vision Transformers and ResNets.

    Reference:
        - Paper: https://arxiv.org/abs/2104.02057
        - Code: https://github.com/facebookresearch/moco-v3

    Args:
        backbone (nn.Module, optional): Encoder backbone. Defaults to ResNet-50.
        feature_size (int, optional): Feature dimension from the backbone. Defaults to 2048.
        projection_dim (int): Output dimension of the projection head. Defaults to 256.
        hidden_dim (int): Hidden layer dimension for projection/prediction heads. Defaults to 4096.
        moving_average_decay (float): Decay factor for the moving-average target encoder. Defaults to 1.0.
        **kwargs: Additional optional arguments.
    """

    def __init__(
        self,
        backbone: nn.Module = None,
        feature_size: int = 2048,
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        moving_average_decay: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        # Default to ResNet-50 backbone if none provided
        if backbone is None:
            base = models.resnet50(weights=None)
            backbone = nn.Sequential(*list(base.children())[:-1])
            feature_size = 2048

        self.encoder_base = backbone
        self.feat_dim = feature_size
        self.proj_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.momentum_decay = moving_average_decay

        # Online (query) encoder
        self.projector = SimCLRProjectionHead(
            input_dim=self.feat_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.proj_dim,
        )
        self.encoder_q = nn.Sequential(self.encoder_base, self.projector)

        # Predictor network
        self.predictor = BYOLPredictionHead(
            input_dim=self.proj_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.proj_dim,
        )

        # Momentum (target) encoder
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self._initialize_target_encoder()

    @torch.no_grad()
    def _initialize_target_encoder(self):
        """Initializes target encoder with query encoder weights (no gradient)."""
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False

    @torch.no_grad()
    def _update_target_encoder(self):
        """Updates target encoder with moving average of query encoder weights."""
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.momentum_decay + p_q.data * (1.0 - self.momentum_decay)

    def forward(self, view_q: torch.Tensor, view_k: torch.Tensor):
        """Encodes and updates momentum encoder."""
        q1 = self.predictor(self.encoder_q(view_q))
        q2 = self.predictor(self.encoder_q(view_k))

        with torch.no_grad():
            self._update_target_encoder()
            k1 = self.encoder_k(view_q)
            k2 = self.encoder_k(view_k)

        return (q1, q2), (k1, k2)


class MoCoV2(nn.Module):
    """MoCo v2: Improved Momentum Contrastive Learning.

    Reference:
        - Paper: https://arxiv.org/abs/2003.04297
        - Code: https://github.com/facebookresearch/moco

    Args:
        backbone (nn.Module, optional): Backbone encoder (default ResNet-50).
        feature_size (int, optional): Feature size from backbone. Defaults to 2048.
        projection_dim (int): Projection head output dimension. Defaults to 128.
        temperature (float): Softmax temperature for contrastive loss. Defaults to 0.07.
        K (int): Size of the negative key queue. Defaults to 65536.
        m (float): Momentum coefficient for key encoder updates. Defaults to 0.999.
        **kwargs: Additional parameters.
    """

    def __init__(
        self,
        backbone: nn.Module = None,
        feature_size: int = 2048,
        projection_dim: int = 128,
        temperature: float = 0.07,
        K: int = 65536,
        m: float = 0.999,
        **kwargs,
    ):
        super().__init__()

        # Default backbone
        if backbone is None:
            resnet = models.resnet50(weights=None)
            backbone = nn.Sequential(*list(resnet.children())[:-1])
            feature_size = 2048

        self.encoder_base = backbone
        self.proj_dim = projection_dim
        self.feat_dim = feature_size
        self.temp = temperature
        self.queue_size = K
        self.momentum = m

        # Build encoders
        self.projector = SimCLRProjectionHead(
            input_dim=self.feat_dim,
            hidden_dim=self.feat_dim,
            output_dim=self.proj_dim,
        )
        self.encoder_q = nn.Sequential(self.encoder_base, self.projector)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self._init_target_encoder()

        # Create and normalize queue
        self.register_buffer("queue", torch.randn(self.proj_dim, self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _init_target_encoder(self):
        """Initialize key encoder parameters from query encoder."""
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of key encoder parameters."""
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.momentum + p_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """Shuffles a batch for contrastive learning (single GPU)."""
        idx_shuffle = torch.randperm(x.shape[0]).cuda()
        idx_unshuffle = torch.argsort(idx_shuffle)
        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """Restores the original batch order after shuffling."""
        return x[idx_unshuffle]

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Maintains the key queue (FIFO)."""
        bsz = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % bsz == 0
        self.queue[:, ptr:ptr + bsz] = keys.t()
        ptr = (ptr + bsz) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, img_q: torch.Tensor, img_k: torch.Tensor):
        """Compute MoCo contrastive outputs."""
        q = nn.functional.normalize(self.encoder_q(img_q), dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            img_k, idx_unshuffle = self._batch_shuffle_single_gpu(img_k)
            k = nn.functional.normalize(self.encoder_k(img_k), dim=1)
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # Compute logits
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temp
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)

        self._dequeue_and_enqueue(k)

        return logits, labels


# Register both methods
register_method(
    name="mocov2",
    model_cls=MoCoV2,
    loss=nn.CrossEntropyLoss,
    transformation=SimCLRViewTransform,
    logs=lambda model, loss_fn: (
        "\n"
        "---------------- MoCoV2 Configuration ----------------\n"
        f"Projection Dimension                : {model.proj_dim}\n"
        f"Number of negative keys             : {model.queue_size}\n"
        f"Momentum for updating key encoder   : {model.momentum}\n"
        "Loss                                : InfoNCE Loss\n"
        "Transformation                      : SimCLRViewTransform"
    ),
)

register_method(
    name="mocov3",
    model_cls=MoCoV3,
    loss=InfoNCE_MoCoV3,
    transformation=SimCLRViewTransform,
    logs=lambda model, loss_fn: (
        "\n"
        "---------------- MoCoV3 Configuration ----------------\n"
        f"Projection Dimension         : {model.proj_dim}\n"
        f"Projection Hidden Dimension  : {model.hidden_dim}\n"
        f"Moving Average Decay         : {model.momentum_decay}\n"
        "Loss                         : InfoNCE Loss\n"
        "Transformation               : SimCLRViewTransform\n"
        "Transformation Prime         : SimCLRViewTransform"
    ),
)
