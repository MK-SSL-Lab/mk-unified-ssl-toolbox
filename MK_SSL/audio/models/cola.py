import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from MK_SSL.audio.models.modules.heads import COLAProjectionHead
from MK_SSL.audio.models.modules.backbones import EfficientNetAudioEncoder

class COLA(nn.Module):
    """
    COLA: Contrastive Learning of General-Purpose Audio Representations.
    
    This class implements the COLA model based on the paper:
    "CONTRASTIVE LEARNING OF GENERAL-PURPOSE AUDIO REPRESENTATIONS"
    (https://arxiv.org/abs/2010.10915)

    COLA learns transferable audio representations via contrastive learning
    on log-compressed mel-filterbank inputs. It maps two audio segments 
    (from the same or different clips) through a shared encoder and a 
    projection head, and computes similarity between the resulting embeddings.

    The encoder can be any convolutional architecture (e.g., EfficientNet),
    and the projection head is a small MLP that transforms the pooled encoder
    outputs to a space suitable for contrastive loss (e.g., NT-Xent).

    Note:
        This model expects pre-computed log-compressed mel-filterbanks 
        as input tensors with shape (B, 1, F, T), where F is the number of 
        mel bands and T is the number of frames.
    """


    def __init__(
        self,
        feature_size: int,
        backbone: nn.Module = None,
        projection_dim: int = 512,
        projection_num_layers: int = 1,
        projection_batch_norm: bool = True,
        **kwargs
    ):
        """
        Args:
            backbone (nn.Module): Backbone model (e.g., EfficientNet-B0) to extract features from log-mel spectrograms.
            feature_size (int): Output feature dimension of the backbone (e.g., 1280 for EfficientNet-B0).
            projection_dim (int): Output dimension of the projection head.
            projection_num_layers (int): Number of layers in the projection head.
            projection_batch_norm (bool): Whether to use normalization (e.g., LayerNorm) in the projection head.
        """
        super().__init__()
        self.feature_size = feature_size
        self.projection_dim = projection_dim
        self.projection_num_layers = projection_num_layers
        self.projection_batch_norm = projection_batch_norm

        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = EfficientNetAudioEncoder()

        self.projection_head = COLAProjectionHead(
            input_dim=self.feature_size,
            hidden_dim=self.feature_size,
            output_dim=self.projection_dim,
            num_layers=self.projection_num_layers,
            batch_norm=self.projection_batch_norm,
        )

    

    def forward(self, x0: torch.Tensor, x1: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for COLA.

        Args:
            x0 (torch.Tensor): First input audio batch as log-mel spectrograms, shape (B, C, F, T).
            x1 (torch.Tensor, optional): Second input audio batch for contrastive training. Defaults to None.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Projected embeddings.
        """
        f0 = self.backbone(x0)           # (B, D, 1, 1) if global pooled
        f0_flat = f0.view(f0.size(0), -1)  # (B, D)
        out0 = self.projection_head(f0_flat)

        if x1 is None:
            return out0

        f1 = self.backbone(x1)
        f1_flat = f1.view(f1.size(0), -1)
        out1 = self.projection_head(f1_flat)

        return out0, out1
