import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from MK_SSL.audio.models.modules.heads import COLAProjectionHead
from MK_SSL.audio.models.modules.backbones import EfficientNetAudioEncoder

from MK_SSL.audio.models.modules.losses import InfoNCELoss
from MK_SSL.audio.models.modules.transformations import COLAAudioTransform

from MK_SSL.audio.models.utils.registry import register_method


class COLA(nn.Module):
    """Contrastive Learning of General‑Purpose Audio Representations (COLA).

    Based on:
        Saeed, A., et al. *Contrastive Learning of General‑Purpose Audio
        Representations*. 2021. https://arxiv.org/abs/2010.10915

    This model receives raw mono audio waveforms and produces 512‑dimensional
    embeddings for contrastive pre‑training.

    Workflow:
        1. Two 960 ms segments are randomly cropped from each input waveform.
        2. Each segment is converted to a log‑mel spectrogram and fed to an
        EfficientNet‑B0 backbone.
        3. Global max‑pooled features are projected to 512‑D embeddings.
        4. An InfoNCE loss maximises agreement between paired segments.

    Attributes:
        feature_size (int): Backbone output dimension.
        projection_dim (int): Projection head output dimension.

    Args:
        feature_size (int, optional): Dimension of the backbone output.
            Defaults to 1280.
        backbone (nn.Module, optional): Feature‑extractor backbone. If ``None``,
            ``EfficientNetAudioEncoder`` is used. Defaults to ``None``.
        projection_dim (int, optional): Projection head output dimension.
            Defaults to 512.
        projection_num_layers (int, optional): Number of layers in the projection
            head. Defaults to 1.
        projection_batch_norm (bool, optional): If ``True``, applies batch
            normalisation in the projection head. Defaults to ``True``.

    Inputs:
        x0 (Tensor): Waveform segment of shape ``(B, 1, T)``.
        x1 (Tensor, optional): Second waveform segment of shape ``(B, 1, T)``.

    Returns:
        Tensor | Tuple[Tensor, Tensor]: If ``x1`` is provided, returns
        ``(emb0, emb1)`` each of shape ``(B, 512)``; otherwise a single
        tensor of shape ``(B, 512)``.

    Raises:
        ValueError: If input tensor shapes are invalid.
    """



    def __init__(
        self,
        feature_size: int = 1280,
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

    

    def forward(
        self,
        x0: torch.Tensor,
        x1: Optional[torch.Tensor] = None,
        lengths0: Optional[torch.Tensor] = None,
        lengths1: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for COLA with support for padding masks.

        Args:
            x0 (torch.Tensor):
                First audio view as waveform segments, shape (B, 1, T).
            x1 (torch.Tensor, optional):
                Second audio view for contrastive training, shape (B, 1, T).
            lengths0 (torch.Tensor, optional):
                Original (unpadded) lengths of the raw audio waveforms corresponding
                to `x0`, shape (B,). Used to mask out padded frames so they do not
                influence the pooled embeddings. If ``None``, no masking is applied.
            lengths1 (torch.Tensor, optional):
                Original (unpadded) lengths of the raw audio waveforms corresponding
                to `x1`, shape (B,). Used the same way as `lengths0`.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
                - If only `x0` is provided: embedding tensor of shape (B, projection_dim).
                - If both `x0` and `x1` are provided: tuple of embeddings
                `(out0, out1)`, each of shape (B, projection_dim).

        Notes:
            - The `lengths` arguments refer to the **full original waveform length**
            before any cropping in `COLAAudioTransform`. The masking logic ensures
            that padded audio samples (zeros) from dataset preprocessing do not
            affect the representation.
        """
        f0 = self.backbone(x0, lengths=lengths0)
        out0 = self.projection_head(f0)

        if x1 is None:
            return out0

        f1 = self.backbone(x1, lengths=lengths1)
        out1 = self.projection_head(f1)

        return out0, out1




register_method(
    name= "cola",
    model_cls= COLA,
    loss= InfoNCELoss,
    transformation= COLAAudioTransform,
    default_params={},
    logs=lambda model, loss: (
        "\n"
        "---------------- COLA Configuration ----------------\n"
        f"Input Type                       : Log-mel spectrograms (B, 1, F, T)\n"
        f"Backbone Architecture            : {model.backbone.__class__.__name__}\n"
        "Loss                             : InfoNCE Loss\n"
        "Augmentation                     : COLAAudioTransform"

    )
)