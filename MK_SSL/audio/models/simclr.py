import torch
import torch.nn as nn
from typing import Optional, Tuple

from MK_SSL.audio.models.modules.heads import (
    InputSpeechSimCLRProjectionHead,
    SpeechSimCLRProjectionHead,
)
from MK_SSL.audio.models.modules.feature_extractors import FBANKFeatureExtractor
from MK_SSL.audio.models.modules.backbones import TransformerEncoder
from MK_SSL.audio.models.modules.losses.NTXent_loss import NTXentLoss
from MK_SSL.audio.models.modules.transformations import SimCLRAudioTransform
from MK_SSL.audio.models.utils import register_method


class SimCLRSpeech(nn.Module):
    """SimCLR framework for self-supervised speech representation learning.

    Raw waveforms → FBANK → (optional) input projection → Transformer backbone
    → projection head → contrastive loss.

    Args:
        embed_dim (int, optional): Transformer embedding dimension.
        projection_dim (int, optional): Output dimension of projection head.
        projection_num_layers (int, optional): Layers in projection head.
        projection_batch_norm (bool, optional): Use LayerNorm in projection head.
        backbone (nn.Module, optional): Custom backbone; defaults to 3-layer
            Transformer encoder.
        sample_rate (int, optional): Sampling rate for FBANK extraction.
        use_input_proj (bool, optional): Insert learnable projection when
            ``embed_dim != 80``.
        input_proj_dropout (float, optional): Dropout after input projection.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        projection_dim: int = 128,
        projection_num_layers: int = 2,
        projection_batch_norm: bool = True,
        backbone: Optional[nn.Module] = None,
        sample_rate: int = 16000,
        use_input_proj: bool = True,
        input_proj_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.fbank = FBANKFeatureExtractor(sample_rate=sample_rate)
        self.embed_dim = embed_dim
        if embed_dim != 80 and use_input_proj:
            self.input_proj = InputSpeechSimCLRProjectionHead(
                input_dim=80,
                output_dim=embed_dim,
                use_layer_norm=True,
                dropout=input_proj_dropout,
            )
        else:
            self.input_proj = nn.Identity()

        self.backbone = backbone or TransformerEncoder(
            embed_dim=embed_dim,
            num_layers=3,
            num_heads=12,
            ff_interm_features=3072,
            dropout_input=0.1,
            attention_dropout=0.1,
            ff_dropout=0.1,
            final_dropout=0.1,
            layer_norm_first=True,
            layer_drop=0.0,
            pos_conv_kernel=128,
            pos_conv_groups=16,
        )

        self.projection_head = SpeechSimCLRProjectionHead(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=projection_dim,
            num_layers=projection_num_layers,
            batch_norm=projection_batch_norm,
        )

    # ------------------------------------------------------------------ #
    # Encoding / Forward                                                 #
    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single view.

        Args:
            x (torch.Tensor): Raw waveform of shape ``(B, 1, T)``.

        Returns:
            torch.Tensor: Projected embedding of shape ``(B, projection_dim)``.
        """
        x = self.fbank(x)          # (B, T', 80)
        x = self.input_proj(x)     # (B, T', embed_dim)
        x = self.backbone(x)       # (B, T', embed_dim)
        x = x.mean(dim=1)          # (B, embed_dim)
        return self.projection_head(x)

    def forward(
        self,
        x0: torch.Tensor,
        x1: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for one or two augmented views.

        Args:
            x0 (torch.Tensor): First view, shape ``(B, 1, T)``.
            x1 (torch.Tensor, optional): Second view, same shape as ``x0``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor] or torch.Tensor:
                - If ``x1`` provided: two embeddings ``(out0, out1)``.
                - Else: only ``out0``.
        """
        out0 = self.encode(x0)
        if x1 is None:
            return out0
        out1 = self.encode(x1)
        return out0, out1


register_method(
    name="simclr",
    model_cls=SimCLRSpeech,
    loss=NTXentLoss,
    transformation=SimCLRAudioTransform,
    default_params={},
    logs=lambda model, loss: (
        "\n"
        "---------------- SimCLRSpeech Configuration ----------------\n"
        "Input Feature Dimension           : 80 (FBANK)\n"
        f"Input Projection Used             : {not isinstance(model.input_proj, nn.Identity)}\n"
        f"Backbone Architecture             : {model.backbone.__class__.__name__}\n"
        "Loss                              : NT-Xent\n"
        "Augmentation                      : SimCLRAudioTransform"
    ),
)
