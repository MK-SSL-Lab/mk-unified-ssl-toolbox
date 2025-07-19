import torch
import torch.nn as nn
from typing import Optional, Tuple

from MK_SSL.audio.models.modules.heads import InputSpeechSimCLRProjectionHead, SpeechSimCLRProjectionHead
from MK_SSL.audio.models.modules.feature_extractors import FBANKFeatureExtractor
from MK_SSL.audio.models.modules.backbones import TransformerEncoder
from MK_SSL.audio.models.modules.losses import NTXent_loss
from MK_SSL.audio.models.modules.transformations import SimCLRAudioTransform


from MK_SSL.audio.models.utils import register_method


class SimCLRSpeech(nn.Module):
    """
    SimCLR for Speech: A framework for contrastive learning of speech representations.
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
        **kwargs
    ):
        """
        Args:
            embed_dim (int): Dimension of Transformer input/output.
            projection_dim (int): Output dimension of projection head.
            projection_num_layers (int): Number of layers in projection head.
            projection_batch_norm (bool): Use LayerNorm in projection head.
            backbone (nn.Module): Transformer encoder backbone.
            sample_rate (int): Audio sample rate for FBANK extraction.
            use_input_proj (bool): Whether to apply a learnable input projection.
            input_proj_dropout (float): Dropout after input projection.
        """
        super().__init__()
        self.fbank = FBANKFeatureExtractor(sample_rate=sample_rate)

        if embed_dim != 80 and use_input_proj:
            self.input_proj = InputSpeechSimCLRProjectionHead(
                input_dim=80,
                output_dim=embed_dim,
                use_layer_norm=True,
                dropout=input_proj_dropout,
            )
        else:
            self.input_proj = nn.Identity()


        if backbone is not None:
            self.backbone = backbone

        else:
            self.backbone = TransformerEncoder(
                embed_dim=embed_dim,
                num_layers=3,
                num_heads=12,
                ff_interm_features=3072,
                dropout_input=0.1,
                attention_dropout=0.1,
                ff_dropout=0.1,
                final_dropout=0.1,
                layer_norm_first=True,   # the paper switched all BatchNorm to LayerNorm; Pre-LN is fine
                layer_drop=0.0,
                pos_conv_kernel=7,    # not used in the paper; standard sinusoidal positional enc. suffices
                pos_conv_groups=None,
            )

        self.projection_head = SpeechSimCLRProjectionHead(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=projection_dim,
            num_layers=projection_num_layers,
            batch_norm=projection_batch_norm,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns pooled projection from input audio."""
        x = self.fbank(x)                  # (B, T, 80)
        x = self.input_proj(x)            # (B, T, embed_dim)
        x = self.backbone(x)              # (B, T, embed_dim)
        x = x.mean(dim=1)                 # mean pooling → (B, embed_dim)
        return self.projection_head(x)    # (B, projection_dim)

    def forward(self, x0: torch.Tensor, x1: Optional[torch.Tensor] = None):
        """
        Forward pass for SimCLR.

        Args:
            x0 (torch.Tensor): First audio sample (B, T).
            x1 (torch.Tensor, optional): Second audio sample (B, T).

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        """
        out0 = self.encode(x0)

        if x1 is None:
            return out0

        out1 = self.encode(x1)
        return out0, out1


register_method(
    name= "simclr",
    model_cls= SimCLRSpeech,
    loss= NTXent_loss,
    transformation= SimCLRAudioTransform,
    default_params={},
    logs=lambda model, loss: (
        "\n"
        "---------------- SimCLRSpeech Configuration ----------------\n"
        f"Input Feature Dimension           : 80 (FBANK)\n"
        f"Input Projection Used             : {not isinstance(model.input_proj, nn.Identity)}\n"
        f"Backbone Architecture            : {model.backbone.__class__.__name__}\n"
        "Loss                              : NT-Xent (Normalized Temperature-scaled Cross Entropy)\n"
        "Augmentation                      : SimCLRAudioTransform"

    )
)