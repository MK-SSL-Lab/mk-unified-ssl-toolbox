import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from MK_SSL.audio.models.modules.feature_extractors import ConvFeatureExtractor
from MK_SSL.audio.models.modules.backbones import TransformerEncoder
from MK_SSL.audio.models.modules.losses import HuBERTLoss

from MK_SSL.audio.models.utils.registry import register_method


class HuBERT(nn.Module):
    """
    HuBERT model for self-supervised speech representation learning.

    Args:
        variant (str): Model variant to use. One of {"base", "large"}.
        mask_prob (float): Probability of masking a given time step.
        mask_length (int): Length of each mask span.
        mask_channel_prob (float): Probability of masking a feature channel.
        mask_channel_length (int): Length of channel mask span.
    """

    def __init__(
        self,
        variant: str = "base",
        mask_prob: float = 0.065,
        mask_length: int = 10,
        mask_channel_prob: float = 0.0,
        mask_channel_length: int = 10,
    ):
        super().__init__()
        config = self.get_config(variant)

        self.feature_extractor = ConvFeatureExtractor(
            variant=config["extractor_norm"],
            conv_layers=config["conv_layers"],
            conv_bias=config["conv_bias"]
        )

        self.encoder = TransformerEncoder(
            embed_dim=config["encoder_embed_dim"],
            num_layers=config["encoder_num_layers"],
            num_heads=config["encoder_num_heads"],
            ff_interm_features=config["encoder_ff_interm_features"],
            dropout_input=config["encoder_projection_dropout"],
            attention_dropout=config["encoder_attention_dropout"],
            ff_dropout=config["encoder_ff_interm_dropout"],
            final_dropout=config["encoder_dropout"],
            layer_norm_first=config["encoder_layer_norm_first"],
            layer_drop=config["encoder_layer_drop"],
            pos_conv_kernel=config["encoder_pos_conv_kernel"],
            pos_conv_groups=config["encoder_pos_conv_groups"],
        )

        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_length = mask_channel_length

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of HuBERT.

        Args:
            x (Tensor): Raw audio waveform, shape (B, T).
            padding_mask (Optional[Tensor]): Padding mask, shape (B, T), with True for padded positions.

        Returns:
            Tuple:
                - context features (Tensor): shape (B, T', D)
                - time mask indices (Tensor): boolean mask of masked positions
                - None (placeholder for potential feature penalty)
        """
        features = self.feature_extractor(x)
        features = features.transpose(1, 2)  # (B, T', D)

        mask_indices = self.compute_mask_indices(
            features.shape[:2],
            mask_prob=self.mask_prob,
            mask_length=self.mask_length,
            device=features.device,
            attention_mask=padding_mask,
        )

        masked_features = features.clone()
        masked_features[mask_indices] = 0.0

        context = self.encoder(masked_features, padding_mask)
        return context, mask_indices, None

    @staticmethod
    def get_config(variant: str) -> dict:
        base_conv = [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ]

        configs = {
            "base": dict(
                extractor_norm="group_norm",
                conv_layers=base_conv,
                conv_bias=False,
                encoder_embed_dim=768,
                encoder_projection_dropout=0.1,
                encoder_pos_conv_kernel=128,
                encoder_pos_conv_groups=16,
                encoder_num_layers=12,
                encoder_num_heads=12,
                encoder_attention_dropout=0.1,
                encoder_ff_interm_features=3072,
                encoder_ff_interm_dropout=0.1,
                encoder_dropout=0.1,
                encoder_layer_norm_first=False,
                encoder_layer_drop=0.1,
            ),
            "large": dict(
                extractor_norm="group_norm",
                conv_layers=base_conv,
                conv_bias=False,
                encoder_embed_dim=1024,
                encoder_projection_dropout=0.1,
                encoder_pos_conv_kernel=128,
                encoder_pos_conv_groups=16,
                encoder_num_layers=24,
                encoder_num_heads=16,
                encoder_attention_dropout=0.1,
                encoder_ff_interm_features=4096,
                encoder_ff_interm_dropout=0.1,
                encoder_dropout=0.1,
                encoder_layer_norm_first=False,
                encoder_layer_drop=0.1,
            ),
        }

        if variant not in configs:
            raise ValueError(f"Unknown HuBERT variant: {variant}")

        return configs[variant]


register_method(
    name= "hubert",
    model_cls= HuBERT,
    loss_fn= HuBERTLoss,
    transformation= None,
    logs=lambda model, loss: (
        "\n"
        "---------------- HuBERT Configuration ----------------\n"
        f"Model Variant                     : {model.variant}\n"
        f"Encoder Embedding Dimension       : {model.encoder.embed_dim}\n"
        f"Encoder Layers                    : {model.encoder.num_layers}\n"
        f"Encoder Attention Heads           : {model.encoder.num_heads}\n"
        f"Feedforward Hidden Dimension      : {model.encoder.ff_interm_features}\n"
        f"Feature Projection Dropout        : {model.encoder.dropout_input}\n"
        f"Time Mask Probability             : {model.mask_prob}\n"
        f"Time Mask Length                  : {model.mask_length}\n"
        f"Channel Mask Probability          : {model.mask_channel_prob}\n"
        f"Channel Mask Length               : {model.mask_channel_length}\n"
        "Loss                              : HuBERT Loss (Cross Entropy over predicted codes)\n"
        f"Loss Reduction                : {loss.reduction}\n"
        "Augmentation                      : Internal latent-space masking only"
    )
)