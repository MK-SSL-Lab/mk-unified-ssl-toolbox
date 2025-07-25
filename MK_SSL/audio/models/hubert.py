import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple, Optional

from MK_SSL.audio.models.modules.feature_extractors import ConvFeatureExtractor
from MK_SSL.audio.models.modules.backbones import TransformerEncoder
from MK_SSL.audio.models.modules.losses import HuBERTLoss
from MK_SSL.audio.models.utils import register_method
from MK_SSL.audio.models.modules.heads import HuBERTProjectionHead


class HuBERT(nn.Module):
    """HuBERT model for self-supervised speech representation learning."""

    def __init__(
        self,
        variant: str = "base",
        mask_prob: float = 0.065,
        mask_length: int = 10,
        mask_channel_prob: float = 0.0,
        mask_channel_length: int = 10,
        num_clusters: int = 100,
        init_from_mfcc: bool = True,
        sample_rate: int = 16000,
        **kwargs
    ):
        super().__init__()
        self.variant = variant
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_length = mask_channel_length
        self.num_clusters = num_clusters
        self.init_from_mfcc = init_from_mfcc
        self.sample_rate = sample_rate

        config = self._get_config(variant)
        self.config = config

        # Feature extractor
        self.feature_extractor = ConvFeatureExtractor(
            variant="layer_norm",
            conv_layers=config["conv_layers"],
        )
        feature_extractor_output_dim = config["feature_extractor_output_dim"]

        # Feature projection to encoder dimension
        self.feature_projection = nn.Linear(
            feature_extractor_output_dim, config["encoder_embed_dim"]
        )
        self.post_extract_proj_norm = nn.LayerNorm(config["encoder_embed_dim"])
        self.post_extract_proj_dropout = nn.Dropout(config["encoder_dropout_input"])

        # Transformer Encoder
        self.encoder = TransformerEncoder(
            embed_dim=config["encoder_embed_dim"],
            num_layers=config["encoder_num_layers"],
            num_heads=config["encoder_num_heads"],
            ff_interm_features=config["encoder_ff_interm_features"],
            dropout_input=config["encoder_dropout_input"],
            attention_dropout=config["encoder_attention_dropout"],
            ff_dropout=config["encoder_activation_dropout"],
            final_dropout=config["encoder_dropout"],
            layer_norm_first=config["encoder_layer_norm_first"],
            layer_drop=config["encoder_layer_drop"],
            pos_conv_kernel=128,
            pos_conv_groups=16,
        )

        # Projection and prediction heads
        self.projection_head = HuBERTProjectionHead(
            input_dim=config["encoder_embed_dim"],
            output_dim=config.get("projection_dim", 256),
            dropout=config["encoder_dropout"]
        )
        self.prediction_head = nn.Linear(config.get("projection_dim", 256), num_clusters)

        # Mask embedding
        self.mask_embedding = nn.Parameter(
            torch.FloatTensor(config["encoder_embed_dim"]).uniform_(-0.1, 0.1)
        )

    def _get_config(self, variant: str):
        configs = {
            "base": dict(
                conv_layers=[
                    (512, 10, 5), (512, 3, 2), (512, 3, 2),
                    (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)
                ],
                feature_extractor_output_dim=512,
                encoder_embed_dim=768,
                encoder_ff_interm_features=3072,
                encoder_num_layers=12,
                encoder_num_heads=8,
                encoder_dropout_input=0.1,
                encoder_attention_dropout=0.1,
                encoder_activation_dropout=0.0,
                encoder_dropout=0.1,
                encoder_layer_norm_first=False,
                encoder_layer_drop=0.1,
                projection_dim=256,
                max_iterations=2,
                extractor_layer=6,
                pseudo_label_sample_ratio=0.1,
            ),
            "large": dict(
                conv_layers=[
                    (512, 10, 5), (512, 3, 2), (512, 3, 2),
                    (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)
                ],
                feature_extractor_output_dim=512,
                encoder_embed_dim=1024,
                encoder_ff_interm_features=4096,
                encoder_num_layers=24,
                encoder_num_heads=16,
                encoder_dropout_input=0.1,
                encoder_attention_dropout=0.1,
                encoder_activation_dropout=0.0,
                encoder_dropout=0.1,
                encoder_layer_norm_first=False,
                encoder_layer_drop=0.1,
                projection_dim=256,
                max_iterations=2,
                extractor_layer=9,
                pseudo_label_sample_ratio=0.1,
            ),
            "x-large": dict(
                conv_layers=[
                    (512, 10, 5), (512, 3, 2), (512, 3, 2),
                    (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)
                ],
                feature_extractor_output_dim=512,
                encoder_embed_dim=1280,
                encoder_ff_interm_features=5120,
                encoder_num_layers=48,
                encoder_num_heads=16,
                encoder_dropout_input=0.1,
                encoder_attention_dropout=0.1,
                encoder_activation_dropout=0.0,
                encoder_dropout=0.1,
                encoder_layer_norm_first=False,
                encoder_layer_drop=0.1,
                projection_dim=256,
                max_iterations=2,
                extractor_layer=9,
                pseudo_label_sample_ratio=0.1,
            ),
        }
        if variant not in configs:
            raise ValueError(f"Unknown HuBERT variant: {variant}")
        return configs[variant]

    def _apply_masking(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply non-overlapping time-based masking to the feature sequence.
        Follows the logic of FAIRSEQ's `_compute_mask_indices`.

        Args:
            features (Tensor): Feature tensor of shape (B, T, C).

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - masked_features: Features with masked positions replaced by self.mask_embedding.
                - mask_indices: Boolean tensor of shape (B, T) where True indicates a masked time step.
                - masked_lengths: 1D tensor of shape (B,) with the count of masked frames per sample.
        """
        B, T, C = features.shape
        mask_indices = torch.zeros((B, T), dtype=torch.bool, device=features.device)
        masked_lengths = torch.zeros(B, dtype=torch.long, device=features.device)

        # Number of masks per sequence
        num_masks = max(1, int(self.mask_prob * T / self.mask_length))

        for i in range(B):
            spans = []
            attempts = 0
            while len(spans) < num_masks and attempts < num_masks * 5:
                start = torch.randint(0, max(1, T - self.mask_length + 1), (1,)).item()
                end = min(T, start + self.mask_length)

                # Avoid overlapping spans
                if all(end <= s or start >= e for s, e in spans):
                    spans.append((start, end))
                attempts += 1

            # Set mask indices
            for s, e in spans:
                mask_indices[i, s:e] = True
            masked_lengths[i] = sum(e - s for s, e in spans)

        # Apply mask embedding
        mask_embed = self.mask_embedding.to(features.device)
        masked_features = features.clone()
        masked_features[mask_indices] = mask_embed

        return masked_features, mask_indices, masked_lengths


    def forward(
        self,
        waveforms: Tensor,
        lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    
        if waveforms.dim() != 3 or waveforms.size(1) != 1:
            raise ValueError(f"Expected input shape (B, 1, T), but got {tuple(waveforms.shape)}")
        
        features, lengths = self.feature_extractor(waveforms, lengths)


        # Projection to encoder
        features = self.feature_projection(features)
        features = self.post_extract_proj_norm(features)
        features = self.post_extract_proj_dropout(features)

        # Apply masking
        features, mask_indices, masked_lengths = self._apply_masking(features)
        
        # Transformer encoder
        encoder_outputs = self.encoder(features, lengths)

        # Projection head
        projected_outputs = self.projection_head(encoder_outputs)

        # Only masked logits
        masked_encoder_outputs = projected_outputs[mask_indices]
        logits = self.prediction_head(masked_encoder_outputs)

        return logits, mask_indices, lengths, masked_lengths


register_method(
    name="hubert",
    model_cls=HuBERT,
    loss=HuBERTLoss,
    transformation=None,
    default_params={
        "init_from_mfcc": True,
        "sample_rate": 16000,
    },
    logs=lambda model, loss: (
        "\n"
        "---------------- HuBERT Configuration ----------------\n"
        f"Model Variant                     : {model.variant}\n"
        f"Encoder Embedding Dimension       : {model.encoder.embed_dim}\n"
        f"Encoder Layers                    : {model.encoder.num_layers}\n"
        f"Encoder Attention Heads           : {model.encoder.num_heads}\n"
        f"Feedforward Hidden Dimension      : {model.encoder.ff_interm_features}\n"
        f"Feature Projection Dropout        : {model.post_extract_proj_dropout.p}\n"
        f"Time Mask Probability             : {model.mask_prob}\n"
        f"Time Mask Length                  : {model.mask_length}\n"
        f"Channel Mask Probability          : {model.mask_channel_prob}\n"
        f"Channel Mask Length               : {model.mask_channel_length}\n"
        f"Number of Clusters (Prediction Head Output): {model.num_clusters}\n"
        f"Extractor Layer for Subsequent Iterations: {model.config['extractor_layer']}\n"
        "Loss                              : HuBERT Loss (Cross Entropy over predicted codes)\n"
        f"Loss Reduction                    : {loss.reduction}\n"
        "Augmentation                      : Internal latent-space masking only"
    )
)
