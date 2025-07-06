import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from MK_SSL.audio.models.modules.feature_extractors import ConvFeatureExtractor
from MK_SSL.audio.models.modules.backbones import TransformerEncoder
from MK_SSL.audio.models.modules.losses import HuBERTLoss
from dataclasses import dataclass

from MK_SSL.audio.models.utils import register_method


@dataclass
class HubertConfig:
    variant: str = "base"
    mask_prob: float = 0.065
    mask_length: int = 10
    mask_channel_prob: float = 0.0
    mask_channel_length: int = 10
    num_clusters: int = 100
    kmeans_seed: int = 0
    init_from_mfcc: bool = True
    # extractor_layer: int = 6 # This will now be determined by the variant
    sample_rate: int = 16000
    lr: float = 1e-4
    epochs: int = 10
    iterations: int = 2 # Setting default to 2 based on paper (total cycles)


class HuBERT(nn.Module):
    """
    HuBERT model for self-supervised speech representation learning.

    Args:
        variant (str): Model variant to use. One of {"base", "large", "x-large"}.
        mask_prob (float): Probability of masking a given time step.
        mask_length (int): Length of each mask span.
        mask_channel_prob (float): Probability of masking a feature channel.
        mask_channel_length (int): Length of channel mask span.
        num_clusters (int): Number of clusters for the prediction head output.
    """

    def __init__(
        self,
        variant: str = "base",
        mask_prob: float = 0.065,
        mask_length: int = 10,
        mask_channel_prob: float = 0.0,
        mask_channel_length: int = 10,
        num_clusters: int = 100,
    ):
        super().__init__()
        self.variant = variant
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_length = mask_channel_length
        self.num_clusters = num_clusters

        # Get configuration based on the variant
        model_config = self._get_config(variant)

        # Store variant-specific config for easy access in logs/elsewhere
        self.model_config = model_config

        # Convolutional Feature Extractor
        self.feature_extractor = ConvFeatureExtractor(
            conv_layers=model_config["conv_layers"],
            conv_dropout=model_config["conv_dropout"],
        )
        feature_extractor_output_dim = model_config["feature_extractor_output_dim"]

        # Feature projection for Transformer input
        self.feature_projection = nn.Linear(
            feature_extractor_output_dim, model_config["encoder_embed_dim"]
        )
        self.post_extract_proj_norm = nn.LayerNorm(model_config["encoder_embed_dim"])
        self.post_extract_proj_dropout = nn.Dropout(model_config["encoder_dropout_input"])


        # Transformer Encoder
        self.encoder = TransformerEncoder(
            embed_dim=model_config["encoder_embed_dim"],
            ff_interm_features=model_config["encoder_ff_interm_features"],
            num_layers=model_config["encoder_num_layers"],
            num_heads=model_config["encoder_num_heads"],
            dropout=model_config["encoder_dropout"],
            attention_dropout=model_config["encoder_attention_dropout"],
            activation_dropout=model_config["encoder_activation_dropout"],
            encoder_layer_norm_first=model_config["encoder_layer_norm_first"],
            encoder_layer_drop=model_config["encoder_layer_drop"],
        )

        # Prediction Head (maps Transformer output to number of clusters)
        self.prediction_head = nn.Linear(model_config["encoder_embed_dim"], num_clusters)


    def _get_config(self, variant: str):
        """
        Returns model configuration parameters based on the specified variant.
        """
        configs = {
            "base": dict(
                conv_layers=[
                    (512, 10, 5), (512, 3, 2), (512, 3, 2),
                    (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)
                ],
                conv_dropout=0.0,
                feature_extractor_output_dim=512, # Output of the last conv layer
                encoder_embed_dim=768,
                encoder_ff_interm_features=3072,
                encoder_num_layers=12,
                encoder_num_heads=8, # Corrected based on your image!
                encoder_dropout_input=0.1,
                encoder_attention_dropout=0.1,
                encoder_activation_dropout=0.0,
                encoder_dropout=0.1,
                encoder_layer_norm_first=False,
                encoder_layer_drop=0.1,
                extractor_layer=6, # For subsequent iterations of base
            ),
            "large": dict(
                conv_layers=[
                    (512, 10, 5), (512, 3, 2), (512, 3, 2),
                    (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)
                ],
                conv_dropout=0.0,
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
                extractor_layer=9, # For subsequent iterations of large
            ),
            "x-large": dict(
                conv_layers=[
                    (512, 10, 5), (512, 3, 2), (512, 3, 2),
                    (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)
                ],
                conv_dropout=0.0,
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
                extractor_layer=9, # For subsequent iterations of x-large
            ),
        }

        if variant not in configs:
            raise ValueError(f"Unknown HuBERT variant: {variant}")

        return configs[variant]

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_features_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            source (torch.Tensor): Input audio tensor of shape (B, T_audio_samples).
            padding_mask (torch.Tensor, optional): Boolean mask for padding, shape (B, T_audio_samples).
                                                True indicates a padded position.
            return_features_only (bool): If True, only returns the raw features
                                         from the feature extractor. Useful for K-means.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - logits (Tensor): Predictions for masked tokens (B, T_masked, num_clusters)
                - mask_indices (Tensor): Boolean mask of shape (B, T_features)
                - feature_lengths (Tensor): Actual lengths of features after convolution (B,)
                - masked_lengths (Tensor): Number of masked tokens per batch item (B,)
        """
        # 1. Feature Extraction
        features = self.feature_extractor(source) # (B, C, T')
        features = features.transpose(1, 2) # (B, T', C)

        # If a padding mask is provided for the raw audio, we need to adjust it
        # for the downsampled features.
        feature_lengths = torch.sum(~padding_mask, dim=1) if padding_mask is not None else None
        if feature_lengths is not None:
            # Adjust padding mask length based on feature extractor's total stride
            feature_lengths = self.feature_extractor.get_output_lengths(source.shape[1], source.device)
            # Create a new padding mask for the features
            feature_padding_mask = torch.arange(features.size(1), device=source.device).unsqueeze(0) >= feature_lengths.unsqueeze(1)
        else:
            feature_padding_mask = None


        if return_features_only:
            return features, feature_padding_mask # Return features before projection/masking

        # 2. Feature Projection and Normalization/Dropout
        features = self.feature_projection(features)
        features = self.post_extract_proj_norm(features)
        features = self.post_extract_proj_dropout(features)

        # 3. Masking
        # Get masking indices and apply masking
        batch_size, seq_len, _ = features.shape
        mask_indices = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=features.device)
        masked_lengths = torch.zeros(batch_size, dtype=torch.long, device=features.device)

        for i in range(batch_size):
            # Apply time masking
            num_masked_spans = int(seq_len * self.mask_prob / self.mask_length)
            masked_spans = []
            for _ in range(num_masked_spans):
                start = torch.randint(0, seq_len - self.mask_length + 1, (1,)).item()
                end = start + self.mask_length
                masked_spans.extend(range(start, end))

            # Ensure masked_spans don't exceed seq_len or padding mask
            if feature_padding_mask is not None:
                valid_mask_indices = torch.where(~feature_padding_mask[i])[0]
                if len(valid_mask_indices) > 0:
                    max_valid_idx = valid_mask_indices[-1].item()
                    masked_spans = [idx for idx in masked_spans if idx <= max_valid_idx]
                else:
                    masked_spans = [] # No valid positions to mask

            mask_indices[i, masked_spans] = True
            masked_lengths[i] = len(masked_spans)
            features[i, mask_indices[i]] = 0.0 # Set masked positions to zero or learned mask embedding

        # 4. Transformer Encoder
        # The padding_mask for the Transformer needs to be True for padded/masked tokens
        # Combine feature_padding_mask (if exists) and mask_indices
        transformer_padding_mask = mask_indices
        if feature_padding_mask is not None:
            transformer_padding_mask = transformer_padding_mask | feature_padding_mask

        encoder_outputs = self.encoder(features, padding_mask=transformer_padding_mask)
        # encoder_outputs is typically (B, T', D)

        # 5. Prediction Head
        masked_encoder_outputs = encoder_outputs[mask_indices] # (N_masked, D)

        logits = self.prediction_head(masked_encoder_outputs) # (N_masked, num_clusters)

        # Reshape logits to (B, T_features, num_clusters) to align with targets for loss calculation
        full_logits = torch.zeros(
            (batch_size, seq_len, self.num_clusters),
            dtype=logits.dtype,
            device=logits.device
        )
        full_logits[mask_indices] = logits


        return full_logits, mask_indices, feature_lengths, masked_lengths


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
        f"Feature Projection Dropout        : {model.post_extract_proj_dropout.p}\n"
        f"Time Mask Probability             : {model.mask_prob}\n"
        f"Time Mask Length                  : {model.mask_length}\n"
        f"Channel Mask Probability          : {model.mask_channel_prob}\n"
        f"Channel Mask Length               : {model.mask_channel_length}\n"
        f"Number of Clusters (Prediction Head Output): {model.num_clusters}\n"
        f"Extractor Layer for Subsequent Iterations: {model.model_config['extractor_layer']}\n" # Added
        "Loss                              : HuBERT Loss (Cross Entropy over predicted codes)\n"
        f"Loss Reduction                    : {loss.reduction}\n"
        "Augmentation                      : Internal latent-space masking only"
    )
)