import random
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional

from MK_SSL.audio.models.modules.feature_extractors import ConvFeatureExtractor
from MK_SSL.audio.models.modules.backbones import TransformerEncoder
from MK_SSL.audio.models.modules.quantizer import GumbelVectorQuantizer
from MK_SSL.audio.models.modules.heads import Wav2Vec2FeatureProjectionHead
from MK_SSL.audio.models.modules.losses import Wav2Vec2Loss
from MK_SSL.audio.models.utils import register_method


class Wav2Vec2(nn.Module):
    """wav2vec 2.0 pretraining model.

    This module implements the wav2vec 2.0 architecture for self-supervised speech representation
    learning. It includes a convolutional feature extractor (CNN), feature projection to Transformer
    dimensions, a Transformer encoder for context modeling, and a Gumbel vector quantizer.

    Attributes:
        variant (str): Model variant ("base", "large", "large_lv60k").
        feature_extractor (ConvFeatureExtractor): CNN feature encoder.
        feature_proj (Wav2Vec2FeatureProjectionHead): Projects CNN outputs to Transformer input dim.
        encoder (TransformerEncoder): Transformer-based context encoder.
        quantizer (GumbelVectorQuantizer): Gumbel-softmax quantizer for discrete codebook targets.
        num_mask_time_steps (int): Consecutive time steps to mask during pretraining.
        mask_time_prob (float): Probability of starting a mask span at each timestep.
    """

    def __init__(
        self,
        variant: str = "large",
        quantizer_num_groups: int = 2,
        quantizer_num_entries_per_codebook: int = 320,
        quantizer_temp: float = 2.0,
        num_mask_time_steps: int = 10,
        mask_time_prob: float = 0.065,
        **kwargs
    ):
        """Initializes Wav2Vec2.

        Args:
            variant (str): Model variant, one of {"base", "large", "large_lv60k"}.
            quantizer_num_groups (int): Number of groups in the codebook quantizer.
            quantizer_num_entries_per_codebook (int): Number of codebook entries per group.
            quantizer_temp (float): Initial temperature for Gumbel-softmax quantizer.
            num_mask_time_steps (int): Number of consecutive timesteps to mask.
            mask_time_prob (float): Probability of masking at each timestep.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.variant = variant
        self.model_config = self._get_config(variant)

        self.__quantizer_num_groups = quantizer_num_groups
        self.__quantizer_num_entries_per_codebook = quantizer_num_entries_per_codebook
        self.num_mask_time_steps = num_mask_time_steps
        self.mask_time_prob = mask_time_prob

        self.feature_extractor = ConvFeatureExtractor(
            variant=self.model_config["extractor_norm"],
            conv_layers=self.model_config["conv_layers"],
            conv_bias=self.model_config["conv_bias"],
        )

        self.feature_proj = Wav2Vec2FeatureProjectionHead(
            input_dim=self.model_config["conv_layers"][-1][0],
            output_dim=self.model_config["encoder_embed_dim"],
            use_layer_norm=True,
            dropout=self.model_config["encoder_projection_dropout"],
        )

        self.encoder = TransformerEncoder(
            embed_dim=self.model_config["encoder_embed_dim"],
            num_layers=self.model_config["encoder_num_layers"],
            num_heads=self.model_config["encoder_num_heads"],
            ff_interm_features=self.model_config["encoder_ff_interm_features"],
            dropout_input=self.model_config["encoder_projection_dropout"],
            attention_dropout=self.model_config["encoder_attention_dropout"],
            ff_dropout=self.model_config["encoder_ff_interm_dropout"],
            final_dropout=self.model_config["encoder_dropout"],
            layer_norm_first=self.model_config["encoder_layer_norm_first"],
            layer_drop=self.model_config["encoder_layer_drop"],
            pos_conv_kernel=self.model_config["encoder_pos_conv_kernel"],
            pos_conv_groups=self.model_config["encoder_pos_conv_groups"],
        )

        self.quantizer = GumbelVectorQuantizer(
            dim=self.model_config["encoder_embed_dim"],  # Match projected dimension
            num_entries_per_codebook=self.__quantizer_num_entries_per_codebook,
            code_vector_size=self.model_config["encoder_embed_dim"],
            temp=quantizer_temp,
            num_groups=self.__quantizer_num_groups,
            combine_groups=False,
        )

        self.mask_embedding = nn.Parameter(
            torch.FloatTensor(self.model_config["encoder_embed_dim"]).uniform_()
        )


    def forward(
        self,
        waveforms: Tensor,
        lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass of Wav2Vec2.

        Args:
            waveforms (Tensor): Raw audio input of shape (B, 1, T).
            lengths (Tensor): Lengths of each audio sample before padding, shape (B,).

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
                - Contextualized encoder outputs (B, T', D).
                - Quantized latent vectors (B, T', D).
                - Codebook probabilities (G, V).
                - Boolean mask indices indicating masked positions (B, T').
        """
        if waveforms.dim() != 3 or waveforms.size(1) != 1:
            raise ValueError(f"Expected input shape (B, 1, T), but got {tuple(waveforms.shape)}")

        # CNN feature extraction
        hidden_states, lengths = self.feature_extractor(waveforms, lengths)

        # Project features before quantization
        hidden_states = self.feature_proj(hidden_states)

        # Quantize projected features
        quantized_features, codevector_probs, _ = self.quantizer(hidden_states, lengths)

        # Apply time masking on projected features
        masked_hidden_states, time_mask_indices = self.time_masking(hidden_states.clone(), lengths)

        # Transformer context encoder
        context = self.encoder(masked_hidden_states, lengths)

        return context, quantized_features, codevector_probs, time_mask_indices

    def time_masking(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        channel_mask_prob: float = 0.1,
        channel_mask_width: int = 64,
    ) -> Tuple[Tensor, Tensor]:
        """Apply time and channel masking to hidden states.

        Args:
            hidden_states (Tensor): Projected feature tensor of shape (B, T, D).
            lengths (Tensor): Valid lengths for each batch element, shape (B,).
            channel_mask_prob (float): Probability of applying channel masking.
            channel_mask_width (int): Width (number of consecutive channels) to mask.

        Returns:
            Tuple[Tensor, Tensor]:
                - Masked hidden states of shape (B, T, D).
                - Boolean mask indices indicating time-masked positions (B, T).
        """
        B, T, D = hidden_states.size()


        time_mask_indices = torch.zeros(B, T, device=hidden_states.device, dtype=torch.bool)
        for b in range(B):
            valid_length = int(lengths[b])
            if valid_length <= 1:
                continue
            max_start = max(1, valid_length - self.num_mask_time_steps)
            starts = random.sample(range(max_start), max(1, int(self.mask_time_prob * valid_length)))
            for s in starts:
                end = min(valid_length, s + self.num_mask_time_steps)
                time_mask_indices[b, s:end] = 1

        hidden_states[time_mask_indices] = self.mask_embedding.to(hidden_states.device)

        # Channel masking (SpecAugment-style)
        if random.random() < channel_mask_prob:
            num_channels_to_mask = min(channel_mask_width, D)
            start_channel = random.randint(0, D - num_channels_to_mask)
            hidden_states[:, :, start_channel:start_channel + num_channels_to_mask] = 0.0

        return hidden_states, time_mask_indices


    @property
    def quantizer_num_groups(self) -> int:
        """Returns the number of quantizer groups."""
        return self.__quantizer_num_groups

    @property
    def quantizer_num_entries_per_codebook(self) -> int:
        """Returns the number of entries per quantizer codebook."""
        return self.__quantizer_num_entries_per_codebook

    def _get_config(self, variant: str) -> dict:
        """Returns configuration dictionary for the specified variant."""
        base_conv = [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ]
        presets = {
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
            "large_lv60k": dict(
                extractor_norm="layer_norm",
                conv_layers=base_conv,
                conv_bias=True,
                encoder_embed_dim=1024,
                encoder_projection_dropout=0.1,
                encoder_pos_conv_kernel=128,
                encoder_pos_conv_groups=16,
                encoder_num_layers=24,
                encoder_num_heads=16,
                encoder_attention_dropout=0.0,
                encoder_ff_interm_features=4096,
                encoder_ff_interm_dropout=0.1,
                encoder_dropout=0.0,
                encoder_layer_norm_first=True,
                encoder_layer_drop=0.1,
            ),
        }
        if variant not in presets:
            raise ValueError(f"Invalid variant: {variant}")
        return presets[variant]


register_method(
    name="wav2vec2",
    model_cls=Wav2Vec2,
    loss=Wav2Vec2Loss,
    transformation=None,
    default_params={},
    logs=lambda model, loss: (
        "\n"
        "---------------- Wav2Vec2 Configuration ----------------\n"
        f"Model Variant                : {model.variant}\n"
        f"Encoder Embedding Dimension  : {model.encoder.embed_dim}\n"
        f"Encoder Layers               : {model.encoder.num_layers}\n"
        f"Encoder Attention Heads      : {model.encoder.num_heads}\n"
        f"Feedforward Hidden Dimension : {model.encoder.ff_interm_features}\n"
        f"Feature Projection Dropout   : {model.encoder.dropout_input}\n"
        f"Quantizer Groups             : {model.quantizer_num_groups}\n"
        f"Entries per Codebook         : {model.quantizer_num_entries_per_codebook}\n"
        f"Code Vector Size             : {model.quantizer.code_vector_size}\n"
        f"Loss Temperature             : {loss.temperature}\n"
        f"Loss Num Distractors         : {loss.num_distractors}\n"
        f"Loss Alpha                   : {loss.alpha}\n"
    )
)
