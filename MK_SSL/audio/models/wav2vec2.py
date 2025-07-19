import random

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from MK_SSL.audio.models.modules.feature_extractors import ConvFeatureExtractor
from MK_SSL.audio.models.modules.backbones import TransformerEncoder
from MK_SSL.audio.models.modules.quantizer import GumbelVectorQuantizer
from MK_SSL.audio.models.modules.heads import Wav2Vec2FeatureProjectionHead
from MK_SSL.audio.models.modules.losses import Wav2Vec2Loss

from MK_SSL.audio.models.utils import register_method


class Wav2Vec2(nn.Module):
    """
    wav2vec 2.0 pretraining model (feature extractor + transformer + quantizer).

    Args:
        variant (str): Model variant to use. One of {"base", "large", "large_lv60k"}.
        quantizer_groups (int): Number of groups in the codebook quantizer.
        quantizer_vars (int): Number of total codebook entries.
        quantizer_temp (float): Initial temperature for the Gumbel softmax quantizer.
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
        super().__init__()
        self.variant = variant

        model_config = self._get_config(self.variant)
        self.model_config = model_config

        self.__quantizer_num_groups = quantizer_num_groups
        self.__quantizer_num_entries_per_codebook = quantizer_num_entries_per_codebook
        
        self.num_mask_time_steps = num_mask_time_steps
        self.mask_time_prob = mask_time_prob

        self.feature_extractor = ConvFeatureExtractor(
            variant=self.model_config["extractor_norm"],
            conv_layers=self.model_config["conv_layers"],
            conv_bias=self.model_config["conv_bias"],
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
            dim=self.model_config["conv_layers"][-1][0],
            num_entries_per_codebook=self.__quantizer_num_entries_per_codebook,
            code_vector_size=self.model_config["code_vector_size"],
            temp=quantizer_temp,
            num_groups=self.__quantizer_num_groups,
            combine_groups=False,
        )

        self.feature_proj = Wav2Vec2FeatureProjectionHead(
            input_dim=self.model_config["conv_layers"][-1][0],
            output_dim=self.model_config["encoder_embed_dim"],
            use_layer_norm=True,
            dropout=self.model_config["encoder_projection_dropout"],
        )

    def forward(
        self,
        waveforms: Tensor,
        lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass through Wav2Vec2.

        Args:
            waveforms (Tensor): Raw audio input, shape (B, T).
            lengths (Tensor): Lengths of each audio sample (before padding), shape (B,).

        Returns:
            context (Tensor): Contextualized representations from encoder (B, T', C).
            quantized (Tensor): Quantized features from vector quantizer (B, T', C).
            mask_indices (BoolTensor): Boolean mask showing which positions were masked (B, T').
            code_perplexity (Tensor): Codebook perplexity, scalar tensor.
            prob_perplexity (Tensor): Probability perplexity, scalar tensor.
            codevector_probs (Tensor): Soft assignment probabilities from quantizer (B, T', G, V).
        """
        # 1. Feature extraction
        hidden_states, lengths = self.feature_extractor(waveforms, lengths)
        
        # 2. Quantization (detach to prevent gradient flow)
        quantized_features, perplexity = self.quantizer(hidden_states, lengths)

        # 3.Project the quantized features to the encoder's embedding dimension
        hidden_states = self.feature_proj(hidden_states)

        # 4. Compute and apply masking
        masked_hidden_states, time_mask_indices = self.time_masking(hidden_states.clone(), lengths)
        
        # 5. Contextualization
        context = self.encoder(masked_hidden_states, lengths)


        return context, quantized_features, perplexity, time_mask_indices
    
    
    def _get_config(self, variant: str) -> dict:
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
                code_vector_size=768,
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
                code_vector_size=1024,
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
                code_vector_size=1024,
            ),
        }

        if not variant:
            raise ValueError("Variant must be specified. Choose from: 'base', 'large', 'large_lv60k'.")
        if variant not in presets:
            raise ValueError(f"Invalid variant: {variant}")
        return presets[variant]

    def time_masking(self, hidden_states: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.BoolTensor]:
        """
        Args:
            hidden_states (torch.Tensor): with shape `(B, L, D)`
            lengths (torch.Tensor): with shape `(B)`

        Returns:
            tuple(
            Masked hidden states (torch.Tensor with shape `(B, L, D)`),
            Time mask (torch.BoolTensor with `(B, L)`)
            )
        """

        batch_size, num_steps, hidden_size = hidden_states.size()

        # non mask: 0, mask: 1
        time_mask_indices = torch.zeros(
            batch_size, num_steps + self.num_mask_time_steps,
            device=hidden_states.device, dtype=torch.bool
        )

        for batch in range(batch_size):
            time_mask_idx_candidates = list(range(int(lengths[batch])))
            k = int(self.mask_time_prob * lengths[batch])
            start_time_idx_array = torch.tensor(
                random.sample(time_mask_idx_candidates, k=k), device=hidden_states.device
            )

            for i in range(self.num_mask_time_steps):
                time_mask_indices[batch, start_time_idx_array+i] = 1

        time_mask_indices = time_mask_indices[:, :-self.num_mask_time_steps]
        num_masks = sum(time_mask_indices.flatten())

        # Mask hidden states
        mask_values = torch.zeros(num_masks, hidden_size, device=hidden_states.device)
        hidden_states[time_mask_indices] = mask_values

        return hidden_states, time_mask_indices
    
    @property
    def quantizer_num_groups(self) -> int:
        return self.__quantizer_num_groups
    
    @property
    def quantizer_num_entries_per_codebook(self) -> int:
        return self.__quantizer_num_entries_per_codebook
    


register_method(
    name= "wav2vec2",
    model_cls= Wav2Vec2,
    loss= Wav2Vec2Loss,
    transformation= None,
    default_params={},
    logs=lambda model, loss: (
        "\n"
        "---------------- Wav2Vec2 Configuration ----------------\n"
        f"Model Variant                     : {model.variant}\n"
        f"Encoder Embedding Dimension       : {model.encoder.embed_dim}\n"
        f"Encoder Layers                    : {model.encoder.num_layers}\n"
        f"Encoder Attention Heads           : {model.encoder.num_heads}\n"
        f"Feedforward Hidden Dimension      : {model.encoder.ff_interm_features}\n"
        f"Feature Projection Dropout        : {model.encoder.dropout_input}\n"
        f"Quantizer Groups                  : {model.quantizer_num_groups}\n"
        f"Entries per Codebook              : {model.quantizer_num_entries_per_codebook}\n"
        f"Code Vector Size             : {model.quantizer.code_vector_size}\n"
        f"Feature Projection Dropout        : {model.feature_proj.dropout}\n"
        "Masking                           : Applied internally on latent features\n"
        "Loss                              : Contrastive + Diversity Loss (Wav2Vec2Loss)"
        f"Loss Temperature                : {loss.temperature}\n"
        f"Loss Alpha                     : {loss.alpha}\n"
    )
)