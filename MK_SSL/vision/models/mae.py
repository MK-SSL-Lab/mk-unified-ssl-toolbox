import torch
import torch.nn as nn
from typing import Optional, Tuple
from MK_SSL.vision.models.modules.mae_encoder import ViTAudioEncoder
from MK_SSL.vision.models.modules.mae_decoder import MAEDecoder
from MK_SSL.vision.models.modules.transformations import MAEVisionTransform
from MK_SSL.vision.models.utils import register_method


class MAE(nn.Module):
    """
    Masked Autoencoder (MAE) for audio spectrogram inputs.

    Based on: "Masked Autoencoders Are Scalable Vision Learners"
    (https://arxiv.org/abs/2111.06377)

    This class uses a ViT-based encoder on visible patches only, and a lightweight
    transformer decoder to reconstruct masked patches from learned latent features.
    """

    def __init__(
        self,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        patch_size: int = 16,
        masking_ratio: float = 0.75,
        input_dim: Tuple[int, int] = (128, 1024),  # (freq, time) dimension
        **kwargs
    ):
        super().__init__()
        self.masking_ratio = masking_ratio
        self.input_dim = input_dim
        self.patch_size = patch_size

        self.encoder = encoder if encoder is not None else ViTAudioEncoder(
            patch_size=self.patch_size, input_shape=input_dim
        )

        self.decoder = decoder if decoder is not None else MAEDecoder(
            embed_dim=self.encoder.embed_dim,
            patch_size=self.patch_size,
            input_dim=input_dim,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input log-mel spectrograms (B, 1, F, T)

        Returns:
            Tuple:
                - reconstruction (torch.Tensor): Reconstructed masked patches
                - target (torch.Tensor): Ground truth of masked patches
                - mask (torch.Tensor): Binary mask indicating which patches were masked
        """
        visible_patches, target, mask, ids_restore = self.encoder.forward_masked(x, self.masking_ratio)
        reconstruction = self.decoder(visible_patches, ids_restore)
        return reconstruction, target, mask


register_method(
    name="mae",
    model_cls=MAE,
    loss=MAEMaskedLoss,
    transformation=MAEVsionTransform,
    params={},
    logs=lambda model, loss: (
        "\n"
        "---------------- MAE Configuration ----------------\n"
        f"Input Type                       : Log-mel spectrograms (B, 1, F, T)\n"
        f"Encoder Architecture             : {model.encoder.__class__.__name__}\n"
        f"Decoder Architecture             : {model.decoder.__class__.__name__}\n"
        f"Masking Ratio                    : {model.masking_ratio}\n"
        "Loss                             : MAE Masked MSE\n"
        "Augmentation                     : MAEAudioTransform"
    )
)
