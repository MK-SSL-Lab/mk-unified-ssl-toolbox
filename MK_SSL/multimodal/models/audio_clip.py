import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

from MK_SSL.multimodal.models.modules.backbones import AudioResNeXtStem
from MK_SSL.multimodal.models.modules.backbones import CLIPImageEncoder
from MK_SSL.multimodal.models.modules.backbones import CLIPTextEncoder
from MK_SSL.multimodal.models.modules.losses import AudioCLIPLoss
from MK_SSL.multimodal.models.utils import register_method



class AudioCLIP(nn.Module):
    """
    AudioCLIP-style contrastive pretraining for joint audio, image, and text embeddings.
    """


    def __init__(
        self,
        audio_encoder: Optional[nn.Module] = None,
        image_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        temperature_init: float = 0.07,
        text_template: str = "{}",
        device: str = 'cpu',
        **kwargs

    ):

        super().__init__()
        self.device = device

        self.audio_encoder = audio_encoder
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.temperature = nn.Parameter(torch.tensor(temperature_init))
        self.text_template = text_template

        if audio_encoder is not None:
            self.audio_encoder = audio_encoder
        else:
            self.audio_encoder = AudioResNeXtStem()

        if image_encoder is not None:
            self.image_encoder = image_encoder
        else:
            self.image_encoder = CLIPImageEncoder(device=self.device, model_name = "RN50")

        if text_encoder is not None:
            self.text_encoder = text_encoder
        else:
            self.text_encoder = CLIPTextEncoder(device=self.device)

        self.audio_clip_loss = AudioCLIPLoss()

    def forward(
        self,
        audio_input: Optional[torch.Tensor] = None,
        image_input: Optional[torch.Tensor] = None,
        text_input: Optional[List[str]] = None,
    ) -> Tuple[
        Optional[torch.Tensor], 
        Optional[torch.Tensor], 
        Optional[torch.Tensor], 
        Optional[torch.Tensor], 
        Optional[torch.Tensor], 
        Optional[torch.Tensor],  
    ]:
        """
        Forward pass for contrastive learning over available modality pairs.

        Args:
            audio_input (Optional[Tensor]): Audio input batch.
            image_input (Optional[Tensor]): Image input batch.
            text_input (Optional[List[str]]): Raw text inputs.

        Returns:
            Tuple of:
                - audio_emb: Normalized audio embeddings.
                - image_emb: Normalized image embeddings.
                - text_emb: Normalized text embeddings.
                - sim_text_audio: Similarity (text, audio).
                - sim_text_image: Similarity (text, image).
                - sim_audio_image: Similarity (audio, image).
        """
        audio_emb = image_emb = text_emb = None
        sim_text_audio = sim_text_image = sim_audio_image = None

        # encoder
        if audio_input is not None:
            audio_emb = F.normalize(self.audio_encoder(audio_input), dim=-1)

        if image_input is not None:
            image_emb = F.normalize(self.image_encoder(image_input), dim=-1)

        if text_input is not None:
            text_emb = F.normalize(self.text_encoder(text_input), dim=-1)


        # similarity matrix
        if text_emb is not None and audio_emb is not None:
            sim_text_audio = self.temperature * torch.matmul(text_emb, audio_emb.T)

        if text_emb is not None and image_emb is not None:
            sim_text_image = self.temperature * torch.matmul(text_emb, image_emb.T)

        if audio_emb is not None and image_emb is not None:
            sim_audio_image = self.temperature * torch.matmul(audio_emb, image_emb.T)

        return (
            audio_emb,
            image_emb,
            text_emb,
            sim_text_audio,
            sim_text_image,
            sim_audio_image,
        )

    def criterion(
        self,
        sim_text_audio: torch.Tensor,
        sim_text_image: torch.Tensor,
        sim_audio_image: torch.Tensor,
    ) -> torch.Tensor:

        return self.audio_clip_loss(
            sim_text_audio=sim_text_audio,
            sim_text_image=sim_text_image,
            sim_audio_image=sim_audio_image,
        )


register_method(
    name="audio_clip", 
    model_cls=AudioCLIP, 
    logs=lambda model: (
        "\n"
        "---------------- AudioCLIP Configuration ----------------\n"
        f"Audio Encoder                    : {model.audio_encoder.__class__.__name__}\n"
        f"Image Encoder                    : {model.image_encoder.__class__.__name__}\n"
        f"Text Encoder                     : {model.text_encoder.__class__.__name__}\n"
        f"Contrastive Temperature          : {model.temperature.item():.4f}\n"
        f"Text Template                    : \"{model.text_template}\"\n"
        "Modality Fusion                 : Similarity matrices computed for all present modality pairs\n"
        "Loss                            : Contrastive Triplet Loss (AudioCLIPLoss)\n"
    )
)
