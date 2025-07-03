import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple , Optional
from MK_SSL.multimodal.models.modules.encoders import CNN14
from MK_SSL.multimodal.models.modules.encoders import BERTTextEncoder


class CLAPContrastivePretraining(nn.Module):
    """
    CLAP-style Contrastive Pretraining for learning joint audio-text embeddings.
    Based on: https://arxiv.org/abs/2206.04769 (CLAP: Learning Audio Concepts from Natural Language Supervision)
    """

    def __init__(
        self,
        audio_embedding_dim: int,
        text_embedding_dim: int,
        audio_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        projection_dim: int = 1024,
        temperature_init: float = 0.007,
    ):
        """
        Args:
            audio_encoder (nn.Module): Audio encoder to extract representations from audio input.
                As recommended in the CLAP paper, CNN14 (from PANNs) is a suitable choice.

            text_encoder (nn.Module): Text encoder to extract representations from input text.
                As per the paper, BERT-base (uncased) is used and recommended.

            audio_embedding_dim (int): Output dimension of the raw audio encoder (e.g., 2048 for CNN14).
            text_embedding_dim (int): Output dimension of the raw text encoder (e.g., 768 for BERT-base).
            projection_dim (int): Output dimension of the shared multimodal embedding space.
            temperature_init (float): Initial value for the temperature scaling factor. Default is 0.007 as in the paper.
        """
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        # Projection heads for mapping raw encoder outputs into a shared embedding space.
        self.audio_proj = nn.Linear(audio_embedding_dim, projection_dim)
        self.text_proj = nn.Linear(text_embedding_dim, projection_dim)
        self.temperature = nn.Parameter(torch.tensor(temperature_init))

        if audio_encoder is not None:
            self.audio_encoder = audio_encoder
        else:
            self.audio_encoder = CNN14()
        
        if self.text_encoder is not None: 
            self.text_encoder = text_encoder
        else: 
            self.text_encoder = BERTTextEncoder()


    def forward(
        self,
        audio_input: torch.Tensor,
        text_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for contrastive pretraining.

        Args:
            audio_input (torch.Tensor): Input tensor for the audio encoder (e.g., log-mel spectrograms), shape (B, ...).
            text_input (torch.Tensor): Input tensor for the text encoder (e.g., tokenized text), shape depends on tokenizer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - audio_proj: Projected audio embeddings of shape (B, D)
                - text_proj: Projected text embeddings of shape (B, D)
                - similarity_matrix: Scaled cosine similarity matrix of shape (B, B)
        """
        audio_emb = self.audio_encoder(audio_input)    # (B, D_a)
        text_emb = self.text_encoder(text_input)       # (B, D_t)

        audio_proj = F.normalize(self.audio_proj(audio_emb), dim=-1)  # (B, D)
        text_proj = F.normalize(self.text_proj(text_emb), dim=-1)     # (B, D)

        similarity_matrix = self.temperature * torch.matmul(text_proj, audio_proj.T)  # (B, B)
        return audio_proj, text_proj, similarity_matrix
