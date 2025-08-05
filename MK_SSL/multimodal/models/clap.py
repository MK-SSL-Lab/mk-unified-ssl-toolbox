import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Tuple , Optional
from transformers import BatchEncoding
import numpy as np 

from MK_SSL.multimodal.models.modules.backbones import CNN14
from MK_SSL.multimodal.models.modules.backbones import BERTTextEncoder
from MK_SSL.multimodal.models.utils import register_method
from MK_SSL.multimodal.models.modules.losses import CLAPLoss

class CLAP(nn.Module):
    """
    CLAP-style Contrastive Pretraining for learning joint audio-text embeddings.
    Based on: https://arxiv.org/abs/2206.04769 (CLAP: Learning Audio Concepts from Natural Language Supervision)


        Note:
        For users who need to prepare audio-text batches (e.g., tokenizing raw text),
        the `audio_text_collate_fn`function defined in `MK_SSL.utils.data_utils`.
        These utilities handle tokenization using HuggingFace's `BertTokenizer` and 
        collate audio-text data into ready-to-use batches.

    """

    def __init__(
        self,
        audio_embedding_dim: Optional[int] = 2048,
        text_embedding_dim: Optional[int] = 768,
        audio_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        projection_dim: int = 1024,
        temperature_init: float = 1,
        device: str = 'cpu',
        **kwargs

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
        self.device = device
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        # Projection heads for mapping raw encoder outputs into a shared embedding space.
        self.audio_proj = nn.Linear(audio_embedding_dim, projection_dim)
        self.text_proj = nn.Linear(text_embedding_dim, projection_dim)
        self.temperature = nn.Parameter(torch.tensor(np.log(1 /temperature_init)))

        if audio_encoder is not None:
            self.audio_encoder = audio_encoder
        else:
            self.audio_encoder = CNN14(use_groupnorm= False)
        
        if self.text_encoder is not None: 
            self.text_encoder = text_encoder
        else: 
            self.text_encoder = BERTTextEncoder()

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_fft=1024,
            hop_length=320,
            n_mels=64,
            f_min=50,
            f_max=8000
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

        self.clap_loss = CLAPLoss()


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

        audio_input = self.mel_spectrogram_transform(audio_input)  # output: (B, 1, F, T)
    
        audio_emb = self.audio_encoder(audio_input)    # (B, D_a)

        
        if isinstance(text_input, BatchEncoding):
            text_input = dict(text_input)
        elif isinstance(text_input, dict):
            pass
        elif isinstance(text_input, torch.Tensor):
            text_input = {"input_ids": text_input, "attention_mask": (text_input != 0).long()}
        else:
            raise TypeError(f"Unsupported text_input type: {type(text_input)}")
        
        text_emb = self.text_encoder(**text_input)
        
        audio_proj = F.normalize(self.audio_proj(audio_emb), dim=-1)  # (B, D)
        text_proj = F.normalize(self.text_proj(text_emb), dim=-1)     # (B, D)

        # similarity_matrix = self.temperature * torch.matmul(text_proj, audio_proj.T)  # (B, B)
        # similarity_matrix = similarity_matrix.clamp(-100.0, 100.0)
        similarity_matrix = torch.exp(self.temperature) * torch.matmul(text_proj, audio_proj.T)
        # similarity_matrix = similarity_matrix.clamp(-100.0, 100.0)

        return audio_proj, text_proj, similarity_matrix

    def criterion(self, similarity_matrix: torch.Tensor,) -> torch.Tensor:
        return self.clap_loss(similarity_matrix)

    def mel_spectrogram_transform(
        self,
        audio_input: torch.Tensor,
        target_len: int = 5 * 44_100,   # 5 s clip
    ) -> torch.Tensor:
        """
        Robust CLAP front-end that is safe under torch.cuda.amp.autocast.
        Returns (B, 1, 64, T) float-16/32 tensor in [-1, 1], with no NaNs/Infs.
        """
        # ------------------------------------------------------------------ #
        # 0. clean & reshape                                                 #
        # ------------------------------------------------------------------ #
        if audio_input.dim() == 3 and audio_input.size(1) == 1:  # (B,1,L) → (B,L)
            audio_input = audio_input.squeeze(1)

        # wipe NaNs/Infs that might be in raw wav (rare but worth it)
        audio_input = torch.nan_to_num(audio_input, nan=0.0, posinf=0.0, neginf=0.0)

        B, L = audio_input.shape

        # ------------------------------------------------------------------ #
        # 1. pad / crop to exactly 5 s                                       #
        # ------------------------------------------------------------------ #
        if L < target_len:
            audio_input = F.pad(audio_input, (0, target_len - L))
        elif L > target_len:
            start = (
                torch.randint(0, L - target_len + 1, (), device=audio_input.device).item()
                if self.training else (L - target_len) // 2
            )
            audio_input = audio_input[:, start : start + target_len]

        # ------------------------------------------------------------------ #
        # 2. Mel-spectrogram and dB conversion **in float32**                #
        #    (disable autocast so we don’t drop to fp16 here)                #
        # ------------------------------------------------------------------ #
        with torch.cuda.amp.autocast(enabled=False):
            wav32 = audio_input.float()                      # (B,L) fp32
            mel = self.mel_transform(wav32)                  # (B,64,T) power

            # Floor at 1e-5 (-50 dB) *in fp32* so it never underflows.
            mel = torch.clamp(mel, min=1e-5)

            mel = self.amplitude_to_db(mel)                 # [-80, 0] dB range fp32

            # ------------------------------------------------------------------ #
            # 3. scale to [-1,1] and cast back to original dtype (amp chosen)     #
            # ------------------------------------------------------------------ #
            mel = (mel + 80.0) / 80.0        # [0,1]
            mel = mel * 2.0 - 1.0            # [-1,1]

        mel = mel.to(audio_input.dtype)      # back to fp16 if autocast asked for it
        mel = torch.nan_to_num(mel)          # final safety net

        return mel.unsqueeze(1)              # (B,1,64,T)



register_method(
    name= "clap",
    model_cls= CLAP,
    logs=lambda model: (
        "\n"
        "---------------- CLAP Configuration ----------------\n"
        f"Audio Encoder                    : {model.audio_encoder.__class__.__name__}\n"
        f"Text Encoder                     : {model.text_encoder.__class__.__name__}\n"
        f"Audio Embedding Dim              : {model.audio_proj.in_features}\n"
        f"Text Embedding Dim               : {model.text_proj.in_features}\n"
        f"Shared Projection Dimension      : {model.audio_proj.out_features}\n"
        f"Contrastive Temperature          : {model.temperature.item():.6f}\n"
        "Modality Pairing                 : Audio ↔ Text contrastive alignment\n"
        "Loss                             : Symmetric InfoNCE (CLAPLoss)\n"
    )
)