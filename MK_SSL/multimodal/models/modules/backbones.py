import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torchvision.models import resnext50_32x4d
from transformers import BertModel, BertConfig
import open_clip




from typing import Optional

from MK_SSL.multimodal.models.modules.feature_extractors import ResNetFeatureExtractor
from MK_SSL.multimodal.models.modules.heads import Wav2ClipProjectionHead

class Wav2ClipAudioEncoder(nn.Module):
    """
    Wav2CLIP audio encoder module.

    Converts raw waveform into spectrograms and processes them through a ResNet-based encoder
    and an optional projection head.

    Args:
        backbone (nn.Module, optional): Custom CNN backbone. If None, uses default ResNetAudio.
        projection_dim (int, optional): Output dimension of projection head. If None, no projection is applied.
        input_dim (int): Dimension of backbone output (default: 512 for ResNetAudio).
        freeze_backbone (bool): If True, freezes the backbone during training.
        sample_rate (int): Sampling rate of input waveform.
        n_fft (int): FFT window size for spectrogram.
        hop_length (int): Hop length for spectrogram.
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        projection_dim: Optional[int] = None,
        input_dim: int = 512,
        freeze_backbone: bool = False,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
    ):
        super().__init__()

        self.spectrogram = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=None,  # returns complex tensor
        )
        self.magnitude = lambda x: x.abs()  # get magnitude of spectrogram

        self.backbone = backbone if backbone is not None else ResNetFeatureExtractor.get_default_resnet_audio()
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.projection = None
        if projection_dim is not None:
            self.projection = Wav2ClipProjectionHead(input_dim=input_dim, output_dim=projection_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the audio encoder.

        Args:
            waveform (torch.Tensor): Input tensor of shape (B, T).

        Returns:
            torch.Tensor: Encoded (and optionally projected) audio representation of shape (B, D).
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # (B, 1, T)

        spec = self.spectrogram(waveform)  # (B, 1, F, T)
        mag = self.magnitude(spec)         # drop phase
        features = self.backbone(mag)

        if self.projection is not None:
            features = self.projection(features)

        return features



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()

        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x


class CNN14(nn.Module):
    """
    CNN14 architecture from PANNs (Pretrained Audio Neural Networks), without pretrained weights.

    Input:
        log-mel spectrogram: Tensor of shape (B, 1, F, T), e.g., (B, 1, 64, 1024)

    Output:
        Feature embedding: Tensor of shape (B, 2048)
    """

    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ConvBlock(1, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            ConvBlock(512, 1024),
            ConvBlock(1024, 2048),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        Args:
            x (Tensor): log-mel spectrogram (B, 1, F, T)

        Returns:
            Tensor: (B, 2048) feature embedding
        """
        x = self.conv_blocks(x)  # (B, 2048, F', T')
        x = self.global_pool(x)  # (B, 2048, 1, 1)
        x = x.view(x.size(0), -1)
        return x




class BERTTextEncoder(nn.Module):
    """
    BERTTextEncoder: Extracts text embeddings using BERT-base (uncased).

    This module returns the [CLS] token embedding from the final layer as the text representation.
    """

    def __init__(self):
        super().__init__()

        # Load architecture of bert-base-uncased (no pretrained weights)
        config = BertConfig.from_pretrained("bert-base-uncased")
        self.bert = BertModel(config)

        self.embedding_dim = config.hidden_size  # usually 768

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids (Tensor): Token IDs, shape (B, T)
            attention_mask (Tensor): Attention mask, shape (B, T)

        Returns:
            Tensor: (B, 768) [CLS] token representation
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_representation = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return cls_representation





class TimeFrequencyFrontEnd(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1,1), padding=(1,1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (batch, 1, time, freq)
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)

  
import torch
import torch.nn as nn
import torch.nn.functional as F



class CLIPTextEncoder(nn.Module):
    """
    Default text encoder for AudioCLIP, using a pre-trained CLIP text encoder (RN50).
    This class handles both tokenization and encoding of raw text into embeddings.
    """

    def __init__(self, device: str = "cpu", model_name: str = "RN50", pretrained: str = "openai"):
        super().__init__()
        self.device = device

        # Load CLIP model and tokenizer (RN50 variant)
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(device)
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer(model_name)

    def forward(self, text: list[str]) -> torch.Tensor:
        """
        Encodes a batch of raw text into normalized CLIP embeddings.

        Args:
            text (list[str]): List of text strings.

        Returns:
            torch.Tensor: text embeddings [B, D].
        """
        tokens = self.tokenizer(text).to(self.device)
        with torch.no_grad():
            text_emb = self.model.encode_text(tokens)
        return F.normalize(text_emb, dim=-1)


class CLIPImageEncoder(nn.Module):
    """
    Pre-trained CLIP image encoder (RN50) used in AudioCLIP.
    """

    def __init__(self, device: str = "cpu", model_name: str = "RN50", pretrained: str = "openai"):
        super().__init__()
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.visual = self.model.visual  # CLIP's visual encoder
        self.visual.eval()  # keep frozen by default

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encodes images into normalized embeddings.
        Args:
            images (torch.Tensor): Batch of images [B, C, H, W].
        Returns:
            torch.Tensor: Normalized image embeddings [B, D].
        """
        with torch.no_grad():
            img_emb = self.visual(images.to(self.device))
        return F.normalize(img_emb, dim=-1)




class FBSPFrontEnd(nn.Module):
    """
    Trainable front-end inspired by ESResNeXt's FBSP (Frequency B-Spline) transform.
    Converts raw waveforms [B, 1, L] into 2D time-frequency features [B, 1, F, T].
    """

    def __init__(self, n_filters: int = 64, kernel_size: int = 400, stride: int = 160):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.relu = nn.ReLU(inplace=True)

        # Additional layers to refine frequency representation
        self.conv2 = nn.Conv1d(
            n_filters,
            n_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(n_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, L]
        x = self.relu(self.bn1(self.conv1(x)))  # [B, n_filters, T]
        x = self.relu(self.bn2(self.conv2(x)))  # [B, n_filters, T]
        return x.unsqueeze(1)  # [B, 1, n_filters, T]


class AudioResNeXtStem(nn.Module):
    """
    Audio encoder following ESResNeXt + FBSP design from the AudioCLIP paper.
    Accepts raw audio [B, 1, L] and outputs normalized embeddings of size 1024.
    """

    def __init__(self, embed_dim: int = 1024, n_filters: int = 64):
        super().__init__()
        self.frontend = FBSPFrontEnd(n_filters=n_filters)

        # ESResNeXt backbone
        base = resnext50_32x4d(pretrained=False)
        base.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # Replace attention pooling with global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Raw audio [B, 1, L]
        Returns:
            Tensor: Normalized embedding [B, embed_dim]
        """
        x = self.frontend(x)  # [B, 1, F, T]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x).flatten(1)  # [B, 2048]
        x = self.fc(x)  # [B, embed_dim]
        return F.normalize(x, dim=-1)
