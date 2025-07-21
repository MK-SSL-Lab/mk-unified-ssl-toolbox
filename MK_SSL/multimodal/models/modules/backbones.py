import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torchvision.models import resnet50
from transformers import BertModel, BertConfig




from typing import Optional

from MK_SSL.multimodal.models.modules.feature_extractors import ResNetFeatureExtractor
from MK_SSL.multimodal.models.modules.heads import Wav2ClipProjectionHead

class Wav2ClipEncoder(nn.Module):
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

  

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int = 7, embed_dim: int = 1024, num_heads: int = 8):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        cls_token = x.mean(dim=1, keepdim=True)  # (B, 1, C)
        x = torch.cat([cls_token, x], dim=1)  # (B, HW+1, C)
        x = x + self.positional_embedding[: x.size(1), :]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        attn = F.multi_head_attention_forward(
            query=q,
            key=k,
            value=v,
            embed_dim_to_check=q.shape[-1],
            num_heads=self.num_heads,
            in_proj_weight=None,
            in_proj_bias=None,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            training=self.training,
            need_weights=False,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight
        )
        return attn[0][:, 0]  # return [CLS] token output




class TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int = 1024, num_heads: int =16, mlp_dim: int = 4096, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        mlp_out = self.mlp(x)
        return self.ln2(x + mlp_out)


class CLIPTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, max_len=77, num_layers=12, num_heads=8, mlp_dim=2048):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.empty(max_len, embed_dim).normal_(std=embed_dim**-0.5))
        self.transformer = nn.Sequential(
            *[TransformerLayer(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)]
        )
        self.ln_final = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, token_ids):
        x = self.token_embedding(token_ids) + self.pos_embedding[:token_ids.shape[1]]
        x = x.permute(1, 0, 2)  # for multi-head attention (T, B, C)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (B, T, C)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), token_ids.argmax(dim=-1)]  # get <EOT> or max token
        return F.normalize(self.fc(x), dim=-1)



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
    Accepts raw audio [B, 1, L] and outputs normalized embeddings.
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, n_filters: int = 64):
        super().__init__()
        self.frontend = FBSPFrontEnd(n_filters=n_filters)

        base = resnet50(pretrained=False)
        base.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.attnpool = AttentionPool2d(spacial_dim=7, embed_dim=2048, num_heads=num_heads)
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
        x = self.attnpool(x)
        x = self.fc(x)
        return F.normalize(x, dim=-1)