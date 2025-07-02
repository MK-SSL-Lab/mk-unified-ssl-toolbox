import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


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
