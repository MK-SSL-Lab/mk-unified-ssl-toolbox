import torch.nn as nn

import torchaudio.transforms as T

from typing import List, Tuple, Optional


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-based audio encoder.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride for the first convolutional layer.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        """Forward pass for the residual block."""
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-based audio encoder for spectrogram inputs.

    Args:
        block (nn.Module): Residual block type.
        layers (Tuple[int]): Number of blocks in each ResNet stage.
        in_channels (int): Number of input channels. Default is 1 for spectrograms.
    """
    def __init__(self, block=BasicBlock, layers=(2, 2, 2, 2), in_channels=1):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        Constructs a ResNet stage with multiple residual blocks.

        Args:
            block (nn.Module): Block type.
            out_channels (int): Output channels for the stage.
            blocks (int): Number of blocks in the stage.
            stride (int): Stride for the first block.

        Returns:
            nn.Sequential: A sequential container of blocks.
        """
        layers = []
        layers.append(block(self.inplanes, out_channels, stride))
        self.inplanes = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the ResNet encoder."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)
    
    @staticmethod
    def get_default_resnet_audio():
        """
        Constructs the default ResNet-based audio encoder.

        Returns:
            ResNetAudio: An instance of the default ResNetAudio.
        """
        return ResNetFeatureExtractor(block=BasicBlock, layers=(2, 2, 2, 2))
