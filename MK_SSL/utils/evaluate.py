import torch
import torch.nn as nn
from torch import Tensor


class EvaluateNet(nn.Module):
    """
    EvaluateNet: CTC-based evaluation on top of a pre-trained backbone.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_size: int,
        num_classes: int,
        is_linear: bool,
    ):
        """
        Args:
            backbone (nn.Module): Backbone to extract features.
            feature_size (int): Size of feature embeddings.
            num_classes (int): Vocabulary size (including blank).
            is_linear (bool): Whether to freeze backbone parameters.
        """
        super().__init__()
        self.backbone = backbone

        for param in self.backbone.parameters():
            param.requires_grad = not is_linear

        if is_linear:
            self.backbone.eval()

        # Frame-level projection → vocab
        self.fc = nn.Linear(feature_size, num_classes, bias=True)

    def forward(self, x: Tensor, lengths: Tensor = None):
        """
        Args:
            x (Tensor): Input waveforms (B, 1, T).
            lengths (Tensor, optional): Original audio lengths.

        Returns:
            log_probs (Tensor): (B, T, num_classes) log-probs.
            out_lengths (Tensor): Lengths of predicted sequences.
        """
        feats = self.backbone(x, lengths)   # (B, T_out, E)

        logits = self.fc(feats)             # (B, T_out, num_classes)
        log_probs = nn.functional.log_softmax(logits, dim=-1)


        out_lengths = torch.full(
            size=(logits.size(0),), fill_value=logits.size(1), dtype=torch.long
        )

        return log_probs, out_lengths


