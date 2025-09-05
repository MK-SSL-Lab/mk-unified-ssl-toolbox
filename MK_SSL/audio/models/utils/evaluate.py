import torch
import torch.nn as nn
from torch import Tensor


class CTCEvaluateNet(nn.Module):
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
            lengths (Tensor, optional): Raw audio lengths (samples).

        Returns:
            log_probs (Tensor): (B, T_out, num_classes)
            out_lengths (Tensor): (B,)
        """
        feats_out = self.backbone(x, lengths)          # either Tensor or (Tensor, t_lengths)

        if isinstance(feats_out, tuple):
            feats, out_lengths = feats_out             # (B, T_out, E), (B,)
        else:
            feats = feats_out                          # (B, T_out, E)
            out_lengths = torch.full(
                size=(feats.size(0),),
                fill_value=feats.size(1),
                dtype=torch.long,
                device=feats.device,
            )

        logits = self.fc(feats)                        # (B, T_out, C)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return log_probs, out_lengths


