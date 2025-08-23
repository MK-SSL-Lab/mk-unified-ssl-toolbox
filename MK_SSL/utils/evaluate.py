import torch
import torch.nn as nn
from torch import Tensor

class EvaluateNet(nn.Module):
    """
    EvaluateNet: CTC-based evaluation on top of a pre-trained backbone.
    """

    def __init__(self, backbone: nn.Module, feature_size: int, num_classes: int, is_linear: bool):
        super().__init__()
        self.backbone = backbone

        for param in self.backbone.parameters():
            param.requires_grad = not is_linear
        if is_linear:
            self.backbone.eval()

        self.fc = nn.Linear(feature_size, num_classes, bias=True)

    def forward(self, x: Tensor, lengths: Tensor = None):
        """
        Args:
            x:       (B, 1, T_pad) padded waveforms
            lengths: (B,) valid lengths in **samples**

        Returns:
            log_probs:   (B, P_max, num_classes)
            out_lengths: (B,) valid frame counts after patching (for CTC)
        """
        feats, out_lengths = self.backbone(x, lengths)  # feats: (B,P_max,E), out_lengths: (B,)
        logits = self.fc(feats)                         # (B,P_max,C)
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        # Safety: make sure lengths don't exceed time dim
        T_max = logits.size(1)
        out_lengths = out_lengths.clamp(min=0, max=T_max)

        return log_probs, out_lengths
