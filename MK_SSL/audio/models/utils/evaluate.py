import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


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



class ClassificationEvalNet(nn.Module):
    """
    Simple classifier on top of a backbone.

    Plays nicely with `EATBackbone`:
      - If the backbone exposes a CLS token (has `cls_token` attr), we request `return_cls=True`
        and use the CLS embedding for classification.
      - Otherwise, it accepts either:
          * clip-level features: (B, E)
          * token sequences:     (B, T, E) or (B, P, E)  -> pooled by mean

    Freeze behavior:
      - `is_linear=True` freezes the backbone params and keeps it in eval() to avoid BN/Dropout drift.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_size: int,
        num_classes: int,
        is_linear: bool,
        *,
        pooling: str = "mean",        # used when no CLS is available
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        norm: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.is_linear = is_linear
        self.pooling = pooling

        # freeze or finetune
        for p in self.backbone.parameters():
            p.requires_grad = not is_linear
        if is_linear:
            self.backbone.eval()

        proj_in = feature_size
        head = []
        if norm:
            head.append(nn.LayerNorm(proj_in))
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        if hidden_dim and hidden_dim > 0:
            head += [nn.Linear(proj_in, hidden_dim), nn.ReLU(inplace=True)]
            if dropout > 0:
                head.append(nn.Dropout(dropout))
            proj_in = hidden_dim
        head.append(nn.Linear(proj_in, num_classes))
        self.head = nn.Sequential(*head)

    @torch.no_grad()
    def _keep_backbone_eval_if_linear(self):
        if self.is_linear:
            self.backbone.eval()

    def _pool_no_cls(self, feats: Tensor) -> Tensor:
        # feats: (B, E) or (B, T, E)/(B, P, E)
        if feats.dim() == 2:
            return feats
        if feats.dim() == 3:
            if self.pooling == "max":
                return feats.amax(dim=1)
            # default mean
            return feats.mean(dim=1)
        raise RuntimeError(f"Unexpected feat shape {tuple(feats.shape)}")

    def forward(self, x: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
        """
        Returns logits (B, C).
        - If the backbone has `cls_token`, we call it with `return_cls=True` and use the CLS vector.
        - Otherwise we take the output and pool if needed.
        """
        self._keep_backbone_eval_if_linear()

        use_cls = hasattr(self.backbone, "cls_token")
        if use_cls:
            # EATBackbone signature: (waveforms, lengths=None, return_cls=False)
            out = self.backbone(x, lengths=lengths, return_cls=True)
            if not isinstance(out, tuple) or len(out) < 3:
                # Fallback in case a different backbone ignores return_cls
                feats = out[0] if isinstance(out, tuple) else out
                pooled = self._pool_no_cls(feats)
            else:
                _, _, cls_emb = out  # (B, E)
                pooled = cls_emb
        else:
            out = self.backbone(x, lengths) if lengths is not None else self.backbone(x)
            feats = out[0] if isinstance(out, tuple) else out
            pooled = self._pool_no_cls(feats)

        logits = self.head(pooled)  # (B, C)
        return logits
