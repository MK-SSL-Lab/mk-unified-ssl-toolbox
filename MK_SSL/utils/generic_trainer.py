# trainer.py
import torch
import inspect
from torch import nn
from dataclasses import dataclass
from typing import Callable, Optional, Dict, List, Tuple, Any


# =========================
# Utilities
# =========================

@dataclass
class LabelAliases:
    """
    Map 'target argument name' -> list of possible keys in the batch.
    The first existing key will be copied under the target name.

    Example for HF BERT pretraining:
        LabelAliases({
            "labels": ["mlm_labels"],
            "next_sentence_label": ["nsp_labels"],
        })
    """
    mapping: Dict[str, List[str]]

    def apply(self, batch: Dict[str, Any], accepted_names: List[str]) -> Dict[str, Any]:
        if not self.mapping:
            return batch
        out = dict(batch)
        for target, sources in self.mapping.items():
            if target in accepted_names and target not in out:
                for s in sources:
                    if s in out:
                        out[target] = out[s]
                        break
        return out


def _filter_kwargs_by_signature(fn: Callable, base: dict, extra: Optional[dict] = None) -> dict:
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    merged = dict(base)
    if extra:
        merged.update(extra)
    return {k: v for k, v in merged.items() if k in allowed}


def _safe_model_forward(model: nn.Module, batch: dict, force_return_dict: bool = True):
    """Forward with arg filtering and optional return_dict=True (if supported)."""
    sig = inspect.signature(model.forward)
    extra = {}
    if force_return_dict and "return_dict" in sig.parameters:
        extra["return_dict"] = True
    kwargs = _filter_kwargs_by_signature(model.forward, batch, extra)
    return model(**kwargs)


def _clone_like(x: Any) -> Any:
    """Clone tensors; keep non-tensors as is. Handles (list, tuple, dict) containers."""
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, (list, tuple)):
        seq_type = type(x)
        return seq_type(_clone_like(t) for t in x)
    if isinstance(x, dict):
        return {k: _clone_like(v) for k, v in x.items()}
    return x


# =========================
# Generic loss adapter
# =========================

def default_loss_adapter(
    model: nn.Module,
    batch: dict,
    loss_fn: Optional[Callable] = None,
    label_aliases: Optional[LabelAliases] = None,
):
    """
    Returns (loss, outputs).

    1) If the model output has .loss (e.g., many HF models when labels are passed), use it.
    2) Otherwise call loss_fn(**args_it_declares). If loss_fn is None -> error.
    """
    sig = inspect.signature(model.forward)
    accepted_names = list(sig.parameters.keys())

    # Allow the model to compute its own loss by mapping dataset label keys up front.
    batch_for_model = label_aliases.apply(batch, accepted_names) if label_aliases else batch

    outputs = _safe_model_forward(model, batch_for_model, force_return_dict=True)

    model_loss = getattr(outputs, "loss", None)
    if model_loss is not None:
        return model_loss, outputs

    if loss_fn is None:
        raise RuntimeError(
            "No loss available: model did not return `loss` and no `loss_fn` was provided."
        )

    loss_sig = inspect.signature(loss_fn)
    kwargs = {}
    if "outputs" in loss_sig.parameters:
        kwargs["outputs"] = outputs
    for k in batch.keys():
        if k in loss_sig.parameters:
            kwargs[k] = batch[k]

    loss = loss_fn(**kwargs)
    return loss, outputs


# =========================
# Generic SSL Trainer
# =========================

class GenericSSLTrainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[Callable],
        dataloader,
        optimizer_ctor: Callable,
        epochs: int = 100,
        use_data_parallel: bool = False,

        # Generic knobs:
        label_aliases: Optional[LabelAliases] = None,
        build_views_fn: Optional[Callable] = None,   # def f(batch)->(views:List[dict], extras:List[Optional[dict]])
        view_loss_reduction: str = "mean",           # "mean" | "sum"
        num_views: int = 2,                          # used by the default build_views
        clone_views: bool = True,                    # clone tensors to avoid in-place cross-talk
        loss_adapter: Callable = default_loss_adapter
    ):
        """
        Trainer class for generic (self-)supervised training.

        Args:
            model: Any nn.Module.
            loss_fn: Optional loss if the model doesn't compute .loss itself.
            dataloader: Yields dict-batches with tensors inside.
            optimizer_ctor: Callable(model.parameters()) -> optimizer.
            epochs: Training epochs.
            use_data_parallel: Wrap model with nn.DataParallel if True.

            label_aliases: Dataset→model/loss arg-name mapping.
            build_views_fn: Produce multiple training views/augmentations.
                            Signature: (batch) -> (views: List[dict], extras: List[Optional[dict]])
                            Extras are merged into each view before forward (e.g., masks).
            view_loss_reduction: How to reduce per-view losses ("mean" or "sum").
            num_views: Used by default view builder (duplicates the batch).
            clone_views: If True, tensor values are cloned per view to prevent in-place leakage.
            loss_adapter: Function that returns (loss, outputs) from (model, batch, loss_fn, label_aliases).
        """
        if use_data_parallel and not torch.cuda.is_available():
            raise RuntimeError(
                "DataParallel requires at least one CUDA-enabled GPU, but none were found."
            )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = model if not use_data_parallel else nn.DataParallel(model)
        self.model.to(self.device)

        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.epochs = epochs
        self.optimizer = optimizer_ctor(self.model.parameters())

        self.label_aliases = label_aliases or LabelAliases(mapping={})
        self.view_loss_reduction = view_loss_reduction
        self.loss_adapter = loss_adapter

        # default view builder parameters
        self.num_views = max(1, int(num_views))
        self.clone_views = bool(clone_views)
        self.build_views_fn = build_views_fn or self._default_build_views

    # ---------------------
    # Training loop
    # ---------------------
    def fit(self):
        self.model.train()
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            for raw_batch in self.dataloader:
                # Move tensors to device; keep non-tensors as-is (e.g., metadata ids)
                batch = {
                    k: (v.to(self.device) if hasattr(v, "to") else v)
                    for k, v in raw_batch.items()
                }

                views, extras = self.build_views_fn(batch)

                per_view_losses: List[torch.Tensor] = []
                for i, view in enumerate(views):
                    extra = None
                    if extras is not None and i < len(extras):
                        extra = extras[i] if isinstance(extras[i], dict) else None
                    merged_view = {**view, **(extra or {})}

                    loss, _ = self.loss_adapter(
                        self.model,
                        merged_view,
                        self.loss_fn,
                        self.label_aliases,
                    )
                    per_view_losses.append(loss)

                if len(per_view_losses) == 1:
                    total_loss = per_view_losses[0]
                else:
                    stacked = torch.stack([l for l in per_view_losses])
                    if self.view_loss_reduction == "mean":
                        total_loss = stacked.mean()
                    elif self.view_loss_reduction == "sum":
                        total_loss = stacked.sum()
                    else:
                        raise ValueError("view_loss_reduction must be 'mean' or 'sum'.")

                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                print(f"Loss: {total_loss.item():.4f}")

    # ---------------------
    # Default multi-view builder (generic)
    # ---------------------
    def _default_build_views(self, batch: dict) -> Tuple[List[dict], List[Optional[dict]]]:
        """
        Generic default: returns N *independent* views by cloning tensor values.
        This prevents in-place ops on one view from affecting the others.

        Returns:
            views: List[dict] — shallow dicts whose tensor values are cloned (if clone_views=True).
            extras: List[Optional[dict]] — None per view (no extra kwargs by default).
        """
        views: List[dict] = []
        for _ in range(self.num_views):
            if self.clone_views:
                view = {k: _clone_like(v) for k, v in batch.items()}
            else:
                # fast path: reference same tensors (OK if you guarantee no in-place ops)
                view = dict(batch)
            views.append(view)

        extras = [None] * len(views)
        return views, extras
