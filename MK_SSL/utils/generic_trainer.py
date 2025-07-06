# trainer.py
import torch
import inspect
from torch import nn
from typing import Callable

class GenericSSLTrainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        dataloader,
        optimizer_ctor: Callable,
        epochs: int = 100,
        device: torch.device = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.optimizer = optimizer_ctor(model.parameters())
        self.epochs = epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self):
        self.model.to(self.device)
        self.model.train()
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            for batch in self.dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                views, masks = self._build_views(batch)
                outputs = [self._run_model(view, masks[i]) for i, view in enumerate(views)]
                loss = self._call_loss(outputs=outputs, masks=masks)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                print(f"Loss: {loss.item():.4f}")

    def _run_model(self, batch, extra_kwargs=None):
        sig = inspect.signature(self.model.forward)
        accepted = {k: v for k, v in batch.items() if k in sig.parameters}
        if extra_kwargs:
            accepted.update({k: v for k, v in extra_kwargs.items() if k in sig.parameters})
        return self.model(**accepted)

    def _call_loss(self, **kwargs):
        sig = inspect.signature(self.loss_fn)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return self.loss_fn(**filtered)

    def _build_views(self, batch):
        return [batch, batch], [None, None]
