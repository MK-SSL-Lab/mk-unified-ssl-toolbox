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
        use_data_parallel: bool = False,
    ):
        
        """
        Trainer class for generic self-supervised pretraining.

        Args:
            model (nn.Module): The backbone model to train.
            loss_fn (Callable): The self-supervised loss function.
            dataloader (DataLoader): DataLoader that yields batches of dicts.
            optimizer_ctor (Callable): A function that returns an optimizer when called with model parameters.
            build_views_fn (Callable, optional): A custom function for generating training views (e.g., masking,
                augmentation, or distortion). If None, defaults to returning two identical views and no masks.

                The function should have the signature:
                    def build_views_fn(batch: dict) -> Tuple[List[dict], List[Optional[torch.Tensor]]]

                Example:
                    def my_masking_fn(batch):
                        input_ids = batch["input_ids"]
                        attention_mask = batch["attention_mask"]
                        rand = torch.rand_like(input_ids.float())
                        mask = (rand < 0.15)
                        masked_ids = input_ids.clone()
                        masked_ids[mask] = 103  # [MASK]
                        return [{"input_ids": masked_ids, "attention_mask": attention_mask}], [mask]

                Returns:
                    views (List[dict]): A list of input dicts for different views.
                    masks (List[Optional[torch.Tensor]]): Optional mask(s) per view for use in the loss function.

            epochs (int): Number of training epochs.
            device (torch.device, optional): The device to run training on.
        """     

        if use_data_parallel and not torch.cuda.is_available():

            self.logger.error(
                "DataParallel requires at least one CUDA-enabled GPU, but none were found. "
                "Please set `use_data_parallel=False` or ensure CUDA is available."
            )

            raise RuntimeError(
                "DataParallel requires at least one CUDA-enabled GPU, but none were found. "
                "Please set `use_data_parallel=False` or ensure CUDA is available."
            )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = model

        if use_data_parallel:
            # self.logger.info(f"Wrapping model with DataParallel using {torch.cuda.device_count()} GPUs.")
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.optimizer = optimizer_ctor(model.parameters())
        self.epochs = epochs

    def fit(self):
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
        if self.build_views_fn:
            return self.build_views_fn(batch)
        return [batch, batch], [None, None]
    