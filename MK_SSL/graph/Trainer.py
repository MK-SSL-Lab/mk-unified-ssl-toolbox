from __future__ import annotations

import os
import re
import torch
import numpy as np
from torch import nn
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, Union
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from tqdm.auto import tqdm
import logging
import optuna

# ---- PyG (required for runtime; stubbed for typing) ----
try:
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except Exception:  # pragma: no cover
    PYG_AVAILABLE = False

    # Stubs so editors/type-checkers have concrete symbols
    class Data:  # type: ignore
        pass

    class Batch:  # type: ignore
        pass

# ---- MK_SSL utilities (mirrors audio Trainer style) ----
from MK_SSL.utils import EmbeddingLogger
from MK_SSL.utils import get_logger_handler
from MK_SSL.utils import WandbLogger
from MK_SSL.utils import optimize_hyperparameters

# ---- Graph registry (aligned with register_method usage) ----
from MK_SSL.graph.models.utils.registry import get_method


class Trainer:
    """
    Trainer for Graph SSL (GraphCL), mirroring the Audio Trainer style.

    Workflow:
      • registry -> model, loss, transform
      • DataLoader with collate_fn that returns (Batch_view1, Batch_view2)
      • AMP, tqdm, W&B, periodic checkpoints, optional validation
      • Optional embedding logging (uses model.backbone directly)
    """

    def __init__(
        self,
        method: str = "graphcl",
        backbone: nn.Module = None,
        save_dir: str = ".",
        checkpoint_interval: int = 10,
        reload_checkpoint: bool = False,
        verbose: bool = True,
        mixed_precision_training: bool = True,
        # W&B
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_mode: str = "online",  # "online" | "offline" | "disabled"
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        wandb_notes: Optional[str] = None,
        wandb_tags: Optional[list[str]] = None,
        use_data_parallel: bool = False,
        **kwargs,
    ) -> None:

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.propagate = False
        if not self.logger.hasHandlers():
            self.logger.addHandler(get_logger_handler())
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self.logger.info("Graph Trainer initialized.")

        if not PYG_AVAILABLE:  # pragma: no cover
            raise ImportError("torch_geometric is required for GraphTrainer (PyG).")

        self.method = method.lower()
        self.backbone_override = backbone
        self.mixed_precision_training = mixed_precision_training
        self.checkpoint_interval = checkpoint_interval
        self.reload_checkpoint = reload_checkpoint

        self.save_dir = os.path.join(save_dir, self.method)
        os.makedirs(self.save_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.save_dir, "Pretext")
        os.makedirs(self.checkpoint_path, exist_ok=True)

        if use_data_parallel and not torch.cuda.is_available():
            msg = ("DataParallel requires at least one CUDA-enabled GPU, but none were found. "
                   "Please set `use_data_parallel=False` or ensure CUDA is available.")
            self.logger.error(msg)
            raise RuntimeError(msg)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_workers = os.cpu_count()

        self.logger.info(
            "\n"
            "---------------- MK_SSL: Graph ----------------\n"
            f"Number of workers : {self.num_workers}\n"
            f"Number of GPUs    : {torch.cuda.device_count()}\n"
            f"Device            : {self.device}\n"
            f"Method            : {self.method}\n"
            "----------------------------------------------------"
        )

        # ---- Load from registry ----
        try:
            method_cfg = get_method(self.method)
        except ValueError as e:
            self.logger.error(f"Method {self.method} not found in graph registry.")
            raise e

        # ---- Model args ----
        model_args = {}
        if "default_params" in method_cfg:
            model_args.update(method_cfg["default_params"])
        model_args.update(kwargs)
        if self.backbone_override is not None:
            model_args["backbone"] = self.backbone_override

        # ---- Loss args ----
        loss_args = {}
        if "default_params" in method_cfg:
            loss_args.update(method_cfg["default_params"])
        loss_args.update(kwargs)

        # ---- Create model/loss/transform ----
        self.model = method_cfg["model"](**model_args)
        self.loss = method_cfg["loss"](**loss_args)
        self.transformation = method_cfg["transformation"]() if method_cfg["transformation"] else None

        self.logger.info(method_cfg["logs"](self.model, self.loss))

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision_training)

        if use_data_parallel:
            self.logger.info(f"Wrapping model with DataParallel using {torch.cuda.device_count()} GPUs.")
            self.model = nn.DataParallel(self.model)

        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)

        self.logger.info(
            "\n"
            "---------------- Model Summary ----------------\n"
            f"Model parameters : {np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}\n"
            "----------------------------------------------"
        )

        # ---- W&B ----
        trainer_internal_config = {
            "method": self.method,
            "save_dir": save_dir,
            "checkpoint_interval": checkpoint_interval,
            "reload_checkpoint": reload_checkpoint,
            "mixed_precision_training": mixed_precision_training,
            "device": str(self.device),
            "num_workers": self.num_workers,
            "num_gpus": torch.cuda.device_count(),
            **kwargs
        }
        full_wandb_config = {**trainer_internal_config, **(wandb_config or {})}

        self.wandb_logger = WandbLogger(
            project_name=wandb_project if wandb_project else f"MK_SSL_{self.method}",
            entity=wandb_entity,
            mode=wandb_mode,
            run_name=wandb_run_name,
            config=full_wandb_config,
            notes=wandb_notes if wandb_notes else f"Training {self.method} with MK_SSL.",
            tags=wandb_tags if wandb_tags else [self.method, "training"],
        )

        self.logger.info(
            "\n"
            "-------------------- W&B ---------------------\n"
            f"W&B Active        : {self.wandb_logger.is_active}\n"
            f"W&B Project       : {self.wandb_logger.project_name}\n"
            f"W&B Entity        : {self.wandb_logger.entity}\n"
            f"W&B Mode          : {self.wandb_logger.mode}\n"
            f"W&B Run Name      : {self.wandb_logger.run_name or 'Auto-generated'}\n"
            "----------------------------------------------------"
        )

    # ======================== GraphCL core ========================

    def _train_graphcl(
        self,
        train_loader: DataLoader,
        optimizer,
        epochs: int,
        start_epoch: int = 0,
        val_loader: Optional[DataLoader] = None,
        logger_loader: Optional[DataLoader] = None,
        use_embedding_logger: bool = False,
    ) -> None:
        """Train GraphCL with optional embedding logging (uses model.backbone)."""

        if self.transformation is None:
            self.logger.error("Transformation not given!")
            raise ValueError("Transformation not given!")

        self.model.train()

        # ---- Step-0 embedding logging (graph-level h via backbone) ----
        if use_embedding_logger:
            assert logger_loader is not None, "logger_loader must be provided when use_embedding_logger=True"
            emb_dir = os.path.join(self.checkpoint_path, "embedding_logs")
            embedding_logger = EmbeddingLogger(
                log_dir=emb_dir,
                method_name=self.method,
                reduce_method="tsne",
                log_interval=1,
            )
            self.logger.info(f"Embedding logger initialized at {emb_dir}")

            self.logger.info("[GraphCL - Step 0] Logging pre-training embeddings...")
            all_embeddings, all_labels = [], []
            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(logger_loader, desc="EmbeddingLogger Step 0"):
                    # Expect logger_loader to yield a Batch (no augmentation) and labels (optional)
                    data = batch["data"].to(self.device) if isinstance(batch, dict) else batch.to(self.device)
                    labels = batch["label"].to(self.device) if isinstance(batch, dict) and "label" in batch else None

                    h = self._encode_backbone(data)  # (B, D) graph-level features
                    all_embeddings.append(h)
                    if labels is not None:
                        all_labels.append(labels)

            embeddings = torch.cat(all_embeddings, dim=0)
            labels = torch.cat(all_labels, dim=0) if all_labels else torch.zeros(embeddings.size(0), dtype=torch.long)
            embedding_logger.log_step(step=0, embeddings=embeddings, labels=labels)
            self.logger.info("[GraphCL - Step 0] Pre-training embeddings logged.")
            self.model.train()

        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(self.model)

        # ---- Training loop ----
        for epoch in range(start_epoch, epochs):
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"GraphCL Epoch {epoch+1}/{epochs}")

            for batch_idx, (batch_v0, batch_v1) in enumerate(pbar):
                batch_v0 = batch_v0.to(self.device)
                batch_v1 = batch_v1.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    z0, z1 = self.model(batch_v0, batch_v1)   # (B, P), (B, P)
                    loss = self.loss(z0, z1)

                optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                running_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

                global_step = epoch * len(train_loader) + batch_idx
                if self.wandb_logger.is_active:
                    self.wandb_logger.log({"train/batch_loss": loss.item()}, step=global_step)

            avg_loss = running_loss / max(1, len(train_loader))
            self.logger.info(f"[GraphCL - Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

            epoch_step = (epoch + 1) * len(train_loader)
            if self.wandb_logger.is_active:
                self.wandb_logger.log({"train/epoch_loss": avg_loss, "epoch": epoch + 1}, step=epoch_step)

            # ---- Embedding logger per epoch (optional) ----
            if use_embedding_logger:
                self.logger.info(f"[GraphCL - Epoch {epoch+1}] Logging embeddings...")
                self.model.eval()
                all_embeddings, all_labels = [], []
                with torch.no_grad():
                    for batch in tqdm(logger_loader, desc=f"EmbeddingLogger Epoch {epoch+1}"):
                        data = batch["data"].to(self.device) if isinstance(batch, dict) else batch.to(self.device)
                        labels = batch["label"].to(self.device) if isinstance(batch, dict) and "label" in batch else None
                        h = self._encode_backbone(data)
                        all_embeddings.append(h)
                        if labels is not None:
                            all_labels.append(labels)
                embeddings = torch.cat(all_embeddings, dim=0)
                labels = torch.cat(all_labels, dim=0) if all_labels else torch.zeros(embeddings.size(0), dtype=torch.long)
                embedding_logger.log_step(step=epoch + 1, embeddings=embeddings, labels=labels)
                self.model.train()

            # ---- Validation ----
            if val_loader:
                avg_val_loss = self._validate_graphcl(val_loader, epoch, epoch_step)

            # ---- Optuna pruning ----
            if hasattr(self, "_optuna_trial"):
                metric = avg_val_loss if val_loader else avg_loss
                self._optuna_trial.report(metric, epoch)
                if self._optuna_trial.should_prune():
                    raise optuna.TrialPruned()

            # ---- Periodic checkpoints ----
            if (epoch + 1) % self.checkpoint_interval == 0 and not hasattr(self, "_optuna_trial"):
                ckpt = os.path.join(self.checkpoint_path, f"{self.method}_model_{self.timestamp}_epoch{epoch+1}.pth")
                torch.save(self.model.state_dict(), ckpt)
                self.logger.info(f"Model checkpoint saved: {ckpt}")
                if self.wandb_logger.is_active:
                    self.wandb_logger.save_artifact(
                        ckpt,
                        name=f"{self.method}-model-epoch-{epoch+1}",
                        type="model",
                        metadata={"epoch": epoch + 1, "loss": avg_loss},
                    )

        # ---- Final checkpoint ----
        final_path = os.path.join(self.checkpoint_path, f"{self.method}_model_{self.timestamp}_final.pth")
        torch.save(self.model.state_dict(), final_path)
        self.logger.info(f"Final model checkpoint saved: {final_path}")
        if self.wandb_logger.is_active:
            self.wandb_logger.save_artifact(
                final_path,
                name=f"{self.method}-model-final",
                type="model",
                metadata={"epochs_trained": epochs, "final_loss": avg_loss},
            )

        if use_embedding_logger:
            self.logger.info("Generating final embedding animation...")
            anim_path = embedding_logger.plot_all()
            self.logger.info(f"Embedding animation saved at: {anim_path}")
            if self.wandb_logger.is_active:
                import wandb
                self.wandb_logger.log(
                    {"media/embedding_animation": wandb.Html(anim_path)},
                    step=max(embedding_logger.steps) if embedding_logger.steps else epochs,
                )

        self.logger.info("GraphCL training complete.")

    def _validate_graphcl(self, val_loader: DataLoader, epoch: int, epoch_step: int) -> float:
        """Validation loop for GraphCL."""
        self.model.eval()
        val_running = 0.0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Validation GraphCL Epoch {epoch+1}")
            for (batch_v0, batch_v1) in pbar:
                batch_v0 = batch_v0.to(self.device)
                batch_v1 = batch_v1.to(self.device)
                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    z0, z1 = self.model(batch_v0, batch_v1)
                    loss = self.loss(z0, z1)
                val_running += loss.item()
        avg_val = val_running / max(1, len(val_loader))
        self.logger.info(f"[GraphCL - Epoch {epoch+1}] Val Loss: {avg_val:.4f}")
        if self.wandb_logger.is_active:
            self.wandb_logger.log({"val/loss": avg_val}, step=epoch_step)
        self.model.train()
        return avg_val

    # ======================== Public API ========================

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 128,
        start_epoch: int = 0,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer: str = "adamw",
        use_hpo: bool = False,
        n_trials: int = 20,
        tuning_epochs: int = 5,
        use_embedding_logger: bool = False,
        logger_dataset: Optional[Dataset] = None,  # dataset of plain graphs for embedding viz
        **kwargs,
    ) -> None:
        """
        Main training loop for GraphCL.

        Dataset expectations (per sample):
            return a PyG `Data` object with at least: `x: (N_i, in_dim)`, `edge_index: (2, E_i)`.
            (graph-level labels optional; only used for embedding plots if provided via logger_dataset)
        """
        if not hasattr(self, "_optuna_trial"):
            self.wandb_logger.init_run()
        else:
            self.wandb_logger.mode = "disabled"

        if self.wandb_logger.is_active:
            self.wandb_logger.current_run.config.update({
                "batch_size": batch_size,
                "start_epoch": start_epoch,
                "epochs": epochs,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "optimizer": optimizer,
                **kwargs
            })
            self.logger.info(f"W&B run initialized. View run at: {self.wandb_logger.current_run.url}")
        else:
            self.logger.info("W&B logging is not active for this run.")

        # ---- HPO (optional) ----
        if use_hpo:
            self.logger.info("🧪 Running Optuna for hyperparameter tuning (GraphCL)…")
            best_params = optimize_hyperparameters(
                trainer=self,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                n_trials=n_trials,
                epochs=tuning_epochs,
            )
            self.logger.info(f"🌟 Best hyperparameters found: {best_params}")
            lr = best_params.get("lr", lr)
            batch_size = best_params.get("batch_size", batch_size)
            weight_decay = best_params.get("weight_decay", weight_decay)
            optimizer = best_params.get("optimizer", optimizer)
            kwargs.update({k: v for k, v in best_params.items() if k not in {"lr", "batch_size", "weight_decay", "optimizer"}})
            if self.wandb_logger.is_active:
                self.wandb_logger.log({
                    "hpo/best_lr": lr,
                    "hpo/best_batch_size": batch_size,
                    "hpo/best_weight_decay": weight_decay,
                    "hpo/best_optimizer": optimizer,
                    **{f"hpo/{k}": v for k, v in kwargs.items()}
                })

        # ---- optimizer ----
        match optimizer.lower():
            case "adam":
                optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            case "sgd":
                optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            case "adamw":
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            case _:
                self.logger.error(f"Unsupported Optimizer: {optimizer}")
                raise ValueError(f"Optimizer {optimizer} not supported")

        if self.reload_checkpoint:
            start_epoch = self._reload_latest_checkpoint()

        # ---- DataLoaders ----
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._graphcl_collate,  # returns (Batch_view1, Batch_view2)
        )
        self.logger.info(f"Training dataset loaded with {len(train_dataset)} graphs.")

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self._graphcl_collate,
            )

        logger_loader = None
        if use_embedding_logger and logger_dataset is not None:
            # NOTE: logger_dataset should yield a plain `Data`; no transform applied here
            logger_loader = DataLoader(
                logger_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self._data_loader_safe_collate,
            )

        # ---- train ----
        self._train_graphcl(
            train_loader=train_loader,
            optimizer=optimizer,
            epochs=epochs,
            start_epoch=start_epoch,
            val_loader=val_loader,
            logger_loader=logger_loader,
            use_embedding_logger=use_embedding_logger,
        )

        training_mode = "Main" if not hasattr(self, "_optuna_trial") else "HPO"
        if self.wandb_logger.is_active:
            self.wandb_logger.finish_run()
            self.logger.info(f"{training_mode} training process completed and W&B run finalized.")
        else:
            self.logger.info(f"{training_mode} training process completed.")

    # ======================== Helpers ========================

    def _graphcl_collate(self, data_list: list) -> Tuple[Union[Data, Batch], Union[Data, Batch]]:
        """
        Collate a list of PyG `Data` -> (Batch_view1, Batch_view2) using the configured transform.
        """
        data_batch = Batch.from_data_list(data_list)
        v0, v1 = self.transformation(data_batch)  # both are Batch
        return v0, v1

    def _data_loader_safe_collate(self, batch):
        batch = [item for item in batch if item is not None]
        return default_collate(batch) if batch else None

    def _encode_backbone(self, data_or_batch: Union[Data, Batch]) -> torch.Tensor:
        """
        Graph-level embeddings h using the model's backbone (pre-projection).
        Accepts a PyG Batch with attributes (x, edge_index, batch).
        """
        if hasattr(data_or_batch, "x") and hasattr(data_or_batch, "edge_index") and hasattr(data_or_batch, "batch"):
            h = self.model.backbone(data_or_batch.x, data_or_batch.edge_index, data_or_batch.batch)
        else:
            # last resort; rely on model input path
            h = self.model(data_or_batch)
            if isinstance(h, tuple):  # (z0, z1) path is not expected here
                h = h[0]
        if h.dim() > 2:
            h = h.view(h.size(0), -1)
        return h

    def load_checkpoint(self, checkpoint_path: str) -> None:
        if self.model is None:
            self.logger.error("Model must be initialized before loading a checkpoint.")
            raise RuntimeError("Model must be initialized before loading a checkpoint.")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.logger.info(f"Checkpoint loaded from: {checkpoint_path}")

    def _reload_latest_checkpoint(self) -> int:
        checkpoints = os.listdir(self.checkpoint_path)
        method_prefix = self.method + "_model_"
        filtered = [ckpt for ckpt in checkpoints if ckpt.endswith(".pth") and ckpt.startswith(method_prefix)]
        if not filtered:
            self.logger.warning(
                f"⚠️  No valid checkpoints found for method '{self.method}' in {self.checkpoint_path}. Starting from scratch."
            )
            return 0
        sorted_ckpts = sorted([os.path.join(self.checkpoint_path, ckpt) for ckpt in filtered], key=os.path.getmtime)
        latest_ckpt = sorted_ckpts[-1]
        self.load_checkpoint(latest_ckpt)
        match = re.search(r"epoch(\d+)", latest_ckpt)
        if match:
            epoch = int(match.group(1))
            self.logger.info(f"Reloaded checkpoint from epoch {epoch + 1}")
        else:
            self.logger.warning(
                f"⚠️  No epoch number found in the checkpoint name '{latest_ckpt}'. Resuming from epoch 1."
            )
            epoch = 0
        return epoch

    def __del__(self):
        if hasattr(self, "writer"):
            self.writer.close()
