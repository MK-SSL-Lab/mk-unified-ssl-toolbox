import os
import re
import torch
import numpy as np
import os
from torch import nn
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Subset, DataLoader  # Added DataLoader for clarity
import logging
from torcheval.metrics.functional import multiclass_accuracy
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
import random


import wandb
import optuna

from typing import Optional, Dict, Any

from MK_SSL.vision.models import *
from MK_SSL.vision.models.modules.losses import *
from MK_SSL.vision.models.modules.transformations import *
from MK_SSL.utils import get_logger_handler
from MK_SSL.vision.models.utils import get_method
from MK_SSL.utils import optimize_hyperparameters
from MK_SSL.utils import WandbLogger
from MK_SSL.vision.models.modules import MAEBackbone
from MK_SSL.vision.models.utils import EvaluateNet
from MK_SSL.utils import EmbeddingLogger


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate truncated images

try:
    import cv2
    cv2.setNumThreads(0)  # avoid OpenCV worker thread issues
except Exception:
    pass

class Trainer:
    def __init__(
        self,
        method: str,
        backbone: nn.Module,
        image_size: int,
        save_dir: str = ".",
        feature_size: Optional[int] = None,
        checkpoint_interval: int = 10,
        reload_checkpoint: bool = False,
        verbose: bool = True,
        mixed_precision_training: bool = True,
        # W&B specific arguments
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_mode: str = "online",  # "online", "offline", "disabled"
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        wandb_notes: Optional[str] = None,
        wandb_tags: Optional[list[str]] = None,
        use_data_parallel: bool = False,
        mae_normalize_target: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """
        Description:
            Trainer class for training a model using self-supervised learning methods. This class manages the
            training loop, model saving, and supports advanced features such as mixed precision training and
            checkpointing.

        Args:
            method (str): The self-supervised learning method to be used for training.
                          Available options include:
                          - 'BarlowTwins'
                          - 'BYOL'
                          - 'DINO'
                          - 'MoCov2'
                          - 'MoCov3'
                          - 'SimCLR'
                          - 'SimSiam'
                          - 'SwAV'
            backbone (nn.Module): The neural network module serving as the backbone of the model.
            feature_size (int): The dimensionality of the feature vector output by the backbone model.
            image_size (int): The dimensions (height, width) of the input images. This is generally expected to
                              be a square (i.e., height equals width).
            save_dir (str): Path to the directory where model checkpoints and logs will be saved. Defaults to
                            the current directory ("./").
            checkpoint_interval (int): Frequency (in epochs) at which model checkpoints are saved. For example,
                                        if set to 10, the model will be saved every 10 epochs.
            reload_checkpoint (bool): If set to True, training will resume from the latest checkpoint available
                                      in the `save_dir`. If False, training will start from scratch.
            verbose (bool): If True, detailed logs and progress updates will be printed during training.
            mixed_precision_training (bool): If True, mixed precision (using both 16-bit and 32-bit floats)
                                             will be used during training to improve performance and reduce memory usage.
            wandb_project (str, optional): W&B project name. If None, uses default from W&B.
            wandb_entity (str, optional): W&B entity (username or team name). If None, uses default.
            wandb_mode (str, optional): W&B logging mode ("online", "offline", "disabled"). Defaults to "online".
            wandb_run_name (str, optional): Custom name for the W&B run.
            wandb_config (Dict[str, Any], optional): Dictionary of hyperparameters/settings for W&B.
            wandb_notes (str, optional): Notes for the W&B run.
            wandb_tags (list[str], optional): Tags for the W&B run.
            mae_variant (str): One of 'vit-b', 'vit-l', 'vit-h' specifying model size.

            **kwargs: Additional keyword arguments for extending functionality or overriding default settings
                      specific to the training method or the backbone architecture.
        """

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.propagate = False

        if not self.logger.hasHandlers():
            self.logger.addHandler(get_logger_handler())

        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

        self.logger.info("Vision Trainer initialized.")

        self.method = method.lower()
        self.image_size = image_size
        self.backbone = backbone
        self.feature_size = feature_size
        self.reload_checkpoint = reload_checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.mixed_precision_training = mixed_precision_training

        self.save_dir = os.path.join(save_dir, self.method)

        if self.method != "mae" and self.feature_size is None:
            self.logger.error(
                f"`feature_size` must be explicitly provided for the selected method '{self.method}'. "
                "The feature size should match the output dimension of the chosen backbone encoder."
            )

            raise ValueError(
                f"`feature_size` must be explicitly provided for the selected method '{self.method}'. "
                "The feature size should match the output dimension of the chosen backbone encoder."
            )

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.checkpoint_path = os.path.join(self.save_dir, "Pretext")

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

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
        self.num_workers = os.cpu_count()

        self.logger.info(
            "\n"
            "---------------- MK_SSL: Vision ----------------\n"
            f"Number of workers : {self.num_workers}\n"
            f"Number of GPUs    : {torch.cuda.device_count()}\n"
            f"Device            : {self.device}\n"
            f"Method            : {self.method}\n"
        )

        try:
            method_cfg = get_method(self.method)
        except ValueError as e:
            self.logger.error(f"Method {self.method} not found in registry.")
            raise e

        model_special_overrides = {
            "barlowtwins": (
                {"hidden_dim": self.feature_size} if self.method == "simsiam" else {}
            ),
            "simsiam": (
                {
                    "projection_hidden_dim": self.feature_size,
                    "prediction_hidden_dim": self.feature_size // 4,
                }
                if self.method == "simsiam"
                else {}
            ),
            "mae": (
                {"image_size": self.image_size,
                 "device": self.device}
                if self.method == "mae"
                else {}
            ),
        }

        loss_special_overrides = {
            "dino": (
                {
                    "projection_dim": self.model.projection_dim,
                    "temp_student": self.model.temp_student,
                    "temp_teacher": self.model.temp_teacher,
                }
                if self.method == "dino"
                else {}
            ),
            "swav": (
                {
                    "num_crops": self.model.num_crops + 2,
                }
                if self.method == "swav"
                else {}
            ),
            "mae": {"normalize_target": mae_normalize_target} if self.method == "mae" else {},
        }

        self.model = method_cfg["model"](
            backbone=self.backbone,
            feature_size=self.feature_size,
            **model_special_overrides.get(self.method, {}),
            **kwargs,
        )

        self.loss = method_cfg["loss"](
            **loss_special_overrides.get(self.method, {}), **kwargs
        )

        self.transformation = method_cfg["transformation"](
            image_size=self.image_size, **kwargs
        ) if method_cfg["transformation"] else None

        # Only define transformation_prime if needed
        if self.method in {"byol", "barlowtwins", "simclr", "simsiam", "mocov3"}:
            self.transformation_prime = self.transformation

        if self.method in {"dino"}:
            self.transformation_global1 = self.transformation
            self.transformation_global2 = self.transformation
            self.transformation_local = self.transformation

        if self.method in {"swav"}:
            self.transformation_global = self.transformation
            self.transformation_local = self.transformation

        if use_data_parallel:
            self.logger.info(
                f"Wrapping model with DataParallel using {torch.cuda.device_count()} GPUs."
            )
            self.model = nn.DataParallel(self.model)

        self.logger.info(method_cfg["logs"](self.model, self.loss))

        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision_training)

        self.logger.info(
            "\n"
            "---------------- Model Summary ----------------\n"
            f"Model parameters : {np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}\n"
            "----------------------------------------------"
        )

        # --- W&B Logger Initialization ---
        # Combine trainer_config with any specific wandb_config provided
        trainer_internal_config = {
            "method": self.method,
            "feature_size": self.feature_size,
            "image_size": self.image_size,
            "save_dir": save_dir,
            "checkpoint_interval": checkpoint_interval,
            "reload_checkpoint": reload_checkpoint,
            "mixed_precision_training": mixed_precision_training,
            "device": str(self.device),
            "num_gpus": torch.cuda.device_count(),
            "num_workers": self.num_workers,
            **kwargs,  # Include any other kwargs passed to Trainer init
        }
        full_wandb_config = {
            **trainer_internal_config,
            **(wandb_config if wandb_config else {}),
        }

        self.wandb_logger = WandbLogger(
            project_name=(
                wandb_project if wandb_project else f"MK_SSL_Vision_{self.method}"
            ),  # Default project name
            entity=wandb_entity,
            mode=wandb_mode,
            run_name=wandb_run_name,
            config=full_wandb_config,
            notes=(
                wandb_notes
                if wandb_notes
                else f"Training {self.method} vision model with MK_SSL."
            ),
            tags=wandb_tags if wandb_tags else [self.method, "vision", "training"],
        )

        # This logic mirrors the WandbLogger initialization to accurately log the project name.
        effective_wandb_project = (
            wandb_project if wandb_project else f"MK_SSL_Vision_{self.method}"
        )

        self.logger.info(
            "\n"
            "-------------------- W&B ---------------------\n"
            f"W&B Active        : {wandb_mode != 'disabled'}\n"
            f"W&B Project       : {effective_wandb_project}\n"
            f"W&B Entity        : {wandb_entity or 'Default'}\n"
            f"W&B Mode          : {wandb_mode}\n"
            f"W&B Run Name      : {wandb_run_name or 'Auto-generated'}\n"
            "----------------------------------------------------"
        )


    @staticmethod
    def safe_collate(batch):
        """Drop None samples from a batch to avoid collate crashes."""
        batch = [b for b in batch if b is not None and b[0] is not None]
        if len(batch) == 0:
            raise RuntimeError("Empty batch after filtering — check dataset.")
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)

    @staticmethod
    def worker_init_fn(worker_id):
        """Ensure different RNG seed per worker."""
        seed = torch.initial_seed() % 2**32
        random.seed(seed)

    def _train_mae(
        self,
        train_loader,
        optimizer,
        epochs,
        start_epoch=1,
        use_embedding_logger: bool = True,
        logger_loader: Optional[DataLoader] = None,
        warmup_ratio: float = 0.05,   # unused (kept for signature compatibility)
        max_grad_norm: float = 1.0,
    ):
        """
        Stable MAE training without any LR scheduling (uses optimizer's constant LR).
        """
        self.model.train()

        # numerics: allow TF32 and prefer safer matmul paths when available
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # choose the safest autocast dtype
        amp_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            else torch.float16
        )

        # MAE target "patchify"
        def patchify(imgs, patch_size):
            B, C, H, W = imgs.shape
            p = patch_size
            assert H == W and H % p == 0
            x = imgs.reshape(B, C, H // p, p, W // p, p)
            x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // p) * (W // p), p * p * C)
            return x

        patch_size = self.model.patch_embed.patch_size
        global_step = (start_epoch - 1) * len(train_loader)

        # optional embedding logger (unchanged, but skip non-finite batches)
        if use_embedding_logger:
            assert logger_loader is not None
            embedding_log_dir = os.path.join(self.checkpoint_path, "embedding_logs")
            embedding_logger = EmbeddingLogger(
                log_dir=embedding_log_dir, method_name=self.method, reduce_method="tsne", log_interval=1
            )
            self.logger.info(f"Embedding logger initialized at {embedding_log_dir}")
            self.logger.info("[MAE - Step 0] Logging pre-training embeddings...")
            backbone = MAEBackbone(self.model).to(self.device)
            backbone.eval()
            all_embeddings, all_labels = [], []
            with torch.no_grad():
                for images, labels in tqdm(logger_loader, desc="EmbeddingLogger Step 0"):
                    if not torch.isfinite(images).all():
                        continue
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    embeddings = backbone(images)
                    all_embeddings.append(embeddings)
                    all_labels.append(labels)
            if all_embeddings:
                embeddings = torch.cat(all_embeddings, dim=0)
                labels = torch.cat(all_labels, dim=0)
                embedding_logger.log_step(step=0, embeddings=embeddings, labels=labels)
            self.logger.info("[MAE - Step 0] Pre-training embeddings logged.")
            self.model.train()

        # main loop (constant LR; no per-step/per-epoch schedule)
        for epoch in range(start_epoch - 1, epochs):
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"MAE Training [Epoch {epoch+1}/{epochs}]")

            for step, (images, _) in enumerate(pbar):
                # skip corrupted batches early
                if not torch.isfinite(images).all():
                    self.logger.warning("[MAE] Non-finite input detected; skipping batch.")
                    continue

                images = images.to(self.device, non_blocking=True)

                # forward (mixed precision)
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=self.mixed_precision_training):
                    # build target in fp32 for stability, then loss handles casting
                    target = patchify(images, patch_size).float()
                    pred, mask = self.model(images)
                    loss = self.loss(pred, target, mask)

                # guard against non-finite loss before backward
                if not torch.isfinite(loss):
                    self.logger.error(f"[MAE] Non-finite loss at step {global_step}: {loss.item()}. Skipping update.")
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    continue

                # backward
                optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()

                # clip grads in fp32 space
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                # step
                self.scaler.step(optimizer)
                self.scaler.update()

                running_loss += float(loss.detach().cpu())
                pbar.set_postfix({
                    "loss": f"{float(loss):.5f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6g}",
                })
                global_step += 1

                if self.wandb_logger.is_active:
                    self.wandb_logger.log(
                        {
                            f"{self.method.upper()}/Train/Batch_Loss": float(loss),
                            f"{self.method.upper()}/Train/LR": optimizer.param_groups[0]["lr"],
                        },
                        step=global_step,
                    )

            epoch_loss = running_loss / max(1, len(train_loader))
            epoch_step = (epoch + 1) * len(train_loader)

            if self.wandb_logger.is_active:
                self.wandb_logger.log(
                    {
                        f"{self.method.upper()}/Train/Epoch_Loss": epoch_loss,
                        f"{self.method.upper()}/Train/LR": optimizer.param_groups[0]["lr"],
                        "epoch": epoch + 1,
                    },
                    step=epoch_step,
                )

            # optional embedding logging per-epoch
            if use_embedding_logger:
                self.logger.info(f"[MAE - Epoch {epoch+1}] Logging embeddings...")
                backbone = MAEBackbone(self.model).to(self.device)
                backbone.eval()
                all_embeddings, all_labels = [], []
                with torch.no_grad():
                    for images, labels in tqdm(logger_loader, desc=f"EmbeddingLogger Epoch {epoch+1}]"):
                        if not torch.isfinite(images).all():
                            continue
                        images = images.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                        embeddings = backbone(images)
                        all_embeddings.append(embeddings)
                        all_labels.append(labels)
                if all_embeddings:
                    embeddings = torch.cat(all_embeddings, dim=0)
                    labels = torch.cat(all_labels, dim=0)
                    embedding_logger.log_step(step=epoch + 1, embeddings=embeddings, labels=labels)
                self.logger.info(f"[MAE - Epoch {epoch+1}] Embeddings logged.")
                self.model.train()

            # checkpointing
            if (epoch + 1) % self.checkpoint_interval == 0:
                ckpt_path = os.path.join(
                    self.checkpoint_path, f"{self.method}_model_{self.timestamp}_epoch{epoch+1}.pth"
                )
                torch.save(self.model.state_dict(), ckpt_path)
                if self.wandb_logger.is_active:
                    self.wandb_logger.save_artifact(
                        ckpt_path,
                        name=f"{self.method}-model-epoch-{epoch+1}",
                        type="model",
                        metadata={"epoch": epoch + 1, "loss": epoch_loss},
                    )

        if use_embedding_logger:
            self.logger.info("Generating final embedding animation...")
            animation_path = embedding_logger.plot_all()
            self.logger.info(f"Embedding animation saved at: {animation_path}")
            if self.wandb_logger.is_active:
                self.wandb_logger.log(
                    {"media/embedding_animation": wandb.Html(animation_path)},
                    step=(max(embedding_logger.steps) if embedding_logger.steps else epochs),
                )
                self.logger.info("Embedding animation logged to Weights & Biases.")

        self.logger.info("MAE training complete.")




    def __del__(self):
        pass  # No need for TensorBoard writer close if not used

    def get_backbone(self):
        return self.model.backbone

    def train_one_epoch(
        self, tepoch, optimizer, epoch_idx, total_batches_per_epoch
    ):  # Added epoch_idx, total_batches_per_epoch
        loss_hist_train = 0.0
        # Watch the model with W&B if active
        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(self.model)

        for step, (images, _) in enumerate(
            tepoch
        ):  # Added step for global step calculation
            images = images.to(self.device)
            if self.method.lower() in ["barlowtwins", "byol", "mocov3"]:
                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    view0 = self.transformation(images)
                    view1 = self.transformation_prime(images)
                    z0, z1 = self.model(view0, view1)
                    loss = self.loss(z0, z1)
            elif self.method.lower() in ["dino"]:
                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    view0 = self.transformation_global1(images)
                    view1 = self.transformation_global2(images)
                    viewc = []
                    if self.model.num_crops > 0:
                        for _ in range(self.model.num_crops):
                            viewc.append(self.transformation_local(images))
                    z0, z1 = self.model(view0, view1, viewc)
                    loss = self.loss(z0, z1)
            elif self.method.lower() in ["swav"]:
                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    view0 = self.transformation_global(images)
                    view1 = self.transformation_global(images)
                    viewc = []
                    if self.model.num_crops > 0:
                        for _ in range(self.model.num_crops):
                            viewc.append(self.transformation_local(images))
                    z0, z1 = self.model(view0, view1, viewc)
                    loss = self.loss(z0, z1)
            else:  # SimCLR, SimSiam, MoCov2 (assuming these use transformation twice)
                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    view0 = self.transformation(images)
                    view1 = self.transformation(images)
                    z0, z1 = self.model(view0, view1)
                    loss = self.loss(z0, z1)

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            loss_hist_train += loss.item()

            # Log batch-level metrics to W&B
            if self.wandb_logger.is_active:
                global_batch_step = (epoch_idx * total_batches_per_epoch) + step
                self.wandb_logger.log(
                    {
                        f"{self.method.upper()}/Train/Batch_Loss": loss.item(),
                        f"{self.method.upper()}/Train/LR": optimizer.param_groups[0][
                            "lr"
                        ],
                    },
                    step=global_batch_step,
                )

            tepoch.set_postfix(loss=loss.item())

        return loss_hist_train

    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        batch_size: int = 256,
        start_epoch: int = 1,
        epochs: int = 100,
        optimizer: str = "Adam",
        weight_decay: float = 1e-6,
        learning_rate: float = 1e-3,
        use_hpo: bool = False,
        n_trials: int = 20,
        tuning_epochs: int = 5,
        use_embedding_logger: Optional[bool] = True,
        logger_loader: Optional[DataLoader] = None,
        **kwargs,
    ):
        """
        Description:
            Train the model.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to train.
            batch_size (int): Batch size.
            start_epoch (int): Epoch to start the training.
            epochs (int): Number of epochs.
            optimizer (str): Optimizer to train the model. Options: [Adam, SGD, AdamW]
            weight_decay (float): Weight decay.
            learning_rate (float): Learning rate.
        """
        # Initialize W&B run at the very beginning of the main train method
        self.train_dataset = train_dataset

        if not hasattr(self, "_optuna_trial"):
            self.wandb_logger.init_run()
        else:
            self.wandb_logger.mode = "disabled"

        if self.wandb_logger.is_active:

            self.wandb_logger.current_run.config.update(
                {
                    "batch_size": batch_size,
                    "start_epoch": start_epoch,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "optimizer": optimizer,
                    **kwargs,
                }
            )
            self.logger.info(
                f"W&B run initialized. View run at: {self.wandb_logger.current_run.url}"
            )
        else:
            self.logger.info("W&B logging is not active for this run.")

        if use_hpo:
            self.logger.info("🧪 Running Optuna for hyperparameter tuning...")

            best_params = optimize_hyperparameters(
                trainer=self,
                train_dataset=train_dataset,
                n_trials=n_trials,
                epochs=tuning_epochs,
            )
            self.logger.info(f"🌟 Best hyperparameters found: {best_params}")

            learning_rate = best_params.get("lr", learning_rate)
            batch_size = best_params.get("batch_size", batch_size)
            weight_decay = best_params.get("weight_decay", weight_decay)
            optimizer = best_params.get("optimizer", optimizer)

            kwargs.update(
                {
                    k: v
                    for k, v in best_params.items()
                    if k not in {"lr", "batch_size", "weight_decay", "optimizer"}
                }
            )

            self.wandb_logger.log(
                {
                    "hpo/best_lr": learning_rate,
                    "hpo/best_batch_size": batch_size,
                    "hpo/best_weight_decay": weight_decay,
                    "hpo/best_optimizer": optimizer,
                    **{f"hpo/{k}": v for k, v in kwargs.items()},
                }
            )
            self.logger.info("📡 Best hyperparameters logged to W&B.")

        match optimizer.lower():
            case "adam":
                optimizer = torch.optim.Adam(
                    list(self.model.parameters()),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
            case "sgd":
                optimizer = torch.optim.SGD(
                    list(self.model.parameters()),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
            case "adamw":
                optimizer = torch.optim.AdamW(
                    list(self.model.parameters()),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
            case _:
                self.logger.error(f"Unsupported Optimizer: {optimizer}")

                raise ValueError(f"Optimizer {optimizer} not supported")

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,                  # safer for Kaggle
            pin_memory=False,               # reduce SHM pressure
            persistent_workers=False,       # restart workers each epoch
            prefetch_factor=2,               # small prefetch
            collate_fn=self.safe_collate,
            worker_init_fn=self.worker_init_fn
        )

        if self.method == "mae":
            
            self._train_mae(
                train_loader=train_loader,
                optimizer=optimizer,
                epochs=epochs,
                start_epoch=start_epoch,
                use_embedding_logger=use_embedding_logger,
                logger_loader=logger_loader,
            )

        else:
            total_batches_per_epoch = len(
                train_loader
            )  # Used for global step calculation

            self.model.train()

            if self.reload_checkpoint:
                start_epoch = self._reload_latest_checkpoint() + 1

            for (
                epoch
            ) in tqdm(  # epoch is 0-indexed loop variable (range(start-1, epochs))
                range(start_epoch - 1, epochs),
                unit="epoch",
                desc="Pretext Task Model Training",
                leave=True,
            ):
                with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                    tepoch.set_description(f"Epoch {epoch + 1}")
                    loss_per_epoch = self.train_one_epoch(
                        tepoch, optimizer, epoch, total_batches_per_epoch
                    )  # Pass epoch_idx, total_batches_per_epoch

                # To stop from full training for optuna
                if hasattr(self, "_optuna_trial"):
                    self._optuna_trial.report(loss_per_epoch, epoch)
                    if self._optuna_trial.should_prune():
                        raise optuna.TrialPruned()

                epoch_step = (epoch + 1) * len(train_loader)        

                # Log epoch-level metrics to W&B
                if self.wandb_logger.is_active:
                    self.wandb_logger.log(
                        {
                            f"{self.method.upper()}/Train/Epoch_Loss": loss_per_epoch
                            / len(train_loader),
                            f"{self.method.upper()}/Train/LR": optimizer.param_groups[
                                0
                            ]["lr"],
                            "epoch": epoch + 1,  
                        },
                        step=epoch_step
                    )  # Use epoch + 1 for 1-indexed epoch step

                if (epoch + 1) % self.checkpoint_interval == 0:
                    model_path = (
                        self.checkpoint_path
                        + "/{}_model_{}_epoch{}.pth".format(  # Added / for path joining
                            self.method, self.timestamp, epoch + 1
                        )
                    )
                    torch.save(self.model.state_dict(), model_path)
                    self.logger.info(f"Model checkpoint saved: {model_path}")
                    # Save model checkpoint as W&B artifact
                    if self.wandb_logger.is_active:
                        self.wandb_logger.save_artifact(
                            model_path,
                            name=f"{self.method}-model-epoch-{epoch+1}",
                            type="model",
                            metadata={
                                "epoch": epoch + 1,
                                "loss": loss_per_epoch / len(train_loader),
                            },
                        )

        # Save final model after all epochs
        # Note: 'epoch' here will be the last value from the loop, which is `epochs - 1` (0-indexed)
        # So, for the filename, it should be `epochs` (1-indexed total epochs)
        final_model_path = (
            self.checkpoint_path
            + "/{}_model_{}_final.pth".format(  # Changed to final
                self.method, self.timestamp
            )
        )
        torch.save(self.model.state_dict(), final_model_path)
        self.logger.info(f"Final model saved: {final_model_path}")
        # Save final model as W&B artifact
        if self.wandb_logger.is_active:
            self.wandb_logger.save_artifact(
                final_model_path,
                name=f"{self.method}-model-final",
                type="model",
                metadata={
                    "epochs_trained": epochs,
                    "final_loss": loss_per_epoch / len(train_loader),
                },  # Use final epoch's loss
            )

        training_mode = "Main" if not hasattr(self, "_optuna_trial") else "HPO"
        if self.wandb_logger.is_active:
            self.wandb_logger.finish_run()
            self.logger.info(
                f"{training_mode} training process completed and W&B run finalized."
            )
        else:
            self.logger.info(f"{training_mode} training process completed.")

    def evaluate(
        self,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        eval_method: str = "linear",
        top_k: int = 1,
        epochs: int = 100,
        optimizer: str = "Adam",
        weight_decay: float = 1e-6,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        fine_tuning_data_proportion: float = 1,
        **kwargs,
    ):
        """
        Description:
            Evaluate the model using the given evaluating method.

        Args:
            eval_method (str): Evaluation method. Options: [linear, finetune]
            top_k (int): Top k accuracy.
            epochs (int): Number of epochs.
            optimizer (str): Optimizer to train the model. Options: [Adam, SGD, AdamW]
            weight_decay (float): Weight decay.
            learning_rate (float): Learning rate.
            batch_size (int): Batch size.
            train_dataset (torch.utils.data.Dataset): Dataset to train the downstream model.
            test_dataset (torch.utils.data.Dataset): Dataset to test the downstream model.
            fine_tuning_data_proportion (float): Proportion of the dataset between 0 and 1 to use for fine-tuning.

        """
        # Start W&B run for evaluation if not already active (e.g., if evaluation is run standalone)
        # If train() was called before, the run might still be active.
        # This ensures evaluation metrics are logged to the same run or a new one.
        if not self.wandb_logger.is_active:
            # Re-init W&B logger for evaluation context if not already active from training
            # This might create a new run if not explicitly linked to a previous one.
            # For simplicity, we'll assume a new run if not active.
            # You might want to add a specific project/run_name for evaluation runs.
            self.logger.info("W&B logger not active, initializing for evaluation.")
            self.wandb_logger.init_run(
                "Evaluation"
            )  # This will create a new run if none is active

        # Log evaluation parameters to W&B config
        if self.wandb_logger.is_active:
            self.wandb_logger.current_run.config.update(
                {
                    "eval_method": eval_method,
                    "eval_top_k": top_k,
                    "eval_epochs": epochs,
                    "eval_optimizer": optimizer,
                    "eval_weight_decay": weight_decay,
                    "eval_learning_rate": learning_rate,
                    "eval_batch_size": batch_size,
                    "fine_tuning_data_proportion": fine_tuning_data_proportion,
                }
            )
            self.logger.info("Evaluation parameters logged to W&B config.")

        match eval_method.lower():
            case "linear":
                net = EvaluateNet(
                    self.model.backbone,
                    self.feature_size,
                    len(train_dataset.classes),
                    True,
                )
            case "finetune":
                if not 0 <= fine_tuning_data_proportion <= 1:

                    self.logger.error(
                        f"The fine_tuning_data_proportion parameter must be between 0 and 1."
                    )

                    raise ValueError(
                        "The fine_tuning_data_proportion parameter must be between 0 and 1."
                    )

                net = EvaluateNet(
                    self.model.backbone,
                    self.feature_size,
                    len(train_dataset.classes),
                    False,
                )

                num_samples = len(train_dataset)
                subset_size = int(num_samples * fine_tuning_data_proportion)

                indices = torch.randperm(num_samples)[:subset_size]

                train_dataset = Subset(train_dataset, indices)

        match optimizer.lower():
            case "adam":
                optimizer_eval = torch.optim.Adam(
                    net.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
            case "sgd":
                optimizer_eval = torch.optim.SGD(
                    net.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
            case "adamw":
                optimizer_eval = torch.optim.AdamW(
                    net.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
            case _:

                self.logger.error(f"Unsupported Optimizer: {optimizer}")

                raise ValueError(f"Optimizer {optimizer} not supported")

        net = net.to(self.device)
        criterion = nn.CrossEntropyLoss()

        train_loader_ds = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,  # Add num_workers
        )

        train_loader_ds = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,  # Add num_workers
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2,
            collate_fn=self.safe_collate,
            worker_init_fn=self.worker_init_fn
        )

        if self.method == 'mae':
            self._evaluate_mae(
                train_dataset = train_dataset,
                test_dataset = test_dataset,
                batch_size = batch_size,
                epochs = epochs,
                lr_head = learning_rate,
                lr_backbone = learning_rate*0.1,          # used only in finetune
                optimizer_cls=optimizer_eval,
                **kwargs
                )
            return 0.0 

        total_batches_per_eval_epoch = len(
            train_loader_ds
        )  # For global step calculation

        net.train(True)
        scaler_eval = torch.cuda.amp.GradScaler(enabled=self.mixed_precision_training)

        for epoch in tqdm(
            range(epochs),
            unit="epoch",
            desc="Evaluate Model Training",
            leave=True,
        ):
            with tqdm(train_loader_ds, unit="batch", leave=False) as tepoch_ds:
                tepoch_ds.set_description(f"Epoch {epoch + 1}")
                loss_hist_train, acc_hist_train = 0.0, 0.0

                for step, (images, labels) in enumerate(tepoch_ds):  # Added step
                    correct, total = 0, 0

                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                        outputs = net(images)
                        loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    acc = 100 * correct / total
                    acc_hist_train += acc

                    tepoch_ds.set_postfix(loss=loss.item(), accuracy=f"{acc:.2f}")
                    loss_hist_train += loss.item()
                    optimizer_eval.zero_grad()
                    scaler_eval.scale(loss).backward()
                    scaler_eval.step(optimizer_eval)
                    scaler_eval.update()

                    # Log batch-level evaluation train metrics to W&B
                    if self.wandb_logger.is_active:
                        global_eval_batch_step = (
                            epoch * total_batches_per_eval_epoch
                        ) + step
                        self.wandb_logger.log(
                            {
                                f"Downstream Task/{eval_method.capitalize()}/Batch_Loss": loss.item(),
                                f"Downstream Task/{eval_method.capitalize()}/Batch_Accuracy": acc,
                                f"Downstream Task/{eval_method.capitalize()}/LR": optimizer_eval.param_groups[
                                    0
                                ][
                                    "lr"
                                ],
                            },
                            step=global_eval_batch_step,
                        )

                # Log epoch-level evaluation train metrics to W&B
                if self.wandb_logger.is_active:
                    self.wandb_logger.log(
                        {
                            f"Downstream Task/{eval_method.capitalize()}/Epoch_Loss": loss_hist_train
                            / len(train_loader_ds),
                            f"Downstream Task/{eval_method.capitalize()}/Epoch_Accuracy": acc_hist_train
                            / len(train_loader_ds),
                        },
                        step=epoch + 1,
                    )

        test_loader_ds = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,  # Add num_workers
        )

        acc_test = 0.0
        net.eval()
        with torch.no_grad():
            for images, labels in tqdm(test_loader_ds, unit="batch"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                acc_test += multiclass_accuracy(outputs, labels, k=top_k).item()

        final_test_accuracy = 100 * acc_test / len(test_loader_ds)

        self.logger.info(
            "\n"
            "---------------- Test Accuracy ----------------\n"
            f"The top_{top_k} accuracy of the network on the {len(test_dataset)} test images: {final_test_accuracy:.2f}%\n"  # Formatted for clarity
            "-----------------------------------------------"
        )

        # Log final test accuracy to W&B summary
        if self.wandb_logger.is_active:
            self.wandb_logger.log(
                {
                    f"Downstream Task/{eval_method.capitalize()}/Final_Test_Accuracy_top_{top_k}": final_test_accuracy
                }
            )
            # Also add to summary for easy comparison across runs
            self.wandb_logger.current_run.summary[
                f"final_test_accuracy_top_{top_k}"
            ] = final_test_accuracy

        # Finish W&B run after evaluation if it was started by evaluate and not already finished by train
        if self.wandb_logger.is_active:  # Check again if it's still active
            self.wandb_logger.finish_run()
            self.logger.info("Evaluation process completed and W&B run finalized.")
        else:
            self.logger.info("Evaluation process completed.")

        return final_test_accuracy


    def _evaluate_mae(
        self,
        train_dataset,
        test_dataset,
        num_classes: int,
        *,
        mode: str = "linear",               # "linear" or "finetune"
        batch_size: int = 64,
        epochs: int = 10,
        # Optimizer hyperparams
        lr_head: float = 1e-3,
        lr_backbone: float = 5e-5,          # used only in finetune
        wd_head: float = 0.00,
        wd_backbone: float = 0.05,
        optimizer_cls=torch.optim.AdamW,
        # Scheduler (optional cosine)
        use_cosine: bool = False,
        cosine_T_max: int | None = None,    # defaults to epochs if None
        # Misc
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs,
    ):
        """
        Evaluate a MAE encoder with either linear probing or fine-tuning.

        Args:
            train_dataset, test_dataset: labeled datasets (x, y).
            num_classes: number of classes.
            mode: "linear" (freeze encoder) or "finetune" (train encoder + head).
            batch_size, epochs: training hyperparameters.
            lr_head, lr_backbone: learning rates for head/encoder.
            wd_head, wd_backbone: weight decays for head/encoder.
            optimizer_cls: torch optimizer class (default AdamW).
            use_cosine: whether to apply CosineAnnealingLR.
            cosine_T_max: T_max for cosine scheduler (defaults to epochs).
            num_workers, pin_memory: DataLoader options.
        """

        assert mode in {"linear", "finetune"}, "mode must be 'linear' or 'finetune'"
        freeze_backbone = (mode == "linear")

        # === Backbone and classifier ===
        backbone = MAEBackbone(self.model)  # wraps your MAE encoder
        # If your backbone exposes a feature dim directly, prefer that:
        try:
            feature_size = backbone.encoder.head.in_features  # as in your current code
        except Exception:
            # Fallback: infer by a tiny forward pass on one batch (safe on CPU/GPU)
            tmp_loader = DataLoader(train_dataset, batch_size=1)
            x0, _ = next(iter(tmp_loader))
            x0 = x0.to(self.device)
            backbone = backbone.to(self.device).eval()
            with torch.no_grad():
                f0 = backbone(x0)
            feature_size = f0.shape[-1]

        classifier = EvaluateNet(
            backbone=backbone,
            feature_size=feature_size,
            num_classes=num_classes,
            is_linear=freeze_backbone,  # your EvaluateNet can use this to freeze internally if you prefer
        ).to(self.device)

        # Freeze/unfreeze explicitly to be robust regardless of EvaluateNet internals
        for p in classifier.backbone.parameters():
            p.requires_grad = not freeze_backbone
        for p in classifier.head.parameters():
            p.requires_grad = True

        # Param groups (head only for linear; head + backbone for finetune)
        if freeze_backbone:
            params = [{"params": classifier.head.parameters(), "lr": lr_head, "weight_decay": wd_head}]
        else:
            params = [
                {"params": classifier.backbone.parameters(), "lr": lr_backbone, "weight_decay": wd_backbone},
                {"params": classifier.head.parameters(),     "lr": lr_head,     "weight_decay": wd_head},
            ]

        optimizer = optimizer_cls(params)
        criterion = torch.nn.CrossEntropyLoss()

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )

        scheduler = None
        if use_cosine:
            T_max = cosine_T_max if cosine_T_max is not None else epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(classifier)

        # === Training ===
        for epoch in range(epochs):
            classifier.train()
            running_loss, n = 0.0, 0

            for x, y in train_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

                logits = classifier(x)  # inside, it will call backbone + head
                loss = criterion(logits, y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * y.size(0)
                n += y.size(0)

            if scheduler is not None:
                scheduler.step()

            avg_loss = running_loss / max(1, n)
            self.logger.info(
                f"[MAE Eval:{mode}] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}"
            )

            if self.wandb_logger.is_active:
                # pull current lrs from param groups
                lrs = {f"pg{i}_lr": g["lr"] for i, g in enumerate(optimizer.param_groups)}
                self.wandb_logger.log(
                    {
                        f"mae/{mode}_train_loss": avg_loss,
                        f"mae/{mode}_epoch": epoch + 1,
                        **lrs,
                    },
                    step=epoch + 1,
                )

        # === Evaluation ===
        classifier.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device, non_blocking=True)
                logits = classifier(x)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        txt_report = classification_report(all_labels.numpy(), all_preds.numpy(), digits=4)
        self.logger.info("\n📊 [MAE Evaluation Report - "
                         f"{mode.upper()}]:\n{txt_report}")

        report = classification_report(
            all_labels.numpy(), all_preds.numpy(), digits=4, output_dict=True
        )

        if self.wandb_logger.is_active:
            self.wandb_logger.log(
                {
                    f"mae/{mode}_test_accuracy": report["accuracy"],
                    f"mae/{mode}_test_macro_f1": report["macro avg"]["f1-score"],
                    f"mae/{mode}_test_macro_precision": report["macro avg"]["precision"],
                    f"mae/{mode}_test_macro_recall": report["macro avg"]["recall"],
                }
            )

        return report




    

    def load_checkpoint(self, checkpont_dir: str):
        self.model.load_state_dict(
            torch.load(checkpont_dir, map_location=self.device)
        )  # Add map_location
        self.logger.info(
            "\n"
            "---------------- Checkpoint ----------------\n"
            "Checkpoint loaded.\n"
            "--------------------------------------------"
        )

    def save_backbone(self):
        # Ensure save_dir has a trailing slash or use os.path.join
        backbone_path = os.path.join(self.save_dir, "backbone.pth")
        torch.save(self.model.backbone.state_dict(), backbone_path)

        self.logger.info(
            "\n"
            "---------------- Save Backbone ----------------\n"
            "Backbone saved.\n"
            f"Backbone file path : {backbone_path}\n"
            "------------------------------------------------"
        )
        # Save backbone as W&B artifact
        if self.wandb_logger.is_active:
            self.wandb_logger.save_artifact(
                backbone_path,
                name=f"{self.method}-backbone",
                type="model_backbone",
                metadata={
                    "feature_size": self.feature_size,
                    "image_size": self.image_size,
                },
            )

    def _reload_latest_checkpoint(self):
        checkpoints = os.listdir(self.checkpoint_path)
        sorted_checkpoints = sorted(
            [os.path.join(self.checkpoint_path, i) for i in checkpoints],
            key=os.path.getmtime,
        )

        if len(sorted_checkpoints) == 0:

            self.logger.error(f"No checkpoints found in the directory")

            raise ValueError("No checkpoints found in the directory")

        self.load_checkpoint(sorted_checkpoints[-1])

        match = re.search(r"epoch(\d+)", sorted_checkpoints[-1])
        if match:
            epoch = int(match.group(1))

            self.logger.info(
                "\n"
                "---------------- Checkpoint Reload ----------------\n"
                f"Starting Epoch : {epoch}\n"
                "---------------------------------------------------"
            )

        else:
            self.logger.error(f"No epoch number found in the checkpoint name.")

            raise ValueError("No epoch number found in the checkpoint name.")

        return epoch
