import os
import re
import torch
import numpy as np
from torch import nn
from tqdm.auto import (
    tqdm,
)
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.dataloader import default_collate

import logging

from typing import Optional, Dict, Any
from sklearn.metrics import classification_report
import wandb
import optuna




from MK_SSL.audio.models.utils.registry import get_method
from MK_SSL.audio.models.utils.datasets import HuBERTWrapperDataset

from MK_SSL.audio.models.modules.tools import PseudoLabelGenerator

from MK_SSL.audio.models.modules.cola_backbone import COLABackbone
from MK_SSL.audio.models.modules.wav2vec2_backbone import Wav2Vec2Backbone
from MK_SSL.audio.models.modules.hubert_backbone import HuBERTBackbone
from MK_SSL.audio.models.modules.simclr_backbone import SimCLRBackbone

from MK_SSL.utils import EvaluateNet
from MK_SSL.utils import EmbeddingLogger
from MK_SSL.utils import get_logger_handler
from MK_SSL.utils import WandbLogger
from MK_SSL.utils import optimize_hyperparameters


class Trainer:

    def __init__(
        self,
        method: str,
        backbone: nn.Module = None,
        variant: str = None,
        save_dir: str = ".",
        checkpoint_interval: int = 10,
        reload_checkpoint: bool = False,
        verbose: bool = True,
        mixed_precision_training: bool = True,
        # W&B specific arguments
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_mode: str = "online", # "online", "offline", "disabled"
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        wandb_notes: Optional[str] = None,
        wandb_tags: Optional[list[str]] = None,
        use_data_parallel: bool = False,
        **kwargs,
    ) -> None:
        """
        Initializes the Trainer class for audio self-supervised learning.

        Args:
            method (str): SSL method name (e.g., 'wav2vec2').
            backbone (nn.Module): Model backbone architecture (e.g., ConvNet, Transformer).
            variant (str): Architecture variant (e.g., 'base', 'large') used for model config.
            save_dir (str, optional): Directory to save checkpoints and logs. Defaults to ".".
            checkpoint_interval (int, optional): Frequency (in epochs) to save model checkpoints. Defaults to 10.
            reload_checkpoint (bool, optional): Whether to reload the most recent checkpoint. Defaults to False.
            configure_logger (bool, optional): Whether to initialize logging. Defaults to True.
            verbose (bool, optional): Verbosity flag for logger level. Defaults to True.
            mixed_precision_training (bool, optional): Enable AMP mixed precision training. Defaults to True.
            wandb_project (str, optional): W&B project name. If None, uses default from W&B.
            wandb_entity (str, optional): W&B entity (username or team name). If None, uses default.
            wandb_mode (str, optional): W&B logging mode ("online", "offline", "disabled"). Defaults to "online".
            wandb_run_name (str, optional): Custom name for the W&B run.
            wandb_config (Dict[str, Any], optional): Dictionary of hyperparameters/settings for W&B.
            wandb_notes (str, optional): Notes for the W&B run.
            wandb_tags (list[str], optional): Tags for the W&B run.
            **kwargs: Additional keyword arguments passed to the model or loss.
        """


        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.propagate = False

        if not self.logger.hasHandlers():
            self.logger.addHandler(get_logger_handler())

        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self.logger.info("Audio Trainer initialized.")

        self.method = method.lower()
        self.backbone = backbone
        self.mixed_precision_training = mixed_precision_training
        self.checkpoint_interval = checkpoint_interval
        self.reload_checkpoint = reload_checkpoint

        self.save_dir = os.path.join(save_dir, self.method)
        os.makedirs(self.save_dir, exist_ok=True)

        self.checkpoint_path = os.path.join(self.save_dir, "Pretext")
        os.makedirs(self.checkpoint_path, exist_ok=True)

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
                    "---------------- MK_SSL: Audio ----------------\n"
                    f"Number of workers : {self.num_workers}\n"
                    f"Number of GPUs    : {torch.cuda.device_count()}\n"
                    f"Device            : {self.device}\n"
                    f"Method            : {self.method}\n"
                    "----------------------------------------------------"

                )
        


        # --- Load Model Config ---

        try:
            method_cfg = get_method(self.method)
        except ValueError as e:
            self.logger.error(f"Method {self.method} not found in registry.")
            raise e

        # --- Model Args ---

        model_args = {
            "variant": variant,
        }

        if "params" in method_cfg:
            model_args.update(method_cfg["default_params"])

        model_args.update(kwargs)

        # --- Loss Args ---

        loss_args = {}

        if "params" in method_cfg:
            loss_args.update(method_cfg["default_params"])

        loss_args.update(kwargs)

        # --- Create Generic Model ---
        self.model = method_cfg["model"](**model_args)

        # --- Create Generic Loss ---
        self.loss = method_cfg["loss"](**loss_args)

        # --- Create Generic Transformation ---
        self.transformation = (
            method_cfg["transformation"]()
            if method_cfg["transformation"] is not None
            else None
        )

        self.logger.info(method_cfg["logs"](self.model, self.loss))


        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision_training)

        if use_data_parallel:
            self.logger.info(f"Wrapping model with DataParallel using {torch.cuda.device_count()} GPUs.")
            self.model = nn.DataParallel(self.model)

        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)

        kmeans_clusters = kwargs.get(
            "kmeans_clusters", getattr(self.model, "num_clusters", 100)
        )
        sample_rate = kwargs.get("sample_rate", 16000)
        if self.method == 'hubert':
            self.pseudo_label_generator = PseudoLabelGenerator(
                kmeans_clusters=kmeans_clusters,
                sample_rate=sample_rate,
                save_dir=os.path.join(self.save_dir, "hubert_pseudo_labels"),
                logger=self.logger,  # Pass logger to the generator
            )

        self.logger.info(
            "\n"
            "---------------- Model Summary ----------------\n"
            f"Model parameters : {np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}\n"
            "----------------------------------------------"
        )

        # --- W&B Logger Initialization ---
        trainer_internal_config = {
            "method": self.method,
            "variant": variant,
            "save_dir": save_dir,
            "checkpoint_interval": checkpoint_interval,
            "reload_checkpoint": reload_checkpoint,
            "mixed_precision_training": mixed_precision_training,
            "device": str(self.device),
            "num_workers": self.num_workers,
            "num_gpus" : torch.cuda.device_count(),
            "kmeans_clusters": kmeans_clusters,
            "sample_rate": sample_rate,
            **kwargs # Include any other kwargs passed to Trainer init
        }
        full_wandb_config = {**trainer_internal_config, **(wandb_config if wandb_config else {})}

        self.wandb_logger = WandbLogger(
            project_name=wandb_project if wandb_project else f"MK_SSL_{self.method}", # Default project name
            entity=wandb_entity,
            mode=wandb_mode,
            run_name=wandb_run_name,
            config=full_wandb_config,
            notes=wandb_notes if wandb_notes else f"Training {self.method} model with MK_SSL.",
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


    def _train_wav2vec2(
        self,
        train_loader: DataLoader,
        optimizer,
        epochs: int,
        start_epoch: int = 0,
        val_loader: Optional[DataLoader] = None,
        logger_loader: Optional[DataLoader] = None,
        use_embedding_logger: bool = False,
    ):
        """Train the Wav2Vec2 model with embedding logging via Wav2Vec2Backbone."""

        self.logger.info(f"Starting training for Wav2Vec2 for {epochs} epochs.")
        self.model.train()

        # Initialize embedding logger
        if use_embedding_logger:
            assert logger_loader is not None, "logger_loader must be provided when use_embedding_logger=True"
            embedding_log_dir = os.path.join(self.checkpoint_path, "embedding_logs")
            embedding_logger = EmbeddingLogger(
                log_dir=embedding_log_dir,
                method_name=self.method,
                reduce_method="tsne",
                log_interval=1,
            )
            self.logger.info(f"Embedding logger initialized at {embedding_log_dir}")

            # === Step 0: log initial embeddings before training ===
            self.logger.info("[Wav2Vec2 - Step 0] Logging pre-training embeddings...")
            backbone = Wav2Vec2Backbone(self.model).to(self.device)
            backbone.eval()

            all_embeddings, all_labels = [], []
            with torch.no_grad():
                for batch in tqdm(logger_loader, desc="EmbeddingLogger Step 0"):
                    audio = batch["audio"].to(self.device)
                    lengths = batch['length'].to(self.device)
                    labels = batch["label"].to(self.device)

                    embeddings = backbone(audio, lengths)
                    all_embeddings.append(embeddings)
                    all_labels.append(labels)

            embeddings = torch.cat(all_embeddings, dim=0)
            labels = torch.cat(all_labels, dim=0)
            embedding_logger.log_step(step=0, embeddings=embeddings, labels=labels)
            self.logger.info("[Wav2Vec2 - Step 0] Pre-training embeddings logged.")
            self.model.train()

        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(self.model)

        # === Training Loop ===
        for epoch in range(start_epoch, epochs):
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Wav2Vec2 Epoch {epoch+1}/{epochs}")

            for batch_idx, batch in enumerate(pbar):
                audio = batch['audio'].to(self.device)
                lengths = batch['length'].to(self.device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    context_features, quantized_targets, codevector_probs, time_mask_indices = self.model(audio, lengths)

                    loss = self.loss(
                        context=context_features,
                        quantized=quantized_targets,
                        codevector_probs=codevector_probs,
                        time_mask_indices=time_mask_indices,
                    )

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                running_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

                global_step = epoch * len(train_loader) + batch_idx
                if self.wandb_logger.is_active:
                    self.wandb_logger.log({"train/batch_loss": loss.item()}, step=global_step)

            avg_loss = running_loss / len(train_loader)
            self.logger.info(f"[Wav2Vec2 - Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

            if self.wandb_logger.is_active:
                self.wandb_logger.log({"train/epoch_loss": avg_loss}, step=epoch + 1)

            # === Log embeddings during training ===
            if use_embedding_logger:
                self.logger.info(f"[Wav2Vec2 - Epoch {epoch+1}] Logging embeddings...")
                backbone = Wav2Vec2Backbone(self.model).to(self.device)
                backbone.eval()

                all_embeddings, all_labels = [], []
                with torch.no_grad():
                    for batch in tqdm(logger_loader, desc=f"EmbeddingLogger Epoch {epoch+1}"):
                        audio = batch["audio"].to(self.device)
                        lengths = batch['length'].to(self.device)
                        labels = batch["label"].to(self.device)

                        embeddings = backbone(audio, lengths)
                        all_embeddings.append(embeddings)
                        all_labels.append(labels)

                embeddings = torch.cat(all_embeddings, dim=0)
                labels = torch.cat(all_labels, dim=0)

                embedding_logger.log_step(step=epoch + 1, embeddings=embeddings, labels=labels)
                self.logger.info(f"[Wav2Vec2 - Epoch {epoch+1}] Embeddings logged.")
                self.model.train()

            if val_loader:
                avg_val_loss = self._validate_wav2vec2(val_loader, epoch)

            if hasattr(self, "_optuna_trial"):
                metric = avg_val_loss if val_loader else avg_loss
                self._optuna_trial.report(metric, epoch)
                if self._optuna_trial.should_prune():
                    raise optuna.TrialPruned()

            if (epoch + 1) % self.checkpoint_interval == 0 and not hasattr(self, "_optuna_trial"):
                model_path = os.path.join(self.checkpoint_path, f"{self.method}_model_{self.timestamp}_epoch{epoch+1}.pth")
                torch.save(self.model.state_dict(), model_path)
                self.logger.info(f"Model checkpoint saved: {model_path}")

                if self.wandb_logger.is_active:
                    self.wandb_logger.save_artifact(
                        model_path,
                        name=f"{self.method}-model-epoch-{epoch+1}",
                        type="model",
                        metadata={"epoch": epoch+1, "loss": avg_loss}
                    )

        # === Save final model ===
        final_path = os.path.join(self.checkpoint_path, f"{self.method}_model_{self.timestamp}_final.pth")
        torch.save(self.model.state_dict(), final_path)
        self.logger.info(f"Final model checkpoint saved: {final_path}")

        if self.wandb_logger.is_active:
            self.wandb_logger.save_artifact(
                final_path,
                name=f"{self.method}-model-final",
                type="model",
                metadata={"epochs_trained": epochs, "final_loss": avg_loss}
            )

        if use_embedding_logger:
            self.logger.info("Generating final embedding animation...")
            animation_path = embedding_logger.plot_all()
            self.logger.info(f"Embedding animation saved at: {animation_path}")

            if self.wandb_logger.is_active:
                import wandb
                self.wandb_logger.log(
                    {"embedding_animation": wandb.Html(animation_path)},
                    step=max(embedding_logger.steps) if embedding_logger.steps else epochs
                )
                self.logger.info("Embedding animation logged to Weights & Biases.")

        self.logger.info("Wav2Vec2 training complete.")


    def _validate_wav2vec2(self, val_loader: DataLoader, epoch: int) -> float:
        """Perform validation for the Wav2Vec2 model.

        Args:
            val_loader (DataLoader): PyTorch DataLoader for validation data.
            epoch (int): Current epoch number for logging.

        Returns:
            float: Average validation loss for the current epoch.
        """
        self.model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Validation Wav2Vec2 Epoch {epoch+1}")
            for batch in pbar:
                audio = batch['audio'].to(self.device)
                lengths = batch['length'].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    context_features, quantized_targets, codevector_probs, time_mask_indices = self.model(audio, lengths)

                    loss = self.loss(
                        context=context_features,
                        quantized=quantized_targets,
                        codevector_probs=codevector_probs,
                        time_mask_indices=time_mask_indices,
                    )

                val_running_loss += loss.item()

            avg_val_loss = val_running_loss / len(val_loader)
            self.logger.info(f"[Wav2Vec2 - Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}")

            if self.wandb_logger.is_active:
                self.wandb_logger.log(
                    {"val/loss": avg_val_loss},
                    step=epoch + 1  # Use epoch number as step for epoch-level metrics
                )
        self.model.train()
        return avg_val_loss


    def _train_simclr(
        self,
        train_loader,
        optimizer,
        epochs: int,
        start_epoch: int = 0,
        val_loader: Optional[DataLoader] = None,
        logger_loader: Optional[DataLoader] = None,
        use_embedding_logger: bool = False,
    ) -> None:
        """Train the SimCLR model with embedding logging via SimCLRBackbone."""
        if self.transformation is None:
            self.logger.error("Transformation not given!")
            raise ValueError("Transformation not given!")

        self.model.train()

        # === Step 0: log initial embeddings ===
        if use_embedding_logger:
            assert logger_loader is not None, "logger_loader must be provided when use_embedding_logger=True"
            embedding_log_dir = os.path.join(self.checkpoint_path, "embedding_logs")
            embedding_logger = EmbeddingLogger(
                log_dir=embedding_log_dir,
                method_name=self.method,
                reduce_method="tsne",
                log_interval=1,
            )
            self.logger.info(f"Embedding logger initialized at {embedding_log_dir}")

            self.logger.info("[SimCLR - Step 0] Logging pre-training embeddings...")
            backbone = SimCLRBackbone(self.model).to(self.device)
            backbone.eval()

            all_embeddings, all_labels = [], []
            with torch.no_grad():
                for batch in tqdm(logger_loader, desc="EmbeddingLogger Step 0"):
                    audio = batch["audio"].to(self.device)
                    labels = batch["label"].to(self.device)
                    view0, _ = self.transformation(audio)
                    embeddings = backbone(view0)  # Use SimCLRBackbone for embeddings
                    all_embeddings.append(embeddings)
                    all_labels.append(labels)

            embeddings = torch.cat(all_embeddings, dim=0)
            labels = torch.cat(all_labels, dim=0)
            embedding_logger.log_step(step=0, embeddings=embeddings, labels=labels)
            self.logger.info("[SimCLR - Step 0] Pre-training embeddings logged.")
            self.model.train()

        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(self.model)

        # === Training Loop ===
        for epoch in range(start_epoch, epochs):
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"SimCLR Epoch {epoch+1}/{epochs}")

            for batch_idx, batch in enumerate(pbar):
                audio = batch["audio"].to(self.device)
                view0, view1 = self.transformation(audio)

                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    out0, out1 = self.model(view0, view1)
                    loss = self.loss(out0, out1)

                optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                running_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

                global_step = epoch * len(train_loader) + batch_idx
                if self.wandb_logger.is_active:
                    self.wandb_logger.log({"train/batch_loss": loss.item()}, step=global_step)

            avg_loss = running_loss / len(train_loader)
            self.logger.info(f"[SimCLR - Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

            if self.wandb_logger.is_active:
                self.wandb_logger.log({"train/epoch_loss": avg_loss}, step=epoch + 1)

            # === Embedding logger ===
            if use_embedding_logger:
                self.logger.info(f"[SimCLR - Epoch {epoch+1}] Logging embeddings...")
                backbone = SimCLRBackbone(self.model).to(self.device)
                backbone.eval()

                all_embeddings, all_labels = [], []
                with torch.no_grad():
                    for batch in tqdm(logger_loader, desc=f"EmbeddingLogger Epoch {epoch+1}"):
                        audio = batch["audio"].to(self.device)
                        labels = batch["label"].to(self.device)
                        view0, _ = self.transformation(audio)
                        embeddings = backbone(view0)  # Use SimCLRBackbone for embeddings
                        all_embeddings.append(embeddings)
                        all_labels.append(labels)

                embeddings = torch.cat(all_embeddings, dim=0)
                labels = torch.cat(all_labels, dim=0)
                embedding_logger.log_step(step=epoch + 1, embeddings=embeddings, labels=labels)
                self.logger.info(f"[SimCLR - Epoch {epoch+1}] Embeddings logged.")
                self.model.train()

            if val_loader:
                avg_val_loss = self._validate_simclr(val_loader, epoch)

            if hasattr(self, "_optuna_trial"):
                metric = avg_val_loss if val_loader else avg_loss
                self._optuna_trial.report(metric, epoch)
                if self._optuna_trial.should_prune():
                    raise optuna.TrialPruned()

            if (epoch + 1) % self.checkpoint_interval == 0 and not hasattr(self, "_optuna_trial"):
                model_path = os.path.join(
                    self.checkpoint_path,
                    f"{self.method}_model_{self.timestamp}_epoch{epoch+1}.pth",
                )
                torch.save(self.model.state_dict(), model_path)
                self.logger.info(f"Model checkpoint saved: {model_path}")

                if self.wandb_logger.is_active:
                    self.wandb_logger.save_artifact(
                        model_path,
                        name=f"{self.method}-model-epoch-{epoch+1}",
                        type="model",
                        metadata={"epoch": epoch+1, "loss": avg_loss}
                    )

        final_path = os.path.join(self.checkpoint_path, f"{self.method}_model_{self.timestamp}_final.pth")
        torch.save(self.model.state_dict(), final_path)
        self.logger.info(f"Final model checkpoint saved: {final_path}")

        if self.wandb_logger.is_active:
            self.wandb_logger.save_artifact(
                final_path,
                name=f"{self.method}-model-final",
                type="model",
                metadata={"epochs_trained": epochs, "final_loss": avg_loss}
            )

        if use_embedding_logger:
            self.logger.info("Generating final embedding animation...")
            animation_path = embedding_logger.plot_all()
            self.logger.info(f"Embedding animation saved at: {animation_path}")

            if self.wandb_logger.is_active:
                import wandb
                self.wandb_logger.log(
                    {"embedding_animation": wandb.Html(animation_path)},
                    step=max(embedding_logger.steps) if embedding_logger.steps else epochs
                )
                self.logger.info("Embedding animation logged to Weights & Biases.")

        self.logger.info("SimCLR training complete.")


    def _validate_simclr(self, val_loader: DataLoader, epoch: int):

        self.model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Validation SimCLR Epoch {epoch+1}")
            for batch in pbar:
                audio = batch['audio'].to(self.device)
                view0, view1 = self.transformation(audio)
                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    out0, out1 = self.model(view0, view1)
                    loss = self.loss(out0, out1)

                val_running_loss += loss.item()

            avg_val_loss = val_running_loss / len(val_loader)
            self.logger.info(f"[SimCLR - Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}")
            # Log validation loss to W&B
            if self.wandb_logger.is_active:
                self.wandb_logger.log(
                    {"val/loss": avg_val_loss},
                    step=epoch + 1
                )
        self.model.train()
        return avg_val_loss


    def _train_cola(
        self,
        train_loader,
        optimizer,
        epochs: int,
        start_epoch: int = 0,
        val_loader: Optional[DataLoader] = None,
        logger_loader: Optional[DataLoader] = None,
        use_embedding_logger: bool = False,
    ) -> None:
        """Train the COLA model with embedding logging via COLABackbone."""
        if self.transformation is None:
            self.logger.error("Transformation not given!")
            raise ValueError("Transformation not given!")

        self.model.train()

        # === Step 0: Log pre-training embeddings ===
        if use_embedding_logger:
            assert logger_loader is not None, "logger_loader must be provided when use_embedding_logger=True"
            embedding_log_dir = os.path.join(self.checkpoint_path, "embedding_logs")
            embedding_logger = EmbeddingLogger(
                log_dir=embedding_log_dir,
                method_name=self.method,
                reduce_method="tsne",
                log_interval=1,
            )
            self.logger.info(f"Embedding logger initialized at {embedding_log_dir}")

            self.logger.info("[COLA - Step 0] Logging pre-training embeddings...")
            backbone = COLABackbone(self.model).to(self.device)
            backbone.eval()

            all_embeddings, all_labels = [], []
            with torch.no_grad():
                for batch in tqdm(logger_loader, desc="EmbeddingLogger Step 0"):
                    audio = batch["audio"].to(self.device)
                    labels = batch["label"].to(self.device)
                    embeddings = backbone(audio)  # Get embeddings via COLABackbone
                    all_embeddings.append(embeddings)
                    all_labels.append(labels)

            embeddings = torch.cat(all_embeddings, dim=0)
            labels = torch.cat(all_labels, dim=0)
            embedding_logger.log_step(step=0, embeddings=embeddings, labels=labels)
            self.logger.info("[COLA - Step 0] Pre-training embeddings logged.")
            self.model.train()

        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(self.model)

        # === Training Loop ===
        for epoch in range(start_epoch, epochs):
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"COLA Epoch {epoch+1}/{epochs}")

            for batch_idx, batch in enumerate(pbar):
                audio = batch["audio"].to(self.device)
                view0, view1 = self.transformation(audio)

                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    out0, out1 = self.model(view0, view1)
                    loss = self.loss(out0, out1)

                optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                running_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

                global_step = epoch * len(train_loader) + batch_idx
                if self.wandb_logger.is_active:
                    self.wandb_logger.log({"train/batch_loss": loss.item()}, step=global_step)

            avg_loss = running_loss / len(train_loader)
            self.logger.info(f"[COLA - Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

            if self.wandb_logger.is_active:
                self.wandb_logger.log({"train/epoch_loss": avg_loss}, step=epoch + 1)

            # === Embedding logger ===
            if use_embedding_logger:
                self.logger.info(f"[COLA - Epoch {epoch+1}] Logging embeddings...")
                backbone = COLABackbone(self.model).to(self.device)
                backbone.eval()

                all_embeddings, all_labels = [], []
                with torch.no_grad():
                    for batch in tqdm(logger_loader, desc=f"EmbeddingLogger Epoch {epoch+1}"):
                        audio = batch["audio"].to(self.device)
                        labels = batch["label"].to(self.device)
                        embeddings = backbone(audio)  # Get embeddings via COLABackbone
                        all_embeddings.append(embeddings)
                        all_labels.append(labels)

                embeddings = torch.cat(all_embeddings, dim=0)
                labels = torch.cat(all_labels, dim=0)
                embedding_logger.log_step(step=epoch + 1, embeddings=embeddings, labels=labels)
                self.logger.info(f"[COLA - Epoch {epoch+1}] Embeddings logged.")
                self.model.train()

            # === Validation ===
            if val_loader:
                avg_val_loss = self._validate_cola(val_loader, epoch)

            if hasattr(self, "_optuna_trial"):
                metric = avg_val_loss if val_loader else avg_loss
                self._optuna_trial.report(metric, epoch)
                if self._optuna_trial.should_prune():
                    raise optuna.TrialPruned()

            if (epoch + 1) % self.checkpoint_interval == 0 and not hasattr(self, "_optuna_trial"):
                model_path = os.path.join(
                    self.checkpoint_path,
                    f"{self.method}_model_{self.timestamp}_epoch{epoch+1}.pth",
                )
                torch.save(self.model.state_dict(), model_path)
                self.logger.info(f"Model checkpoint saved: {model_path}")

                if self.wandb_logger.is_active:
                    self.wandb_logger.save_artifact(
                        model_path,
                        name=f"{self.method}-model-epoch-{epoch+1}",
                        type="model",
                        metadata={"epoch": epoch+1, "loss": avg_loss}
                    )

        final_path = os.path.join(
            self.checkpoint_path,
            f"{self.method}_model_{self.timestamp}_final.pth",
        )
        torch.save(self.model.state_dict(), final_path)
        self.logger.info(f"Final model checkpoint saved: {final_path}")

        if self.wandb_logger.is_active:
            self.wandb_logger.save_artifact(
                final_path,
                name=f"{self.method}-model-final",
                type="model",
                metadata={"epochs_trained": epochs, "final_loss": avg_loss}
            )

        # === Final animated embedding plot ===
        if use_embedding_logger:
            self.logger.info("Generating final embedding animation...")
            animation_path = embedding_logger.plot_all()
            self.logger.info(f"Embedding animation saved at: {animation_path}")

            if self.wandb_logger.is_active:
                import wandb
                self.wandb_logger.log(
                    {"embedding_animation": wandb.Html(animation_path)},
                    step=max(embedding_logger.steps) if embedding_logger.steps else epochs
                )
                self.logger.info("Embedding animation logged to Weights & Biases.")

        self.logger.info("COLA training complete.")


    def _validate_cola(self, val_loader: DataLoader, epoch: int):

        self.model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Validation COLA Epoch {epoch+1}")
            for batch in pbar:
                audio = batch['audio'].to(self.device)
                view0, view1 = self.transformation(audio)
                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    out0, out1 = self.model(view0, view1)
                    loss = self.loss(out0, out1)

                val_running_loss += loss.item()

            avg_val_loss = val_running_loss / len(val_loader)
            self.logger.info(f"[COLA - Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}")
            # Log validation loss to W&B
            if self.wandb_logger.is_active:
                self.wandb_logger.log(
                    {"val/loss": avg_val_loss},
                    step=epoch + 1
                )
        self.model.train()
        return avg_val_loss

    def _train_hubert(
        self,
        train_loader_for_training: DataLoader,
        train_loader_full_dataset: DataLoader,
        optimizer,
        epochs: int,
        start_epoch: int = 0,
        start_iteration: int = 0,
        val_loader: Optional[DataLoader] = None,
        num_hubert_iterations: int = 5,
        pseudo_label_sample_ratio: float = 0.1,  # Kept in signature but no longer used
        logger_loader: Optional[DataLoader] = None,
        use_embedding_logger: bool = False,
        **kwargs,
    ):
        """Train the HuBERT model with embedding logging via HuBERTBackbone."""
        transformer_layer = kwargs.get(
            "transformer_layer", getattr(self.model, "extractor_layer", None)
        )
        if transformer_layer is None:
            self.logger.warning("No 'transformer_layer' specified for HuBERT.")

        # === Initialize embedding logger and log initial embeddings ===
        if use_embedding_logger:
            assert logger_loader is not None, "logger_loader must be provided when use_embedding_logger=True"
            embedding_log_dir = os.path.join(self.checkpoint_path, "embedding_logs")
            embedding_logger = EmbeddingLogger(
                log_dir=embedding_log_dir,
                method_name=self.method,
                reduce_method="tsne",
                log_interval=1,
            )
            self.logger.info(f"Embedding logger initialized at {embedding_log_dir}")

            self.logger.info("[HuBERT - Step 0] Logging pre-training embeddings...")
            backbone = HuBERTBackbone(self.model).to(self.device)
            backbone.eval()

            all_embeddings, all_labels = [], []
            with torch.no_grad():
                for batch in tqdm(logger_loader, desc="EmbeddingLogger Step 0"):
                    audio = batch["audio"].to(self.device)
                    labels = batch["label"].to(self.device)
                    embeddings = backbone(audio)
                    all_embeddings.append(embeddings)
                    all_labels.append(labels)

            embeddings = torch.cat(all_embeddings, dim=0)
            labels = torch.cat(all_labels, dim=0)
            embedding_logger.log_step(step=0, embeddings=embeddings, labels=labels)
            self.logger.info("[HuBERT - Step 0] Pre-training embeddings logged.")

        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(self.model)

        # === Iteration Loop ===
        for iteration in range(start_iteration, num_hubert_iterations):
            self.logger.info(f"--- Starting HuBERT Iteration {iteration + 1}/{num_hubert_iterations} ---")
            if self.wandb_logger.is_active:
                self.wandb_logger.log({"hubert_iteration": iteration + 1})

            iteration_pseudo_labels_path = os.path.join(
                self.pseudo_label_generator.save_dir, f"pseudo_labels_iter_{iteration}.npy"
            )

            # === Pseudo-label generation ===
            if os.path.exists(iteration_pseudo_labels_path):
                self.logger.info(f"Loading existing pseudo-labels from {iteration_pseudo_labels_path}")
                pseudo_labels_dict = np.load(iteration_pseudo_labels_path, allow_pickle=True).item()
            else:
                self.logger.info("Generating pseudo-labels for ALL samples (HuBERT).")
                dataloader_for_clustering = train_loader_full_dataset

                pseudo_labels_dict = self.pseudo_label_generator.generate_pseudo_labels(
                    dataloader=dataloader_for_clustering,
                    model=self.model,
                    is_mfcc=(iteration == 0),
                    transformer_layer=transformer_layer,
                    device=self.device,
                )
                np.save(iteration_pseudo_labels_path, pseudo_labels_dict)
                self.logger.info(f"Saved pseudo-labels for iteration {iteration + 1}.")

            # === Adjust pseudo-labels to match dataset ===
            all_dataset_indices = set(range(len(train_loader_for_training.dataset)))
            all_pseudo_indices = set(pseudo_labels_dict.keys())
            
            extra = all_pseudo_indices - all_dataset_indices
            missing = all_dataset_indices - all_pseudo_indices
            
            if extra:
                self.logger.warning(f"Removing {len(extra)} extra pseudo-labels: {sorted(list(extra))[:10]}...")
                for idx in extra:
                    pseudo_labels_dict.pop(idx, None)

            if missing:
                self.logger.warning(f"Filling {len(missing)} missing pseudo-labels with zeros: {sorted(list(missing))[:10]}...")
                for idx in missing:
                    pseudo_labels_dict[idx] = np.zeros(self.model.num_clusters, dtype=np.int64)

            self.logger.info(f"Adjusted pseudo-labels to match dataset size ({len(pseudo_labels_dict)}).")

            train_loader_for_training.dataset.set_pseudo_labels(pseudo_labels_dict)
            self.logger.info("Updated training dataset with pseudo-labels.")

            self.logger.info(f"Starting model training for HuBERT Iteration {iteration + 1} for {epochs} epochs.")
            self.model.train()
            current_iter_start_epoch = start_epoch if iteration == start_iteration else 0

            # === Epoch Loop ===
            for epoch in range(current_iter_start_epoch, epochs):
                running_loss = 0.0
                pbar = tqdm(train_loader_for_training, desc=f"HuBERT Iter {iteration+1}, Epoch {epoch+1}/{epochs}")

                for batch_idx, batch in enumerate(pbar):
                    audio = batch["audio"].to(self.device)
                    lengths = batch["length"].to(self.device)
                    pseudo_labels = batch["pseudo_labels"].to(self.device)

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                        logits, mask_indices, lengths, _ = self.model(audio, lengths)

                        # === NaN loss guard ===
                        if mask_indices.sum() == 0:
                            self.logger.warning(
                                f"[Iteration {iteration+1}, Epoch {epoch+1}, Batch {batch_idx}] No masked positions. Skipping."
                            )
                            continue

                        masked_targets = pseudo_labels[mask_indices]

                        # Target validity check
                        if (masked_targets < 0).any() or (masked_targets >= self.model.num_clusters).any():
                            raise ValueError(f"Invalid pseudo-label values found in batch {batch_idx}.")

                        loss = self.loss(logits, masked_targets)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()

                    running_loss += loss.item()
                    pbar.set_postfix({"loss": loss.item()})

                    global_step = epoch * len(train_loader_for_training) + batch_idx
                    if self.wandb_logger.is_active:
                        self.wandb_logger.log(
                            {
                                "train/batch_loss": loss.item(),
                                f"train/iter_{iteration+1}_batch_loss": loss.item(),
                            },
                            step=global_step
                        )

                avg_loss = running_loss / max(1, len(train_loader_for_training))
                self.logger.info(f"[HuBERT Iter {iteration+1} - Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

                if self.wandb_logger.is_active:
                    self.wandb_logger.log(
                        {
                            "train/epoch_loss": avg_loss,
                            f"train/iter_{iteration+1}_epoch_loss": avg_loss,
                        },
                        step=epoch + 1
                    )

                # === Embedding logger per epoch ===
                if use_embedding_logger:
                    self.logger.info(f"[HuBERT Iter {iteration+1} - Epoch {epoch+1}] Logging embeddings...")
                    backbone = HuBERTBackbone(self.model).to(self.device)
                    backbone.eval()

                    all_embeddings, all_labels = [], []
                    with torch.no_grad():
                        for batch in tqdm(logger_loader, desc=f"EmbeddingLogger Iter {iteration+1} Epoch {epoch+1}"):
                            audio = batch["audio"].to(self.device)
                            labels = batch["label"].to(self.device)
                            embeddings = backbone(audio)
                            all_embeddings.append(embeddings)
                            all_labels.append(labels)

                    embeddings = torch.cat(all_embeddings, dim=0)
                    labels = torch.cat(all_labels, dim=0)
                    embedding_logger.log_step(step=epoch + 1, embeddings=embeddings, labels=labels)
                    self.logger.info(f"[HuBERT Iter {iteration+1} - Epoch {epoch+1}] Embeddings logged.")
                    self.model.train()

                # === Validation and Checkpoint ===
                if (epoch + 1) % self.checkpoint_interval == 0 and not hasattr(self, "_optuna_trial"):
                    model_path = os.path.join(
                        self.checkpoint_path,
                        f"{self.method}_iter{iteration+1}_model_{self.timestamp}_epoch{epoch+1}.pth",
                    )
                    torch.save(self.model.state_dict(), model_path)
                    self.logger.info(f"Model checkpoint saved: {model_path}")

                    if self.wandb_logger.is_active:
                        self.wandb_logger.save_artifact(
                            model_path,
                            name=f"{self.method}-iter{iteration+1}-model-epoch-{epoch+1}",
                            type="model",
                            metadata={"iteration": iteration+1, "epoch": epoch+1, "loss": avg_loss}
                        )

                if val_loader:
                    avg_val_loss = self._validate_hubert(val_loader, iteration, epoch)

                if hasattr(self, "_optuna_trial"):
                    metric = avg_val_loss if val_loader else avg_loss
                    self._optuna_trial.report(metric, epoch)
                    if self._optuna_trial.should_prune():
                        raise optuna.TrialPruned()

            # === Final model saving for iteration ===
            final_model_path = os.path.join(
                self.checkpoint_path,
                f"{self.method}_iter{iteration+1}_final_model_{self.timestamp}.pth",
            )
            torch.save(self.model.state_dict(), final_model_path)
            self.logger.info(f"Final model for HuBERT Iteration {iteration+1} saved: {final_model_path}")

            if self.wandb_logger.is_active:
                self.wandb_logger.save_artifact(
                    final_model_path,
                    name=f"{self.method}-iter{iteration+1}-final-model",
                    type="model",
                    metadata={"iteration": iteration+1, "epochs_trained": epochs, "final_loss": avg_loss}
                )

            # === Final animation logging per iteration ===
            if use_embedding_logger:
                self.logger.info(f"Generating embedding animation for HuBERT Iteration {iteration+1}...")
                animation_path = embedding_logger.plot_all()
                self.logger.info(f"Embedding animation saved at: {animation_path}")

                if self.wandb_logger.is_active:
                    import wandb
                    self.wandb_logger.log(
                        {f"embedding_animation/iter_{iteration+1}": wandb.Html(animation_path)},
                        step=max(embedding_logger.steps) if embedding_logger.steps else epochs
                    )
                    self.logger.info("Embedding animation logged to Weights & Biases.")

        self.logger.info("HuBERT training complete across all specified iterations.")


    def _validate_hubert(
        self, val_loader: DataLoader, iteration: int, epoch: int
    ) -> None:

        self.model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(
                val_loader,
                desc=f"Validation HuBERT Iter {iteration+1}, Epoch {epoch+1}",
            )
            for batch in pbar:
                audio = batch['audio'].to(self.device)
                pseudo_labels = batch["pseudo_labels"].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    logits, mask_indices, _, _ = self.model(audio)
                    loss = self.loss(logits, pseudo_labels, mask_indices)

                val_running_loss += loss.item()

            avg_val_loss = val_running_loss / len(val_loader)
            self.logger.info(
                f"[HuBERT Iter {iteration+1} - Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}"
            )
            # Log validation loss to W&B
            if self.wandb_logger.is_active:
                self.wandb_logger.log(
                    {"val/loss": avg_val_loss,
                     f"val/iter_{iteration+1}_loss": avg_val_loss},
                    step=epoch + 1
                )
        self.model.train()
        return avg_val_loss


    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 16,
        start_epoch: int = 0,
        epochs: int = 100,
        start_iteration: int = 0,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        optimizer: str = "adamw",
        use_hpo: bool = False,
        n_trials: int = 20,
        tuning_epochs: int = 5, 
        use_embedding_logger: bool = False,
        logger_loader: Optional[DataLoader] = None,  # NEW
        **kwargs,
    ) -> None:
        """
        Main training loop to train the model using the given dataset and hyperparameters.

        Args:
            train_dataset (Dataset): Dataset object for training.
            batch_size (int, optional): Mini-batch size. Defaults to 16.
            start_epoch (int, optional): Epoch to resume training from. Defaults to 0.
            epochs (int, optional): Total number of epochs. Defaults to 100.
            start_iteration (int, optional): Iteration to resume HuBERT training from. Defaults to 0.
            lr (float, optional): Learning rate. Defaults to 1e-4.
            weight_decay (float, optional): Weight decay (L2 regularization). Defaults to 1e-2.
            optimizer (str, optional): Optimizer to use ('adam', 'sgd', or 'adamw'). Defaults to 'adamw'.
            **kwargs: Additional keyword arguments passed to optimizer or loss, or HuBERT specific.
        """
        # Initialize W&B run at the very beginning of the main train method
        # This ensures console output is captured from the start and the run is properly set up.
        if not hasattr(self, "_optuna_trial"):
            self.wandb_logger.init_run()
        else:
            self.wandb_logger.mode = 'disabled'

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



        # Auto hyperparameter tuning
        if use_hpo:
            self.logger.info("🧪 Running Optuna for hyperparameter tuning...")
            
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
            
            self.wandb_logger.log({
                "hpo/best_lr": lr,
                "hpo/best_batch_size": batch_size,
                "hpo/best_weight_decay": weight_decay,
                "hpo/best_optimizer": optimizer,
                **{f"hpo/{k}": v for k, v in kwargs.items()}
            })
            self.logger.info("📡 Best hyperparameters logged to W&B.")

        match optimizer.lower():
            case "adam":
                optimizer = torch.optim.Adam(
                    list(self.model.parameters()),
                    lr=lr,
                    weight_decay=weight_decay,
                )
            case "sgd":
                optimizer = torch.optim.SGD(
                    list(self.model.parameters()),
                    lr=lr,
                    weight_decay=weight_decay,
                )
            case "adamw":
                optimizer = torch.optim.AdamW(
                    list(self.model.parameters()),
                    lr=lr,
                    weight_decay=weight_decay,
                )
            case _:
                self.logger.error(f"Unsupported Optimizer: {optimizer}")
                raise ValueError(f"Optimizer {optimizer} not supported")

        if self.reload_checkpoint:
            start_epoch = self._reload_latest_checkpoint()

        if self.method == "wav2vec2":

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self._data_loader_safe_collate,
            )
            self.logger.info(f"Training dataset loaded with {len(train_dataset)} samples.")
            
            first_train_batch = next(iter(train_loader))
            if "audio" not in first_train_batch or "length" not in first_train_batch:
                self.logger.warning(
                    "[Dataset Check] Your dataset should return both 'audio' and 'length' keys. "
                    "Currently missing: "
                    + ", ".join(k for k in ["audio", "length"] if k not in first_train_batch)
                )

            val_loader = None
            if val_dataset:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    collate_fn=self._data_loader_safe_collate,
                )
                first_val_batch = next(iter(val_loader))
                if "audio" not in first_val_batch or "length" not in first_val_batch:
                    self.logger.warning("[Dataset Check] val_loader should return both 'audio' and 'length'.")

            self._train_wav2vec2(
                train_loader,
                optimizer,
                epochs,
                start_epoch,
                val_loader,
                use_embedding_logger=use_embedding_logger,
                logger_loader=logger_loader
            )

        elif self.method == "hubert":

            first_sample = train_dataset[0]
            if "audio" not in first_sample or "length" not in first_sample:
                self.logger.warning(
                    "[Dataset Check] Your dataset should return both 'audio' and 'length' keys. "
                    "Currently missing: "
                    + ", ".join(k for k in ["audio", "length"] if k not in first_sample)
                )

            wrapped_train_dataset = HuBERTWrapperDataset(
                train_dataset,
                feature_extractor=self.model.feature_extractor,
                logger=self.logger
            )

            train_loader_for_pseudo_label_gen = DataLoader(
                wrapped_train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self._data_loader_safe_collate,
            )
            

            train_loader_for_training = DataLoader(
                wrapped_train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self._data_loader_safe_collate,

            )

            val_loader = None
            if val_dataset:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    collate_fn=self._data_loader_safe_collate,
                )


            self._train_hubert(
                train_loader_for_training=train_loader_for_training,
                train_loader_full_dataset=train_loader_for_pseudo_label_gen,
                optimizer=optimizer,
                epochs=epochs,
                start_epoch=start_epoch,
                val_loader=val_loader,
                start_iteration=start_iteration,
                num_hubert_iterations=kwargs.get(
                    "num_hubert_iterations", getattr(self.model, "config", {}).get("max_iterations", 2)
                ),
                transformer_layer=kwargs.get(
                    "transformer_layer", getattr(self.model, "config", {}).get("extractor_layer", None)
                ),
                pseudo_label_sample_ratio=kwargs.get(
                    "pseudo_label_sample_ratio",
                    getattr(self.model, "config", {}).get("pseudo_label_sample_ratio", 0.1),
                ),
                use_embedding_logger= use_embedding_logger,
                logger_loader=logger_loader,
            )

        elif self.method == "simclr": # Added simclr training
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self._data_loader_safe_collate,

            )
            val_loader = None
            if val_dataset:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    collate_fn=self._data_loader_safe_collate,

                )
            self._train_simclr(
                train_loader,
                optimizer,
                epochs,
                start_epoch,
                val_loader,
                use_embedding_logger=use_embedding_logger,
                logger_loader=logger_loader
            )

        elif self.method == "cola": # Added cola training
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self._data_loader_safe_collate,

            )
            val_loader = None
            if val_dataset:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    collate_fn=self._data_loader_safe_collate,

                )
            self._train_cola(
                train_loader,
                optimizer,
                epochs,
                start_epoch,
                val_loader,
                use_embedding_logger=use_embedding_logger,
                logger_loader=logger_loader,
            )

        else:
            raise NotImplementedError(
                f"Training not implemented for method: {self.method}"
            )

        training_mode = "Main" if not hasattr(self, "_optuna_trial") else "HPO"
        if self.wandb_logger.is_active:
            self.wandb_logger.finish_run()
            self.logger.info(f"{training_mode} training process completed and W&B run finalized.")
        else:
            self.logger.info(f"{training_mode} training process completed.")


    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Loads a model checkpoint from the given path.
        Assumes self.model is already initialized and matches the checkpoint's state_dict.

        Args:
            checkpoint_path (str): Path to the checkpoint file (.pth).
        """
        if self.model is None:
            self.logger.error("Model must be initialized before loading a checkpoint.")
            raise RuntimeError("Model must be initialized before loading a checkpoint.")
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.logger.info(f"Checkpoint loaded from: {checkpoint_path}")


    def _evaluate_wav2vec2(
        self,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        num_classes: int,
        batch_size: int = 64,
        lr: float = 1e-3,
        epochs: int = 10,
        freeze_backbone: bool = True,
        **kwargs
    ):
        """
        Evaluation for Wav2Vec2 using linear probing or fine-tuning.

        Args:
            train_dataset (Dataset): Supervised training dataset with (waveform, label) pairs.
            test_dataset (Dataset): Supervised test dataset for evaluation.
            num_classes (int): Number of output classes.
            batch_size (int): Evaluation batch size.
            lr (float): Learning rate.
            epochs (int): Max number of epochs.
            freeze_backbone (bool): Whether to freeze the backbone during evaluation.
        """


        # === Instantiate backbone and classifier ===
        backbone = Wav2Vec2Backbone(pretrained_model=self.model)
        feature_size = self.model.model_config["encoder_embed_dim"]

        model = EvaluateNet(
            backbone=backbone,
            feature_size=feature_size,
            num_classes=num_classes,
            is_linear=freeze_backbone,
        ).to(self.device)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # ✅ Watch the classifier model
        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(model)

        # === Training loop ===
        model.train()
        for epoch in range(epochs):
            for waveforms, labels in train_loader:
                waveforms = waveforms.to(self.device)
                labels = labels.to(self.device)

                logits = model(waveforms)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.logger.info(f"[Wav2Vec2 Eval] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

            if self.wandb_logger.is_active:
                self.wandb_logger.log({
                    "wav2vec2/train_loss": loss.item(),
                    "wav2vec2/epoch": epoch + 1,
                    "wav2vec2/lr": optimizer.param_groups[0]["lr"]
                }, step=epoch + 1)

        # === Evaluation loop ===
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for waveforms, labels in test_loader:
                waveforms = waveforms.to(self.device)
                labels = labels.to(self.device)

                logits = model(waveforms)
                preds = torch.argmax(logits, dim=1)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        self.logger.info("\n📊 [Wav2Vec2 Evaluation Report]:\n" +
                         classification_report(all_labels.numpy(), all_preds.numpy(), digits=4))

        report = classification_report(all_labels.numpy(), all_preds.numpy(), digits=4, output_dict=True)

        if self.wandb_logger.is_active:
            self.wandb_logger.log({
                "wav2vec2/test_accuracy": report["accuracy"],
                "wav2vec2/test_macro_avg_f1": report["macro avg"]["f1-score"],
                "wav2vec2/test_macro_avg_precision": report["macro avg"]["precision"],
                "wav2vec2/test_macro_avg_recall": report["macro avg"]["recall"]
            })


    def _evaluate_simclr(
        self,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        num_classes: int,
        batch_size: int = 64,
        lr: float = 1e-3,
        epochs: int = 10,
        freeze_backbone: bool = True,
        **kwargs
    ):
        """
        Evaluation for SimCLR Speech using linear probing or fine-tuning.

        Args:
            train_dataset (Dataset): Dataset for training the downstream classifier.
            test_dataset (Dataset): Dataset for evaluation after training.
            num_classes (int): Number of output classes.
            batch_size (int): Evaluation batch size.
            lr (float): Learning rate.
            epochs (int): Max number of epochs.
            freeze_backbone (bool): Whether to freeze the backbone during evaluation.
        """

        model = self.model
        backbone = SimCLRBackbone(model)
        feature_size = model.backbone.embed_dim

        classifier = EvaluateNet(
            backbone=backbone,
            feature_size=feature_size,
            num_classes=num_classes,
            is_linear=freeze_backbone,
        ).to(self.device)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # ✅ Watch the classifier model
        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(classifier)

        # === Training loop ===
        classifier.train()
        for epoch in range(epochs):
            for wavs, labels in train_loader:
                wavs, labels = wavs.to(self.device), labels.to(self.device)
                logits = classifier(wavs)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.logger.info(f"[SimCLR Eval] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

            if self.wandb_logger.is_active:
                self.wandb_logger.log({
                    "simclr/train_loss": loss.item(),
                    "simclr/epoch": epoch + 1,
                    "simclr/lr": optimizer.param_groups[0]["lr"]
                }, step=epoch + 1)

        # === Evaluation loop ===
        classifier.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for wavs, labels in test_loader:
                wavs, labels = wavs.to(self.device), labels.to(self.device)
                logits = classifier(wavs)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        self.logger.info("\n📊 [SimCLR Speech Evaluation Report]:\n" +
                         classification_report(all_labels.numpy(), all_preds.numpy(), digits=4))

        report = classification_report(all_labels.numpy(), all_preds.numpy(), digits=4, output_dict=True)

        if self.wandb_logger.is_active:
            self.wandb_logger.log({
                "simclr/test_accuracy": report["accuracy"],
                "simclr/test_macro_avg_f1": report["macro avg"]["f1-score"],
                "simclr/test_macro_avg_precision": report["macro avg"]["precision"],
                "simclr/test_macro_avg_recall": report["macro avg"]["recall"]
            })


    def _evaluate_hubert(
        self,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        num_classes: int,
        batch_size: int = 64,
        lr: float = 1e-3,
        epochs: int = 10,
        freeze_backbone: bool = True,
        **kwargs
    ):
        """
        Evaluation for HuBERT using linear probing or fine-tuning.

        Args:
            train_dataset (Dataset): Supervised training dataset with (waveform, label) pairs.
            test_dataset (Dataset): Supervised test dataset for evaluation.
            num_classes (int): Number of output classes.
            batch_size (int): Evaluation batch size.
            lr (float): Learning rate.
            epochs (int): Max number of epochs.
            freeze_backbone (bool): Whether to freeze the backbone during evaluation.
        """

        model = self.model
        feature_size = model.config["encoder_embed_dim"]

        backbone = HuBERTBackbone(model)

        classifier = EvaluateNet(
            backbone=backbone,
            feature_size=feature_size,
            num_classes=num_classes,
            is_linear=freeze_backbone,
        ).to(self.device)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # ✅ Watch the classifier model
        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(classifier)

        # === Training loop ===
        classifier.train()
        for epoch in range(epochs):
            for waveforms, labels in train_loader:
                waveforms = waveforms.to(self.device)
                labels = labels.to(self.device)

                logits = classifier(waveforms)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.logger.info(f"[HuBERT Eval] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

            if self.wandb_logger.is_active:
                self.wandb_logger.log({
                    "hubert/train_loss": loss.item(),
                    "hubert/epoch": epoch + 1,
                    "hubert/lr": optimizer.param_groups[0]["lr"]
                }, step=epoch + 1)

        # === Evaluation loop ===
        classifier.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for waveforms, labels in test_loader:
                waveforms = waveforms.to(self.device)
                labels = labels.to(self.device)

                logits = classifier(waveforms)
                preds = torch.argmax(logits, dim=1)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        report = classification_report(all_labels.numpy(), all_preds.numpy(), digits=4, output_dict=True)

        self.logger.info("\n📊 [HuBERT Evaluation Report]:\n" +
                         classification_report(all_labels.numpy(), all_preds.numpy(), digits=4))

        if self.wandb_logger.is_active:
            self.wandb_logger.log({
                "hubert/test_accuracy": report["accuracy"],
                "hubert/test_macro_avg_f1": report["macro avg"]["f1-score"],
                "hubert/test_macro_avg_precision": report["macro avg"]["precision"],
                "hubert/test_macro_avg_recall": report["macro avg"]["recall"]
            })


    def _evaluate_cola(
        self,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        num_classes: int,
        batch_size: int = 64,
        lr: float = 1e-3,
        epochs: int = 10,
        freeze_backbone: bool = True,
        **kwargs
    ):
        """
        Evaluation for COLA using linear probing or fine-tuning.

        Args:
            train_dataset (Dataset): Dataset for training the downstream classifier.
            test_dataset (Dataset): Dataset for evaluation after training.
            num_classes (int): Number of output classes.
            batch_size (int): Evaluation batch size.
            lr (float): Learning rate.
            epochs (int): Max number of epochs.
            freeze_backbone (bool): Whether to freeze the backbone during evaluation.
        """

        backbone = COLABackbone(self.model)
        feature_size = self.model.feature_size

        classifier = EvaluateNet(
            backbone=backbone,
            feature_size=feature_size,
            num_classes=num_classes,
            is_linear=freeze_backbone,
        ).to(self.device)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # ✅ Watch the classifier model
        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(classifier)

        # === Training ===
        classifier.train()
        for epoch in range(epochs):
            for batch in tqdm(train_loader, desc=f"[COLA Downstream Training] Epoch {epoch+1}"):
                audio = batch["audio"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = classifier(audio)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.logger.info(f"[COLA Eval] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

            if self.wandb_logger.is_active:
                self.wandb_logger.log({
                    "cola/train_loss": loss.item(),
                    "cola/epoch": epoch + 1,
                    "cola/lr": optimizer.param_groups[0]["lr"]
                }, step=epoch + 1)

        # === Evaluation ===
        classifier.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="[COLA Downstream Evaluation]"):

                audio = batch["audio"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = classifier(audio)
                preds = torch.argmax(logits, dim=1)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        report = classification_report(all_labels.numpy(), all_preds.numpy(), digits=4, output_dict=True)

        self.logger.info("\n📊 [COLA Evaluation Report]:\n" +
                         classification_report(all_labels.numpy(), all_preds.numpy(), digits=4))

        if self.wandb_logger.is_active:
            self.wandb_logger.log({
                "cola/test_accuracy": report["accuracy"],
                "cola/test_macro_avg_f1": report["macro avg"]["f1-score"],
                "cola/test_macro_avg_precision": report["macro avg"]["precision"],
                "cola/test_macro_avg_recall": report["macro avg"]["recall"]
            })


    def evaluate(
        self,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        num_classes: int,
        batch_size: int = 64,
        lr: float = 1e-3,
        epochs: int = 10,
        freeze_backbone: bool = True,
        **kwargs
    ):
        """
        Evaluate the current model using the correct evaluation method.
        """
        if not self.wandb_logger.is_active:
            self.wandb_logger.init_run('Evaluation')

        self.logger.info(f"🔍 Starting evaluation for method: {self.method}")

        match self.method:
            case "cola":
                self._evaluate_cola(train_dataset, test_dataset, num_classes, batch_size, lr, epochs, freeze_backbone, **kwargs)
            case "hubert":
                self._evaluate_hubert(train_dataset, test_dataset, num_classes, batch_size, lr, epochs, freeze_backbone, **kwargs)
            case "simclr":
                self._evaluate_simclr(train_dataset, test_dataset, num_classes, batch_size, lr, epochs, freeze_backbone, **kwargs)
            case "wav2vec2":
                self._evaluate_wav2vec2(train_dataset, test_dataset, num_classes, batch_size, lr, epochs, freeze_backbone, **kwargs)
            case _:
                raise ValueError(f"❌ Unknown method '{self.method}' for evaluation.")

        self.logger.info(f"✅ Evaluation for '{self.method}' completed.")
        if self.wandb_logger.is_active:
            self.wandb_logger.log({f"{self.method}/status": "evaluation_complete"})
            self.wandb_logger.finish_run()

    def _reload_latest_checkpoint(self) -> int:
        """
        Reloads the most recent model checkpoint from the checkpoint directory.

        Returns:
            int: The epoch number from which training should resume.

        Raises:
            ValueError: If no valid checkpoint or epoch information is found.
        """
        checkpoints = os.listdir(self.checkpoint_path)
        method_prefix = self.method + "_model_"
        filtered_checkpoints = [
            ckpt
            for ckpt in checkpoints
            if ckpt.endswith(".pth") and ckpt.startswith(method_prefix)
        ]

        if not filtered_checkpoints:
            self.logger.warning(
                f"No valid checkpoints found for method '{self.method}' in {self.checkpoint_path}. Starting from scratch."
            )
            return 0

        sorted_checkpoints = sorted(
            [os.path.join(self.checkpoint_path, ckpt) for ckpt in filtered_checkpoints],
            key=os.path.getmtime,
        )

        latest_ckpt = sorted_checkpoints[-1]
        self.load_checkpoint(latest_ckpt)

        match = re.search(r"epoch(\d+)", latest_ckpt)
        if match:
            epoch = int(match.group(1))
            self.logger.info(f"Reloaded checkpoint from epoch {epoch + 1}")
        else:
            self.logger.warning(
                f"No epoch number found in the checkpoint name '{latest_ckpt}'. Resuming from epoch 1."
            )
            epoch = 0

        return epoch



    def _data_loader_safe_collate(self, batch):
        batch = [item for item in batch if item is not None]
        return default_collate(batch) if batch else None


    def __del__(self):
        """
        Destructor for the Trainer class.
        Closes the TensorBoard writer if it exists.
        """
        if hasattr(self, "writer"):
            self.writer.close()



