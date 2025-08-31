import os
import re
import torch
import numpy as np
from torch import nn
from tqdm.auto import (
    tqdm,
)
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.dataloader import default_collate

import logging

from typing import Optional, Dict, Any
from sklearn.metrics import classification_report
import wandb
import optuna

from jiwer import wer
from jiwer import wer as jiwer_wer

import editdistance
from editdistance import eval as edit_distance
from torchmetrics.functional import word_error_rate

from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler



from MK_SSL.audio.models.utils.registry import get_method
from MK_SSL.audio.models.utils.datasets import HuBERTWrapperDataset

from MK_SSL.audio.models.modules.tools import PseudoLabelGenerator

from MK_SSL.audio.models.modules.cola_backbone import COLABackbone
from MK_SSL.audio.models.modules.wav2vec2_backbone import Wav2Vec2Backbone
from MK_SSL.audio.models.modules.hubert_backbone import HuBERTBackbone
from MK_SSL.audio.models.modules.simclr_backbone import SimCLRBackbone
from MK_SSL.audio.models.modules.eat_backbone import EATBackbone

from MK_SSL.audio.models.modules.backbones import ViTAudioEncoder


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
        num_workers: Optional[int] = None,

        
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
        self.num_workers = os.cpu_count() if num_workers is None else num_workers

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
            
            epoch_step = (epoch + 1) * len(train_loader)        
            if self.wandb_logger.is_active:
                self.wandb_logger.log({"train/epoch_loss": avg_loss, "epoch": epoch + 1}, step=epoch_step,)

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
                avg_val_loss = self._validate_wav2vec2(val_loader, epoch, epoch_step)

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
                    {"media/embedding_animation": wandb.Html(animation_path)},
                    step=max(embedding_logger.steps) if embedding_logger.steps else epochs
                )
                self.logger.info("Embedding animation logged to Weights & Biases.")

        self.logger.info("Wav2Vec2 training complete.")


    def _validate_wav2vec2(self, val_loader: DataLoader, epoch: int, epoch_step) -> float:
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
                    step=epoch_step  # Use epoch number as step for epoch-level metrics
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


            epoch_step = (epoch + 1) * len(train_loader)        
    
            if self.wandb_logger.is_active:
                self.wandb_logger.log({"train/epoch_loss": avg_loss, "epoch": epoch + 1}, step=epoch_step)

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
                avg_val_loss = self._validate_simclr(val_loader, epoch, epoch_step)

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
                    {"media/embedding_animation": wandb.Html(animation_path)},
                    step=max(embedding_logger.steps) if embedding_logger.steps else epochs
                )
                self.logger.info("Embedding animation logged to Weights & Biases.")

        self.logger.info("SimCLR training complete.")


    def _validate_simclr(self, val_loader: DataLoader, epoch: int, epoch_step):

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
                    step=epoch_step
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
                lengths = batch["length"].to(self.device) 

                view0, view1, len0, len1 = self.transformation(audio, lengths)
                view0, view1 = view0.to(self.device), view1.to(self.device)
                len0, len1 = len0.to(self.device), len1.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    out0, out1 = self.model(view0, view1, lengths0=len0, lengths1=len1)
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
            
            epoch_step = (epoch + 1) * len(train_loader)        
            if self.wandb_logger.is_active:
                self.wandb_logger.log({"train/epoch_loss": avg_loss, "epoch": epoch + 1}, step=epoch_step)

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
                avg_val_loss = self._validate_cola(val_loader, epoch, epoch_step)

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
                    {"media/embedding_animation": wandb.Html(animation_path)},
                    step=max(embedding_logger.steps) if embedding_logger.steps else epochs
                )
                self.logger.info("Embedding animation logged to Weights & Biases.")

        self.logger.info("COLA training complete.")


    def _validate_cola(self, val_loader: DataLoader, epoch: int, epoch_step):

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
                    step=epoch_step
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
        num_hubert_iterations: int = 5,
        logger_loader: Optional[DataLoader] = None,
        use_embedding_logger: bool = False,
        **kwargs,
    ):
        """Train the HuBERT model with embedding logging via HuBERTBackbone."""
        transformer_layer = kwargs.get(
            "transformer_layer", getattr(self.model, "extractor_layer", None)
        )
        if transformer_layer is None:
            self.logger.warning("⚠️  No 'transformer_layer' specified for HuBERT.")

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
                    iteration_id=iteration,  # NEW: version KMeans per iteration
                )
                
                np.save(iteration_pseudo_labels_path, pseudo_labels_dict)
                self.logger.info(f"Saved pseudo-labels for iteration {iteration + 1}.")

            # === Adjust pseudo-labels to match dataset ===
            all_dataset_indices = set(range(len(train_loader_for_training.dataset)))
            all_pseudo_indices = set(pseudo_labels_dict.keys())
            
            extra = all_pseudo_indices - all_dataset_indices
            missing = all_dataset_indices - all_pseudo_indices
            
            if extra:
                self.logger.warning(f"⚠️  Removing {len(extra)} extra pseudo-labels: {sorted(list(extra))[:10]}...")
                for idx in extra:
                    pseudo_labels_dict.pop(idx, None)

            if missing:
                self.logger.warning(f"⚠️  Filling {len(missing)} missing pseudo-labels with zeros: {sorted(list(missing))[:10]}...")
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
                                f"⚠️  [Iteration {iteration+1}, Epoch {epoch+1}, Batch {batch_idx}] No masked positions. Skipping."
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

                    global_step = (iteration+1) * epoch * len(train_loader_for_training) + batch_idx
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

                epoch_step = (iteration+1) * (epoch + 1) * len(train_loader_for_training)        
                if self.wandb_logger.is_active:
                    self.wandb_logger.log(
                        {
                            "train/epoch_loss": avg_loss,
                            f"train/iter_{iteration+1}_epoch_loss": avg_loss,
                            "epoch": epoch + 1
                        },
                        step=epoch_step
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


                if hasattr(self, "_optuna_trial"):
                    metric = avg_loss
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
                        {f"media/embedding_animation/iter_{iteration+1}": wandb.Html(animation_path)},
                        step=max(embedding_logger.steps) if embedding_logger.steps else epochs
                    )
                    self.logger.info("Embedding animation logged to Weights & Biases.")

        self.logger.info("HuBERT training complete across all specified iterations.")

# --------------------------------------------------------------------------

    def _train_eat(
        self,
        train_loader: DataLoader,
        optimizer,
        epochs: int,
        start_epoch: int = 0,
        val_loader: Optional[DataLoader] = None,
        logger_loader: Optional[DataLoader] = None,
        use_embedding_logger: bool = False,
    ) -> None:
        """
        Pre-text training loop for EAT with the UFO loss.

        Works exactly like the other *_train_<method>() helpers:
        • identical W&B & tqdm logging
        • optional t-SNE embedding logger (uses EATBackbone)
        • periodic checkpoints & final checkpoint
        • optional validation loader
        """

        # -------------------------------------------------
        # 0.  (Optional) Step-0 embedding logging
        # -------------------------------------------------
        if use_embedding_logger:
            assert logger_loader is not None, (
                "logger_loader must be provided when use_embedding_logger=True"
            )
            embedding_log_dir = os.path.join(self.checkpoint_path, "embedding_logs")
            embedding_logger = EmbeddingLogger(
                log_dir=embedding_log_dir,
                method_name=self.method,
                reduce_method="tsne",
                log_interval=1,
            )
            self.logger.info(f"Embedding logger initialized at {embedding_log_dir}")

            self.logger.info("[EAT – Step 0] Logging pre-training embeddings…")
            backbone = EATBackbone(self.model).to(self.device).eval()

            all_emb, all_lab = [], []
            with torch.no_grad():
                for batch in tqdm(logger_loader, desc="EmbeddingLogger Step 0"):
                    audio = batch["audio"].to(self.device)
                    labels = batch["label"].to(self.device)
                    all_emb.append(backbone(audio))
                    all_lab.append(labels)

            embedding_logger.log_step(
                step=0,
                embeddings=torch.cat(all_emb,  dim=0),
                labels=torch.cat(all_lab, dim=0),
            )
            self.logger.info("[EAT – Step 0] Pre-training embeddings logged.")
            self.model.train()

        # -------------------------------------------------
        # 1.  W&B: watch model only once
        # -------------------------------------------------
        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(self.model)

        # -------------------------------------------------
        # 2.  Epoch loop
        # -------------------------------------------------
        self.model.train()
        for epoch in range(start_epoch, epochs):
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"EAT Epoch {epoch+1}/{epochs}")

            for batch_idx, batch in enumerate(pbar):
                audio = batch["audio"].to(self.device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    # Forward - returns lists for each masked-clone
                    decoded, target, cls_tok, teacher_avg = self.model(audio)
                    # UFO loss per clone → mean
                    clone_losses = [
                        self.loss(d, t, c, teacher_avg)
                        for d, t, c in zip(decoded, target, cls_tok)
                    ]
                    loss = torch.stack(clone_losses).mean()

                # Back-prop
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                # EMA teacher update
                with torch.no_grad():
                    self.model.update_teacher()

                # Logs
                running_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
                global_step = epoch * len(train_loader) + batch_idx
                if self.wandb_logger.is_active:
                    self.wandb_logger.log({"train/batch_loss": loss.item()}, step=global_step)

            # -------------------------------------------------
            # 2a.  Epoch-level bookkeeping
            # -------------------------------------------------
            avg_loss   = running_loss / len(train_loader)
            epoch_step = (epoch + 1) * len(train_loader)
            self.logger.info(f"[EAT – Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

            if self.wandb_logger.is_active:
                self.wandb_logger.log(
                    {"train/epoch_loss": avg_loss, "epoch": epoch + 1},
                    step=epoch_step,
                )

            # -------------------------------------------------
            # 2b.  (Optional) embedding logger per epoch
            # -------------------------------------------------
            if use_embedding_logger:
                self.logger.info(f"[EAT – Epoch {epoch+1}] Logging embeddings…")
                backbone = EATBackbone(self.model).to(self.device).eval()

                all_emb, all_lab = [], []
                with torch.no_grad():
                    for batch in tqdm(
                        logger_loader, desc=f"EmbeddingLogger Epoch {epoch+1}"
                    ):
                        audio  = batch["audio"].to(self.device)
                        labels = batch["label"].to(self.device)
                        all_emb.append(backbone(audio))
                        all_lab.append(labels)

                embedding_logger.log_step(
                    step=epoch + 1,
                    embeddings=torch.cat(all_emb,  dim=0),
                    labels=torch.cat(all_lab, dim=0),
                )
                self.model.train()

            # -------------------------------------------------
            # 2c.  (Optional) validation
            # -------------------------------------------------
            if val_loader:
                avg_val_loss = self._validate_eat(val_loader, epoch, epoch_step)

            # -------------------------------------------------
            # 2d.  Optuna pruning support
            # -------------------------------------------------
            if hasattr(self, "_optuna_trial"):
                metric = avg_val_loss if val_loader else avg_loss
                self._optuna_trial.report(metric, epoch)
                if self._optuna_trial.should_prune():
                    raise optuna.TrialPruned()

            # -------------------------------------------------
            # 2e.  Periodic checkpoints
            # -------------------------------------------------
            if (
                (epoch + 1) % self.checkpoint_interval == 0
                and not hasattr(self, "_optuna_trial")
            ):
                ckpt_path = os.path.join(
                    self.checkpoint_path,
                    f"{self.method}_model_{self.timestamp}_epoch{epoch+1}.pth",
                )
                torch.save(self.model.state_dict(), ckpt_path)
                self.logger.info(f"Model checkpoint saved: {ckpt_path}")

                if self.wandb_logger.is_active:
                    self.wandb_logger.save_artifact(
                        ckpt_path,
                        name=f"{self.method}-model-epoch-{epoch+1}",
                        type="model",
                        metadata={"epoch": epoch + 1, "loss": avg_loss},
                    )

        # -------------------------------------------------
        # 3.  Final checkpoint
        # -------------------------------------------------
        final_ckpt = os.path.join(
            self.checkpoint_path,
            f"{self.method}_model_{self.timestamp}_final.pth",
        )
        torch.save(self.model.state_dict(), final_ckpt)
        self.logger.info(f"Final model checkpoint saved: {final_ckpt}")

        if self.wandb_logger.is_active:
            self.wandb_logger.save_artifact(
                final_ckpt,
                name=f"{self.method}-model-final",
                type="model",
                metadata={"epochs_trained": epochs, "final_loss": avg_loss},
            )

        # -------------------------------------------------
        # 4.  Final embedding animation
        # -------------------------------------------------
        if use_embedding_logger:
            self.logger.info("Generating final embedding animation…")
            anim_path = embedding_logger.plot_all()
            self.logger.info(f"Embedding animation saved at: {anim_path}")

            if self.wandb_logger.is_active:
                import wandb

                self.wandb_logger.log(
                    {"media/embedding_animation": wandb.Html(anim_path)},
                    step=max(embedding_logger.steps) if embedding_logger.steps else epochs,
                )

        self.logger.info("EAT pre-text training complete.")



    def _validate_eat(self, val_loader: DataLoader, epoch: int, epoch_step):
        """Validation loop for EAT + UFO (no gradient)."""
        self.model.eval()
        val_running_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Validation EAT Epoch {epoch+1}")
            for batch in pbar:
                audio = batch["audio"].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    decoded, target, cls_tok, teacher_avg = self.model(audio)
                    clone_losses = [
                        self.loss(d, t, c, teacher_avg)
                        for d, t, c in zip(decoded, target, cls_tok)
                    ]
                    loss = torch.stack(clone_losses).mean()

                val_running_loss += loss.item()

            avg_val_loss = val_running_loss / len(val_loader)
            self.logger.info(f"[EAT – Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}")

            if self.wandb_logger.is_active:
                self.wandb_logger.log(
                    {"val/loss": avg_val_loss},
                    step=epoch_step,
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
        logger_loader: Optional[DataLoader] = None, 
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
                    "⚠️  [Dataset Check] Your dataset should return both 'audio' and 'length' keys. "
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
                    self.logger.warning("⚠️  [Dataset Check] val_loader should return both 'audio' and 'length'.")

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
                    "⚠️  [Dataset Check] Your dataset should return both 'audio' and 'length' keys. "
                    "Currently missing: "
                    + ", ".join(k for k in ["audio", "length"] if k not in first_sample)
                )
            
            if val_dataset:
                self.logger.warning(
                    "⚠️  HuBERT pre-training uses on-the-fly pseudo-labels; external "
                    "validation sets aren’t compatible. Validation step will be skipped."
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
            )
            

            train_loader_for_training = DataLoader(
                wrapped_train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,

            )

            self._train_hubert(
                train_loader_for_training=train_loader_for_training,
                train_loader_full_dataset=train_loader_for_pseudo_label_gen,
                optimizer=optimizer,
                epochs=epochs,
                start_epoch=start_epoch,
                start_iteration=start_iteration,
                num_hubert_iterations=kwargs.get(
                    "num_hubert_iterations", getattr(self.model, "config", {}).get("max_iterations", 2)
                ),
                transformer_layer=kwargs.get(
                    "transformer_layer", getattr(self.model, "config", {}).get("extractor_layer", None)
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


        elif self.method == "eat":
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
            self._train_eat(
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

    def _evaluate_wav2vec2_2(
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
        Evaluation for Wav2Vec2 using CTC with mixed precision (AMP).
        Uses TorchAudio CTC beam search decoder + KenLM 4-gram (OpenSLR-11).
        Downloads the official LibriSpeech lexicon and filters it to ONLY words
        that (1) are representable by our tokens and (2) exist in the LM vocabulary.
        Computes TRUE WER on words.
        """

        # --- helpers (match EAT) ---
        def normalize_sentence(text: str) -> str:
            text = text.lower().strip()
            return " ".join(text.split())

        def tokens_to_text(tokens):
            # tokens: list[str] of characters, including SPACE token " "
            return "".join(tokens)

        # === Backbone & CTC head ===
        backbone = Wav2Vec2Backbone(pretrained_model=self.model)
        feature_size = self.model.model_config["encoder_embed_dim"]

        classifier = EvaluateNet(
            backbone=backbone,
            feature_size=feature_size,
            num_classes=num_classes,  # vocab size incl. blank (blank=0)
            is_linear=freeze_backbone,
        ).to(self.device)

        # True freeze semantics when requested
        if freeze_backbone:
            classifier.backbone.eval()
            for p in classifier.backbone.parameters():
                p.requires_grad = False

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, classifier.parameters()),
            lr=lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=1e-4
        )
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=self.collate_ctc, num_workers=self.num_workers, pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=self.collate_ctc, num_workers=self.num_workers, pin_memory=True,
        )

        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(classifier)

        # AMP
        use_amp = (self.device.type == "cuda")
        scaler = GradScaler(enabled=use_amp)

        # === Training ===
        classifier.train()
        for epoch in range(epochs):
            running, seen = 0.0, 0
            pbar = tqdm(train_loader, desc=f"[Wav2Vec2-CTC Training] Epoch {epoch+1}")
            for batch in pbar:
                if batch is None:
                    continue
                waveforms = batch["audio"].to(self.device)                 # (B,1,T_pad)
                labels = batch["flat_labels"].to(self.device)              # values 1..C-1 (0 is blank)
                label_lengths = batch["label_lengths"].to(self.device)     # (B,)
                audio_lengths = batch["audio_lengths"].to(self.device)     # (B,)

                with autocast(enabled=use_amp):
                    log_probs, output_lengths = classifier(waveforms, audio_lengths)

                assert output_lengths.shape[0] == waveforms.size(0)
                assert (output_lengths > 0).all()
                assert output_lengths.max().item() <= log_probs.size(1)

                loss = criterion(
                    log_probs.float().permute(1, 0, 2),  # (T,B,C)
                    labels, output_lengths, label_lengths
                )

                if torch.isnan(loss):
                    self.logger.error("❌ NaN loss detected!")
                    self.logger.error(f"log_probs: {log_probs.shape}, "
                                    f"labels: {labels.shape}, "
                                    f"input_lengths: {output_lengths}, "
                                    f"label_lengths: {label_lengths}")
                    continue

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

                bs = waveforms.size(0)
                running += loss.item() * bs
                seen += bs
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            epoch_loss = running / max(seen, 1)
            self.logger.info(f"[Wav2Vec2-CTC Train] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
            if self.wandb_logger.is_active:
                self.wandb_logger.log({
                    "wav2vec2/train_loss": epoch_loss,
                    "wav2vec2/epoch": epoch + 1,
                    "wav2vec2/lr": optimizer.param_groups[0]["lr"]
                }, step=epoch + 1)

        # === Evaluation: TorchAudio CTC beam search + KenLM ===
        classifier.eval()

        # Shared vocab checks
        idx2label = getattr(train_dataset, "idx2label", None)
        if idx2label is None:
            raise RuntimeError("train_dataset is missing 'idx2label' for decoding.")
        if hasattr(test_dataset, "label2idx") and hasattr(train_dataset, "label2idx"):
            assert test_dataset.label2idx == train_dataset.label2idx, "Train/Test vocab mismatch!"
        assert num_classes == len(idx2label) + 1, \
            f"num_classes ({num_classes}) must be len(idx2label)+1 ({len(idx2label)+1}) with blank=0."

        # --- Build decoder tokens (blank '_' first; map SPACE -> '|') ---
        if hasattr(train_dataset, "get_decoder_tokens"):
            decoder_tokens = train_dataset.get_decoder_tokens()
        else:
            decoder_tokens = ["_"] + [("|" if t == " " else t) for t in idx2label]
        assert len(decoder_tokens) == num_classes, "Decoder tokens must match model output classes"

        # --- Download official LM + lexicon, and FILTER the lexicon ---
        from torchaudio.models.decoder import ctc_decoder
        import os, gzip, shutil, urllib.request, re

        def _download(url: str, dst: str):
            if not os.path.exists(dst):
                self.logger.info(f"Downloading: {url} -> {dst}")
                urllib.request.urlretrieve(url, dst)
            return dst

        def _ensure_lm_and_filtered_lexicon(cache_dir: str = "decoder_assets"):
            """
            Downloads from OpenSLR-11:
            - 4-gram.arpa.gz  -> 4-gram.arpa
            - librispeech-lexicon.txt (OFFICIAL)
            Filters the official lexicon to include ONLY words that:
            (1) are spellable with our tokens (a-z and apostrophe), and
            (2) exist in the LM's 1-gram vocabulary.
            Produces a CHARACTER lexicon (word -> sequence of characters + '|').
            Returns (filtered_char_lexicon_path, lm_arpa_path).
            """
            os.makedirs(cache_dir, exist_ok=True)
            base = "https://www.openslr.org/resources/11"

            lm_gz = os.path.join(cache_dir, "4-gram.arpa.gz")
            lm_arpa = os.path.join(cache_dir, "4-gram.arpa")
            official_lex = os.path.join(cache_dir, "librispeech-lexicon.txt")
            char_lex = os.path.join(cache_dir, "lexicon_chars.filtered.txt")

            _download(f"{base}/librispeech-lexicon.txt", official_lex)
            if not os.path.exists(lm_arpa):
                _download(f"{base}/4-gram.arpa.gz", lm_gz)
                with gzip.open(lm_gz, "rb") as fin, open(lm_arpa, "wb") as fout:
                    shutil.copyfileobj(fin, fout)

            # Build LM 1-gram vocabulary (lowercased)
            lm_vocab = set()
            in_unigram = False
            with open(lm_arpa, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("\\1-grams:"):
                        in_unigram = True
                        continue
                    if in_unigram and line.startswith("\\2-grams:"):
                        break
                    if in_unigram:
                        parts = line.split()
                        if len(parts) >= 2:
                            lm_vocab.add(parts[1].lower())

            # Allowed character set (exclude '_' and '|' which are special)
            allowed_chars = set(t for t in decoder_tokens if t not in {"_", "|"})
            # Filter the OFFICIAL lexicon by allowed chars + LM vocab, and emit char spellings
            if not os.path.exists(char_lex):
                kept, skipped = 0, 0
                seen = set()
                with open(official_lex, "r", encoding="utf-8", errors="ignore") as fin, \
                    open(char_lex, "w", encoding="utf-8") as fout:
                    for line in fin:
                        line = line.strip()
                        if not line:
                            continue
                        # official format: WORD PRON1 [PRON2 ...]  (phones — we ignore phones)
                        word = line.split()[0].lower()
                        if word in seen:
                            continue  # multiple pronunciations -> dedupe
                        seen.add(word)

                        # must be in LM vocab and spellable by our charset
                        if word in lm_vocab and re.fullmatch(r"[a-z']+", word) and all(ch in allowed_chars for ch in word):
                            spelling = " ".join(list(word) + ["|"])  # character sequence + word boundary
                            fout.write(f"{word} {spelling}\n")
                            kept += 1
                        else:
                            skipped += 1
                self.logger.info(f"[CTC-Decoder] Filtered official lexicon: kept={kept}, skipped={skipped}")

            return char_lex, lm_arpa

        # Try to build filtered lexicon; if it fails (e.g., no internet), fall back
        try:
            lexicon_path, lm_path = _ensure_lm_and_filtered_lexicon()
        except Exception as e:
            self.logger.warning(f"OpenSLR LM/lexicon setup failed: {e}. "
                                f"Falling back to lexicon-free decoding (no LM).")
            lexicon_path, lm_path = None, None

        # Create the decoder (use '_' as blank; '|' for space/silence).
        # We DO NOT pass lm_dict: by filtering the lexicon to words in the LM,
        # the decoder's internal LM dictionary and our lexicon stay consistent,
        # avoiding the "Unknown entry in dictionary" errors.
        decoder = ctc_decoder(
            lexicon=lexicon_path,
            tokens=decoder_tokens,
            lm=lm_path,
            nbest=1,
            beam_size=100,
            beam_threshold=10.0,
            lm_weight=2.0,
            word_score=-1.0,
            sil_token="|",
            blank_token="_",
            unk_word="<unk>",
        )

        ref_sentences, hyp_sentences = [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="[Wav2Vec2-CTC Evaluation: Beam+LM]"):
                if batch is None:
                    continue
                waveforms = batch["audio"].to(self.device)
                audio_lengths = batch["audio_lengths"].to(self.device)

                # (optional) refs (already +1 shifted; 0 is pad/blank)
                labels_padded = batch["labels"].cpu().tolist()
                label_lengths = batch["label_lengths"].cpu().tolist()

                with autocast(enabled=use_amp):
                    log_probs, out_lengths = classifier(waveforms, audio_lengths)  # (B,T,C) log-probs

                emissions = log_probs.detach().cpu()          # (B,T,C)
                emission_lengths = out_lengths.detach().cpu() # (B,)

                hypos_batch = decoder(emissions, emission_lengths)  # List[List[CTCHypothesis]]

                for hypos, ref_seq, ref_len in zip(hypos_batch, labels_padded, label_lengths):
                    # Best hypothesis -> words if available, otherwise fall back to tokens
                    if hypos and getattr(hypos[0], "words", None):
                        hyp_text = " ".join(hypos[0].words)
                    else:
                        tok_ids = hypos[0].tokens if hypos else []
                        tok_syms = [decoder_tokens[i] for i in tok_ids]
                        hyp_text = "".join(tok_syms).replace("|", " ")

                    # Build reference from character labels (undo +1 shift; map to chars)
                    ref_ids = ref_seq[:ref_len]
                    ref_chars = [idx2label[r - 1] for r in ref_ids]  # r >= 1
                    ref_text = tokens_to_text(ref_chars)

                    # Normalize (lowercase/whitespace)
                    hyp_text = normalize_sentence(hyp_text)
                    ref_text = normalize_sentence(ref_text)

                    hyp_sentences.append(hyp_text)
                    ref_sentences.append(ref_text)

        # TRUE word-level WER
        wer_score = wer(ref_sentences, hyp_sentences)
        self.logger.info(f"📊 [Wav2Vec2-CTC Evaluation] WER(words)={wer_score:.4f} (beam search + LM)")

        if self.wandb_logger.is_active:
            self.wandb_logger.log({"wav2vec2/test_wer": wer_score})

        # Save head weights
        torch.save(classifier.state_dict(), "Wav2Vec2_Classifier.pth")

    def _evaluate_wav2vec2(
        self,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        num_classes: int,
        batch_size: int = 64,
        lr: float = 1e-3,
        epochs: int = 10,
        freeze_backbone: bool = True,
        # Optional: provide phoneme lexicon + ARPA LM if you have them; otherwise decoder runs token-only.
        phoneme_lexicon_path: str | None = None,
        phoneme_lm_arpa_path: str | None = None,
        classifier_path: str | None = None,  # optionally load existing head weights
        **kwargs
    ):
        """
        Wav2Vec2 + CTC for TIMIT phoneme recognition (PER).
        Assumptions:
        - train/test datasets expose `idx2label` (0..P-1 -> phoneme string) and matching `label2idx`.
        - Your collate_ctc returns (+1 shifted labels; 0=CTC blank/pad):
                "audio": (B,1,T),
                "audio_lengths": (B,),  # raw samples
                "labels": (B,Lmax) in [0..P], 0 is pad/blank; valid part is [1..P]
                "flat_labels": (sum L_b,) in [1..P],
                "label_lengths": (B,)
        - num_classes = P + 1  (index 0 is the blank)
        """
        import os
        import editdistance
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torch.cuda.amp import GradScaler, autocast
        from torchaudio.models.decoder import ctc_decoder
        from tqdm.auto import tqdm

        # ----------------- helpers -----------------
        def collapse_and_strip_blanks(pred_ids):
            out, prev = [], None
            for p in pred_ids:
                if p != 0 and p != prev:  # 0 is CTC blank
                    out.append(p)
                prev = p
            return out  # still +1 shifted (>=1)

        def ids_to_phones(ids, idx2label):
            # ids must be in [1..P]; map via (i-1)
            if any(i == 0 for i in ids):
                raise RuntimeError("Found 0 (blank) in phone ids after collapse.")
            return [idx2label[i - 1] for i in ids]

        def compute_per(refs, hyps):
            total_err, total_len = 0, 0
            for r, h in zip(refs, hyps):
                total_err += editdistance.eval(r, h)
                total_len += len(r)
            return total_err / max(total_len, 1)

        # ----------------- backbone + head -----------------
        backbone = Wav2Vec2Backbone(pretrained_model=self.model)
        feature_size = self.model.model_config["encoder_embed_dim"]

        classifier = EvaluateNet(
            backbone=backbone,
            feature_size=feature_size,
            num_classes=num_classes,  # includes blank class at index 0
            is_linear=freeze_backbone,
        ).to(self.device)

        if classifier_path is not None and os.path.exists(classifier_path):
            self.logger.info(f"Loading classifier weights from: {classifier_path}")
            classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))

        if freeze_backbone:
            classifier.backbone.eval()
            for p in classifier.backbone.parameters():
                p.requires_grad = False

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, classifier.parameters()),
            lr=lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=1e-4
        )
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=self.collate_ctc, num_workers=self.num_workers, pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=self.collate_ctc, num_workers=self.num_workers, pin_memory=True,
        )

        if getattr(self, "wandb_logger", None) and self.wandb_logger.is_active:
            self.wandb_logger.watch_model(classifier)

        use_amp = (self.device.type == "cuda")
        scaler = GradScaler(enabled=use_amp)

        # ----------------- training -----------------
        classifier.train()
        for epoch in range(epochs):
            running, seen = 0.0, 0
            pbar = tqdm(train_loader, desc=f"[Wav2Vec2-CTC Training] Epoch {epoch+1}")
            for batch in pbar:
                if batch is None:
                    continue
                waveforms = batch["audio"].to(self.device)                 # (B,1,T)
                labels = batch["flat_labels"].to(self.device)              # +1 shifted in [1..P]
                label_lengths = batch["label_lengths"].to(self.device)     # (B,)
                audio_lengths = batch["audio_lengths"].to(self.device)     # (B,)

                # safety: labels must be >=1 (since +1 shift already applied)
                if (labels == 0).any():
                    raise RuntimeError("flat_labels contains 0, but collate should shift to [1..P].")

                with autocast(enabled=use_amp):
                    log_probs, output_lengths = classifier(waveforms, audio_lengths)  # (B,T,C), (B,)

                assert output_lengths.shape[0] == waveforms.size(0)
                assert (output_lengths > 0).all()
                assert output_lengths.max().item() <= log_probs.size(1)

                loss = criterion(
                    log_probs.float().permute(1, 0, 2),  # (T,B,C)
                    labels, output_lengths, label_lengths
                )

                if torch.isnan(loss):
                    self.logger.error("❌ NaN loss detected!")
                    self.logger.error(f"log_probs: {log_probs.shape}, "
                                    f"labels: {labels.shape}, "
                                    f"input_lengths: {output_lengths}, "
                                    f"label_lengths: {label_lengths}")
                    continue

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

                bs = waveforms.size(0)
                running += loss.item() * bs
                seen += bs
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            epoch_loss = running / max(seen, 1)
            self.logger.info(f"[Wav2Vec2-CTC Train] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
            if getattr(self, "wandb_logger", None) and self.wandb_logger.is_active:
                self.wandb_logger.log({
                    "wav2vec2/train_loss": epoch_loss,
                    "wav2vec2/epoch": epoch + 1,
                    "wav2vec2/lr": optimizer.param_groups[0]["lr"]
                }, step=epoch + 1)

        # ----------------- evaluation (beam search + PER) -----------------
        classifier.eval()

        idx2label = getattr(train_dataset, "idx2label", None)
        if idx2label is None:
            raise RuntimeError("train_dataset is missing 'idx2label' for decoding.")

        if hasattr(test_dataset, "label2idx") and hasattr(train_dataset, "label2idx"):
            assert test_dataset.label2idx == train_dataset.label2idx, "Train/Test vocab mismatch!"

        assert num_classes == len(idx2label) + 1, (
            f"num_classes={num_classes}, but need P+1={len(idx2label)+1} (blank=0)."
        )

        # tokens index-aligned with logits: 0 -> blank "_", 1..P -> idx2label[0..P-1]
        decoder_tokens = ["_"] + list(idx2label)
        assert len(decoder_tokens) == num_classes

        # instantiate decoder (optional LM/lexicon); fall back to greedy if construction fails
        try:
            use_lex = bool(phoneme_lexicon_path) and os.path.exists(phoneme_lexicon_path)
            use_lm  = bool(phoneme_lm_arpa_path) and os.path.exists(phoneme_lm_arpa_path)

            decoder = ctc_decoder(
                # FIX: actually honor use_lex
                lexicon=phoneme_lexicon_path if use_lex else None,
                tokens=decoder_tokens,
                lm=phoneme_lm_arpa_path if use_lm else None,
                nbest=50,
                beam_size=250,
                beam_threshold=12.0,
                lm_weight=1.5,
                word_score=-1.0,
                # FIX: match dataset token, not "SIL"
                sil_token="sil",     # phone-level search
                blank_token="_",
                unk_word="<unk>",
            )


            self.logger.info(f"[CTC-Decoder] Using lexicon ({use_lex}) / LM for decoding ({use_lm}).")


        except Exception as e:
            self.logger.warning(f"CTC decoder setup failed ({e}). Falling back to greedy PER.")
            decoder = None

        ref_seqs, hyp_seqs = [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="[Wav2Vec2-CTC Evaluation: Phonemes]"):
                if batch is None:
                    continue
                waveforms = batch["audio"].to(self.device)
                audio_lengths = batch["audio_lengths"].to(self.device)

                labels_padded = batch["labels"].cpu().tolist()       # +1 shifted
                label_lengths = batch["label_lengths"].cpu().tolist()

                # safety: ensure no 0 inside valid reference spans
                for rs, rl in zip(labels_padded, label_lengths):
                    if any(x == 0 for x in rs[:rl]):
                        raise RuntimeError("Reference contains 0 (blank/pad) within valid length.")

                with autocast(enabled=use_amp):
                    log_probs, out_lengths = classifier(waveforms, audio_lengths)  # (B,T,C)

                emissions = log_probs.detach().cpu()          # (B,T,C)
                emission_lengths = out_lengths.detach().cpu() # (B,)

                if decoder is not None:
                    hypos_batch = decoder(emissions, emission_lengths)  # List[List[CTCHypothesis]]
                    for hypos, ref_seq, ref_len in zip(hypos_batch, labels_padded, label_lengths):
                        if hypos:
                            tok_ids = collapse_and_strip_blanks(hypos[0].tokens)  # in [1..P]
                            if any(i == 0 for i in tok_ids):
                                raise RuntimeError("Decoder hypothesis still has 0 after collapse.")
                            hyp = ids_to_phones(tok_ids, idx2label)
                        else:
                            hyp = []

                        ref_ids = ref_seq[:ref_len]  # in [1..P]
                        ref = ids_to_phones(ref_ids, idx2label)

                        hyp_seqs.append(hyp)
                        ref_seqs.append(ref)
                else:
                    preds = torch.argmax(emissions, dim=-1).tolist()
                    for pred_seq, ref_seq, ref_len in zip(preds, labels_padded, label_lengths):
                        pred_ids = collapse_and_strip_blanks(pred_seq)  # in [1..P]
                        if any(i == 0 for i in pred_ids):
                            raise RuntimeError("Greedy hypothesis still has 0 after collapse.")
                        hyp = ids_to_phones(pred_ids, idx2label)

                        ref_ids = ref_seq[:ref_len]  # in [1..P]
                        ref = ids_to_phones(ref_ids, idx2label)

                        hyp_seqs.append(hyp)
                        ref_seqs.append(ref)

        per_score = compute_per(ref_seqs, hyp_seqs)
        self.logger.info(f"📊 [Wav2Vec2-CTC Evaluation] PER(phonemes)={per_score:.4f}")

        if getattr(self, "wandb_logger", None) and self.wandb_logger.is_active:
            self.wandb_logger.log({"wav2vec2/test_per": per_score})

        torch.save(classifier.state_dict(), "Wav2Vec2_Classifier.pth")




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
        Evaluation for SimCLR Speech using CTC (WER/PER) with mixed precision (AMP).
        """

        model = self.model
        backbone = SimCLRBackbone(model)
        feature_size = model.backbone.embed_dim

        classifier = EvaluateNet(
            backbone=backbone,
            feature_size=feature_size,
            num_classes=num_classes,   # vocab size incl. blank
            is_linear=freeze_backbone,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, classifier.parameters()),
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=1e-4
        )

        criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_ctc,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_ctc,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(classifier)

        # === AMP setup ===
        use_amp = (self.device.type == "cuda")
        scaler = GradScaler(enabled=use_amp)

        # === Training loop ===
        classifier.train()
        if freeze_backbone:
            # Keep backbone deterministic even while classifier is in train mode
            classifier.backbone.eval()

        for epoch in range(epochs):
            for batch in tqdm(train_loader, desc=f"[SimCLR-CTC Training] Epoch {epoch+1}"):
                wavs = batch["audio"].to(self.device)
                labels = batch["flat_labels"].to(self.device)       # 1..C-1 (0 is blank)
                label_lengths = batch["label_lengths"].to(self.device)
                audio_lengths = batch["audio_lengths"].to(self.device)

                # Forward under autocast
                with autocast(enabled=use_amp):
                    log_probs, output_lengths = classifier(wavs, audio_lengths)

                # Compute CTC loss in fp32; CTC expects (T, B, C)
                loss = criterion(
                    log_probs.float().permute(1, 0, 2),
                    labels,
                    output_lengths,
                    label_lengths
                )

                if torch.isnan(loss):
                    self.logger.error("❌ NaN loss detected!")
                    self.logger.error(f"log_probs: {log_probs.shape}, "
                                    f"labels: {labels.shape}, "
                                    f"input_lengths: {audio_lengths}, "
                                    f"label_lengths: {label_lengths}")
                    continue

                optimizer.zero_grad(set_to_none=True)

                # AMP backward/step
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

            self.logger.info(f"[SimCLR-CTC Eval] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

            if self.wandb_logger.is_active:
                self.wandb_logger.log({
                    "simclr/train_loss": loss.item(),
                    "simclr/epoch": epoch + 1,
                    "simclr/lr": optimizer.param_groups[0]["lr"]
                }, step=epoch + 1)

        # === Evaluation loop ===
        classifier.eval()
        all_refs_tokens, all_hyps_tokens = [], []

        # Shared decoding map (0..C-2 → token). Predictions/refs use +1 shift; 0 is blank/pad.
        idx2label = getattr(train_dataset, "idx2label", None)
        if idx2label is None:
            raise RuntimeError("train_dataset is missing 'idx2label' for decoding.")
        if hasattr(test_dataset, "label2idx") and hasattr(train_dataset, "label2idx"):
            assert test_dataset.label2idx == train_dataset.label2idx, "Train/Test vocab mismatch!"

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="[SimCLR-CTC Evaluation]"):
                wavs = batch["audio"].to(self.device)
                audio_lengths = batch["audio_lengths"].to(self.device)

                # Padded labels (+1 shift; 0 is blank/pad) with true lengths
                labels_padded = batch["labels"].cpu().tolist()
                label_lengths = batch["label_lengths"].cpu().tolist()

                # Inference under autocast (no scaler/backprop)
                with autocast(enabled=use_amp):
                    log_probs, out_lengths = classifier(wavs, audio_lengths)

                preds = torch.argmax(log_probs, dim=-1).cpu().tolist()
                out_lengths = out_lengths.cpu().tolist()

                # Greedy CTC decoding with proper trimming/collapse
                for pred_seq, ref_seq, ref_len, out_len in zip(preds, labels_padded, label_lengths, out_lengths):
                    pred_seq = pred_seq[:out_len]  # Trim predictions to valid output length

                    # Collapse repeats & remove blanks (blank_id=0)
                    hyp_ids = []
                    prev = None
                    for p in pred_seq:
                        if p == 0:
                            prev = p
                            continue
                        if p != prev:
                            hyp_ids.append(p)
                        prev = p

                    # Map ids→tokens by undoing the +1 shift (ids are >= 1 here)
                    hyp_tokens = [idx2label[p - 1] for p in hyp_ids]
                    ref_ids = ref_seq[:ref_len]  # slice removes right padding zeros
                    ref_tokens = [idx2label[r - 1] for r in ref_ids]

                    all_hyps_tokens.append(hyp_tokens)
                    all_refs_tokens.append(ref_tokens)

        # -------- DROP sil + closures before scoring --------
        DROP_TOKENS = {"sil", "tcl", "kcl", "pcl", "dcl", "gcl", "bcl"}

        def _filter(tokens):
            return [t for t in tokens if t not in DROP_TOKENS]

        refs_filt = [_filter(r) for r in all_refs_tokens]
        hyps_filt = [_filter(h) for h in all_hyps_tokens]

        # Token-level WER (phones, post-filter)
        ref_texts = [" ".join(r) for r in refs_filt]
        hyp_texts = [" ".join(h) for h in hyps_filt]
        wer_score = wer(ref_texts, hyp_texts)

        # True PER (post-filter)
        per_numer = sum(edit_distance(r, h) for r, h in zip(refs_filt, hyps_filt))
        per_denom = sum(len(r) for r in refs_filt) if refs_filt else 1
        per_score = per_numer / per_denom
        # ----------------------------------------------------

        self.logger.info(f"📊 [SimCLR-CTC Evaluation] WER(phones,no_sil)={wer_score:.4f}, PER(no_sil)={per_score:.4f}")

        if self.wandb_logger.is_active:
            self.wandb_logger.log({
                "simclr/test_wer_no_sil": wer_score,
                "simclr/test_per_no_sil": per_score
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
        Evaluation for HuBERT using CTC (WER/PER) with mixed precision (AMP).
        """

        model = self.model
        feature_size = model.config["encoder_embed_dim"]

        backbone = HuBERTBackbone(model)

        classifier = EvaluateNet(
            backbone=backbone,
            feature_size=feature_size,
            num_classes=num_classes,   # vocab size incl. blank
            is_linear=freeze_backbone,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, classifier.parameters()),
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=1e-4
        )

        criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_ctc,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_ctc,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(classifier)

        # === AMP setup ===
        use_amp = (self.device.type == "cuda")
        scaler = GradScaler(enabled=use_amp)

        # === Training loop ===
        classifier.train()
        if freeze_backbone:
            classifier.backbone.eval()

        for epoch in range(epochs):
            for batch in tqdm(train_loader, desc=f"[HuBERT-CTC Training] Epoch {epoch+1}"):
                waveforms = batch["audio"].to(self.device)
                labels = batch["flat_labels"].to(self.device)       # 1..C-1 (0 is blank)
                label_lengths = batch["label_lengths"].to(self.device)
                audio_lengths = batch["audio_lengths"].to(self.device)

                # Forward under autocast
                with autocast(enabled=use_amp):
                    log_probs, output_lengths = classifier(waveforms, audio_lengths)

                # Compute CTC loss in fp32 for stability; (T, B, C)
                loss = criterion(
                    log_probs.float().permute(1, 0, 2),
                    labels,
                    output_lengths,
                    label_lengths
                )

                if torch.isnan(loss):
                    self.logger.error("❌ NaN loss detected!")
                    self.logger.error(f"log_probs: {log_probs.shape}, "
                                    f"labels: {labels.shape}, "
                                    f"input_lengths: {audio_lengths}, "
                                    f"label_lengths: {label_lengths}")
                    continue

                optimizer.zero_grad(set_to_none=True)

                # AMP backward/step
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

            self.logger.info(f"[HuBERT-CTC Eval] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

            if self.wandb_logger.is_active:
                self.wandb_logger.log({
                    "hubert/train_loss": loss.item(),
                    "hubert/epoch": epoch + 1,
                    "hubert/lr": optimizer.param_groups[0]["lr"]
                }, step=epoch + 1)

        # === Evaluation loop ===
        classifier.eval()
        all_refs_tokens, all_hyps_tokens = [], []

        idx2label = getattr(train_dataset, "idx2label", None)
        if idx2label is None:
            raise RuntimeError("train_dataset is missing 'idx2label' for decoding.")

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="[HuBERT-CTC Evaluation]"):
                waveforms = batch["audio"].to(self.device)
                audio_lengths = batch["audio_lengths"].to(self.device)

                # Padded labels (+1 shift) and true lengths
                labels_padded = batch["labels"].cpu().tolist()
                label_lengths = batch["label_lengths"].cpu().tolist()

                # Inference under autocast (no scaler/backprop)
                with autocast(enabled=use_amp):
                    log_probs, out_lengths = classifier(waveforms, audio_lengths)

                preds = torch.argmax(log_probs, dim=-1).cpu().tolist()
                out_lengths = out_lengths.cpu().tolist()

                # Greedy CTC decoding with proper trimming/collapse
                for pred_seq, ref_seq, ref_len, out_len in zip(preds, labels_padded, label_lengths, out_lengths):
                    pred_seq = pred_seq[:out_len]  # Trim to valid output length

                    # Collapse repeats & remove blanks (blank_id=0)
                    hyp_ids = []
                    prev = None
                    for p in pred_seq:
                        if p == 0:
                            prev = p
                            continue
                        if p != prev:
                            hyp_ids.append(p)
                        prev = p

                    # Map ids→tokens by undoing the +1 shift (ids are >= 1 here)
                    hyp_tokens = [idx2label[p - 1] for p in hyp_ids]
                    ref_ids = ref_seq[:ref_len]  # remove right padding zeros
                    ref_tokens = [idx2label[r - 1] for r in ref_ids]

                    all_hyps_tokens.append(hyp_tokens)
                    all_refs_tokens.append(ref_tokens)

        # -------- DROP sil + closures before scoring --------
        DROP_TOKENS = {"sil", "tcl", "kcl", "pcl", "dcl", "gcl", "bcl"}

        def _filter(tokens):
            return [t for t in tokens if t not in DROP_TOKENS]

        refs_filt = [_filter(r) for r in all_refs_tokens]
        hyps_filt = [_filter(h) for h in all_hyps_tokens]

        # Token-level WER (phones, post-filter)
        ref_texts = [" ".join(r) for r in refs_filt]
        hyp_texts = [" ".join(h) for h in hyps_filt]
        wer_score = wer(ref_texts, hyp_texts)

        # True PER (post-filter)
        per_numer = sum(edit_distance(r, h) for r, h in zip(refs_filt, hyps_filt))
        per_denom = sum(len(r) for r in refs_filt) if refs_filt else 1
        per_score = per_numer / per_denom
        # ----------------------------------------------------

        self.logger.info(f"📊 [HuBERT-CTC Evaluation] WER(phones,no_sil)={wer_score:.4f}, PER(no_sil)={per_score:.4f}")

        if self.wandb_logger.is_active:
            self.wandb_logger.log({
                "hubert/test_wer_no_sil": wer_score,
                "hubert/test_per_no_sil": per_score
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
        Evaluation for COLA using CTC (WER/PER) with mixed precision (AMP).
        """

        backbone = COLABackbone(self.model)
        feature_size = self.model.feature_size

        classifier = EvaluateNet(
            backbone=backbone,
            feature_size=feature_size,
            num_classes=num_classes,   # vocab size incl. blank
            is_linear=freeze_backbone,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, classifier.parameters()),
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=1e-4
        )

        criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_ctc,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_ctc,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        # Watch the classifier model
        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(classifier)

        # === AMP setup ===
        use_amp = (self.device.type == "cuda")
        scaler = GradScaler(enabled=use_amp)

        # === Training ===
        classifier.train()
        if freeze_backbone:
            classifier.backbone.eval()

        for epoch in range(epochs):
            for batch in tqdm(train_loader, desc=f"[COLA-CTC Training] Epoch {epoch+1}"):
                audio = batch["audio"].to(self.device)
                labels = batch["flat_labels"].to(self.device)       # 1..C-1 (0 is blank)
                label_lengths = batch["label_lengths"].to(self.device)
                audio_lengths = batch["audio_lengths"].to(self.device)

                # Forward under autocast
                with autocast(enabled=use_amp):
                    log_probs, output_lengths = classifier(audio, audio_lengths)

                # Compute CTC loss in fp32 for stability
                loss = criterion(
                    log_probs.float().permute(1, 0, 2),  # CTC expects (T, B, C)
                    labels,
                    output_lengths,
                    label_lengths
                )

                if torch.isnan(loss):
                    self.logger.error("❌ NaN loss detected!")
                    self.logger.error(f"log_probs: {log_probs.shape}, "
                                    f"labels: {labels.shape}, "
                                    f"input_lengths: {audio_lengths}, "
                                    f"label_lengths: {label_lengths}")
                    continue

                optimizer.zero_grad(set_to_none=True)

                # AMP backward/step
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

            self.logger.info(f"[COLA-CTC Eval] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

            if self.wandb_logger.is_active:
                self.wandb_logger.log({
                    "cola/train_loss": loss.item(),
                    "cola/epoch": epoch + 1,
                    "cola/lr": optimizer.param_groups[0]["lr"]
                }, step=epoch + 1)

        # === Evaluation ===
        classifier.eval()
        all_refs_tokens, all_hyps_tokens = [], []

        idx2label = getattr(train_dataset, "idx2label", None)
        if idx2label is None:
            raise RuntimeError("train_dataset is missing 'idx2label' for decoding.")

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="[COLA-CTC Evaluation]"):
                audio = batch["audio"].to(self.device)
                audio_lengths = batch["audio_lengths"].to(self.device)

                # Padded labels (+1 shift; 0 is blank/pad). Use lengths to trim.
                labels_padded = batch["labels"].cpu().tolist()
                label_lengths = batch["label_lengths"].cpu().tolist()

                # Inference under autocast (no scaler/backward)
                with autocast(enabled=use_amp):
                    log_probs, out_lengths = classifier(audio, audio_lengths)

                preds = torch.argmax(log_probs, dim=-1).cpu().tolist()
                out_lengths = out_lengths.cpu().tolist()

                # Greedy CTC decoding with collapse + blank removal, and proper trimming
                for pred_seq, ref_seq, ref_len, out_len in zip(preds, labels_padded, label_lengths, out_lengths):
                    pred_seq = pred_seq[:out_len]  # Trim predictions

                    # Collapse repeats & remove blanks (blank_id=0)
                    hyp_ids = []
                    prev = None
                    for p in pred_seq:
                        if p == 0:
                            prev = p
                            continue
                        if p != prev:
                            hyp_ids.append(p)
                        prev = p

                    # ids→tokens by undoing the +1 shift (ids are >=1 here)
                    hyp_tokens = [idx2label[p - 1] for p in hyp_ids]
                    ref_ids = ref_seq[:ref_len]
                    ref_tokens = [idx2label[r - 1] for r in ref_ids]

                    all_hyps_tokens.append(hyp_tokens)
                    all_refs_tokens.append(ref_tokens)

        # -------- DROP sil + closures before scoring --------
        DROP_TOKENS = {"sil", "tcl", "kcl", "pcl", "dcl", "gcl", "bcl"}

        def _filter(tokens):
            return [t for t in tokens if t not in DROP_TOKENS]

        refs_filt = [_filter(r) for r in all_refs_tokens]
        hyps_filt = [_filter(h) for h in all_hyps_tokens]

        # Stringify for token-level WER (phones, post-filter)
        ref_texts = [" ".join(r) for r in refs_filt]
        hyp_texts = [" ".join(h) for h in hyps_filt]
        wer_score = wer(ref_texts, hyp_texts)

        # True PER (post-filter)
        per_numer = sum(edit_distance(r, h) for r, h in zip(refs_filt, hyps_filt))
        per_denom = sum(len(r) for r in refs_filt) if refs_filt else 1
        per_score = per_numer / per_denom
        # -----------------------------------------------------

        self.logger.info(f"📊 [COLA-CTC Evaluation] WER(phones,no_sil)={wer_score:.4f}, PER(no_sil)={per_score:.4f}")

        if self.wandb_logger.is_active:
            self.wandb_logger.log({
                "cola/test_wer_no_sil": wer_score,
                "cola/test_per_no_sil": per_score
            })


    def _evaluate_eat(
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
        Evaluation for EAT using CTC with mixed precision (AMP).
        Computes TRUE WER on words (character CTC with explicit SPACE token).
        """

        def normalize_sentence(text: str) -> str:
            # simple normalization aligned with dataset’s preprocessing
            text = text.lower().strip()
            text = " ".join(text.split())  # collapse whitespace
            return text

        def tokens_to_text(tokens):
            # tokens: list[str] of characters, including SPACE token " "
            return "".join(tokens)

        backbone = EATBackbone(self.model).to(self.device)
        feature_size = self.model.embed_dim

        classifier = EvaluateNet(
            backbone=backbone,
            feature_size=feature_size,
            num_classes=num_classes,   # vocab size incl. blank (blank=0)
            is_linear=freeze_backbone,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, classifier.parameters()),
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=1e-4
        )

        criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_ctc,   # unchanged
            num_workers=self.num_workers,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_ctc,   # unchanged
            num_workers=self.num_workers,
            pin_memory=True,
        )

        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(classifier)

        use_amp = (self.device.type == "cuda")
        scaler = GradScaler(enabled=use_amp)

        # === Training ===
        classifier.train()
        if freeze_backbone:
            classifier.backbone.eval()

        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"[EAT-CTC Training] Epoch {epoch+1}")
            for batch in pbar:
                if batch is None:
                    continue
                audio = batch["audio"].to(self.device)                    # (B, 1, T_pad)
                labels = batch["flat_labels"].to(self.device)             # values 1..C (0 is blank)
                label_lengths = batch["label_lengths"].to(self.device)    # (B,)
                audio_lengths = batch["audio_lengths"].to(self.device)    # (B,)

                with autocast(enabled=use_amp):
                    log_probs, output_lengths = classifier(audio, audio_lengths)

                loss = criterion(
                    log_probs.float().permute(1, 0, 2),  # (T, B, C)
                    labels,
                    output_lengths,
                    label_lengths
                )

                if torch.isnan(loss):
                    self.logger.error("❌ NaN loss detected!")
                    self.logger.error(f"log_probs: {log_probs.shape}, "
                                      f"labels: {labels.shape}, "
                                      f"input_lengths: {output_lengths}, "
                                      f"label_lengths: {label_lengths}")
                    continue

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

                # 🔹 NEW: update tqdm postfix with batch loss
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            self.logger.info(f"[EAT-CTC Train] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

            if self.wandb_logger.is_active:
                self.wandb_logger.log({
                    "eat/train_loss": loss.item(),
                    "eat/epoch": epoch + 1,
                    "eat/lr": optimizer.param_groups[0]["lr"]
                }, step=epoch + 1)

        # === Evaluation ===
        classifier.eval()

        # Shared vocab (train & test must match)
        idx2label = getattr(train_dataset, "idx2label", None)
        if idx2label is None:
            raise RuntimeError("train_dataset is missing 'idx2label' for decoding.")
        if hasattr(test_dataset, "label2idx") and hasattr(train_dataset, "label2idx"):
            assert test_dataset.label2idx == train_dataset.label2idx, "Train/Test vocab mismatch!"

        ref_sentences, hyp_sentences = [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="[EAT-CTC Evaluation]"):
                if batch is None:
                    continue
                audio = batch["audio"].to(self.device)
                audio_lengths = batch["audio_lengths"].to(self.device)

                # (optional) get reference labels (for audit)
                labels_padded = batch["labels"].cpu().tolist()
                label_lengths = batch["label_lengths"].cpu().tolist()

                with autocast(enabled=use_amp):
                    log_probs, out_lengths = classifier(audio, audio_lengths)

                preds = torch.argmax(log_probs, dim=-1).cpu().tolist()
                out_lengths = out_lengths.cpu().tolist()

                for pred_seq, ref_seq, ref_len, out_len in zip(preds, labels_padded, label_lengths, out_lengths):
                    # --- Greedy CTC collapse for hypothesis ---
                    pred_seq = pred_seq[:out_len]
                    hyp_ids = []
                    prev = None
                    for p in pred_seq:
                        if p == 0:         # blank
                            prev = p
                            continue
                        if p != prev:       # collapse repeats
                            hyp_ids.append(p)
                        prev = p

                    # map ids → characters (subtract 1 to undo +1 shift in collate)
                    hyp_chars = [idx2label[p - 1] for p in hyp_ids]      # p >= 1
                    ref_ids = ref_seq[:ref_len]                          # already +1 shifted
                    ref_chars = [idx2label[r - 1] for r in ref_ids]

                    # chars → sentence
                    hyp_text = normalize_sentence(tokens_to_text(hyp_chars))
                    ref_text = normalize_sentence(tokens_to_text(ref_chars))

                    hyp_sentences.append(hyp_text)
                    ref_sentences.append(ref_text)

        # TRUE word-level WER (jiwer splits on whitespace)
        wer_score = wer(ref_sentences, hyp_sentences)

        self.logger.info(f"📊 [EAT-CTC Evaluation] WER(words)={wer_score:.4f}")

        if self.wandb_logger.is_active:
            self.wandb_logger.log({"eat/test_wer": wer_score})

        torch.save(classifier.state_dict(), "EAT_Classifier.pth")




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
        Evaluate the current model using CTC (WER/PER) for the specified SSL method.

        Args:
            train_dataset (Dataset): Training dataset with audio + labels.
            test_dataset (Dataset): Test dataset with audio + labels.
            num_classes (int): Vocabulary size (including blank).
            batch_size (int): Batch size.
            lr (float): Learning rate.
            epochs (int): Number of training epochs.
            freeze_backbone (bool): Whether to freeze the backbone.
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
            case "eat":
                self._evaluate_eat(train_dataset, test_dataset, num_classes, batch_size, lr, epochs, freeze_backbone, **kwargs)
            case _:
                raise ValueError(f"❌ Unknown method '{self.method}' for evaluation.")

        self.logger.info(f"✅ Evaluation for '{self.method}' completed.")
        if self.wandb_logger.is_active:
            self.wandb_logger.log({f"{self.method}/status": "evaluation_complete"})
            self.wandb_logger.finish_run()




    def collate_ctc(self, batch):
        """
        Expects items with:
        - "audio": FloatTensor(1, T)
        - "labels": LongTensor(L) in 0..P-1  (P phones, no blank)
        Returns:
        - audio: (B,1,Tmax)
        - audio_lengths: (B,)
        - labels: (B,Lmax) in [0..P], +1 shift, 0 used for pad/blank
        - flat_labels: (sum L_b,) in [1..P]
        - label_lengths: (B,)
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence

        audios = [x["audio"] for x in batch]
        lengths = torch.tensor([a.size(-1) for a in audios], dtype=torch.long)
        Tmax = int(lengths.max().item())
        B = len(batch)

        # pad audio to Tmax
        audio_pad = []
        for a in audios:
            if a.size(-1) < Tmax:
                pad = torch.zeros(1, Tmax - a.size(-1), dtype=a.dtype)
                audio_pad.append(torch.cat([a, pad], dim=-1))
            else:
                audio_pad.append(a)
        audio = torch.stack(audio_pad, dim=0)  # (B,1,Tmax)

        # labels
        labels_list = [x["labels"] for x in batch]  # each in 0..P-1
        label_lengths = torch.tensor([l.numel() for l in labels_list], dtype=torch.long)

        # +1 shift so 0 is reserved for CTC blank/pad
        labels_shifted = [l + 1 for l in labels_list]
        labels_padded = pad_sequence(labels_shifted, batch_first=True, padding_value=0)  # (B,Lmax)
        flat_labels = torch.cat(labels_shifted, dim=0)  # (sum L_b,)

        return {
            "audio": audio,
            "audio_lengths": lengths,
            "labels": labels_padded,
            "flat_labels": flat_labels,
            "label_lengths": label_lengths,
        }



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
                f"⚠️  No valid checkpoints found for method '{self.method}' in {self.checkpoint_path}. Starting from scratch."
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
                f"⚠️  No epoch number found in the checkpoint name '{latest_ckpt}'. Resuming from epoch 1."
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













