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
from torch.utils.data import Subset, DataLoader, Dataset, RandomSampler
import logging
from torcheval.metrics.functional import (
    multiclass_accuracy,
)
from torch.optim import AdamW
from typing import Optional, Type, Dict, Any

# Assuming these imports are correctly set up in your library structure
from MK_SSL.utils import configure_logging, get_logger_handler
from MK_SSL.audio.models.utils import get_method
from MK_SSL.audio.models.modules.tools import PseudoLabelGenerator
from MK_SSL.audio.models.modules.utils import HuBERTWrapperDataset

from MK_SSL.utils import optimize_hyperparameters

# Import your WandbLogger utility
# Make sure your_library.wandb_utils is accessible, e.g., in the same directory
# or properly installed as part of your package.
from MK_SSL.utils import WandbLogger


class Trainer:

    def __init__(
        self,
        method: str,
        backbone: nn.Module,
        variant: str,
        save_dir: str = ".",
        checkpoint_interval: int = 10,
        reload_checkpoint: bool = False,
        configure_logger: bool = True,
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
        if configure_logger:
            configure_logging()

        self.logger = logging.getLogger(self.__class__.__name__)

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

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_workers = os.cpu_count()

        self.logger.info(
            "\n"
            "---------------- MK_SSL: Audio ----------------\n"
            f"Number of workers : {self.num_workers}\n"
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

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision_training)
        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)

        kmeans_clusters = kwargs.get(
            "kmeans_clusters", getattr(self.model, "num_clusters", 100)
        )
        sample_rate = kwargs.get("sample_rate", 16000)
        self.pseudo_label_generator = PseudoLabelGenerator(
            kmeans_clusters=kmeans_clusters,
            sample_rate=sample_rate,
            save_dir=os.path.join(self.save_dir, "hubert_pseudo_labels"),
            logger=self.logger,  # Pass logger to the generator
        )

        # --- W&B Logger Initialization ---
        # Combine trainer_config with any specific wandb_config provided
        # This allows the trainer's internal config to be logged by W&B
        trainer_internal_config = {
            "method": self.method,
            "variant": variant,
            "save_dir": save_dir,
            "checkpoint_interval": checkpoint_interval,
            "reload_checkpoint": reload_checkpoint,
            "mixed_precision_training": mixed_precision_training,
            "device": str(self.device),
            "num_workers": self.num_workers,
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


    def _train_wav2vec2(
        self,
        train_loader: DataLoader,
        optimizer,
        max_epochs: int,
        start_epoch: int = 0,
        val_loader: Optional[DataLoader] = None,
    ):
        """
        Trains the Wav2Vec2 model using the specified optimizer and data loader.

        Args:
            train_loader (DataLoader): PyTorch DataLoader for training data.
            optimizer (Optimizer): Optimizer instance for training.
            max_epochs (int): Total number of training epochs.
            start_epoch (int, optional): Epoch to start training from. Defaults to 0.
            val_loader (DataLoader, optional): PyTorch DataLoader for validation data.
        """
        self.logger.info(f"Starting training for Wav2Vec2 for {max_epochs} epochs.")
        self.model.train()

        # Watch the model with W&B if active
        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(self.model)

        for epoch in range(start_epoch, max_epochs):
            running_loss = 0.0
            pbar = tqdm(
                train_loader, desc=f"Wav2Vec2 Epoch {epoch+1}/{max_epochs}"
            )

            for batch_idx, batch in enumerate(pbar): # Added batch_idx for logging step
                audio = batch['audio'].to(self.device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    (
                        context_features,
                        quantized_targets,
                        perplexity,
                        time_mask_indices,
                    ) = self.model(audio)

                    loss = self.loss(
                        context=context_features,
                        quantized=quantized_targets,
                        perplexity=perplexity,
                        time_mask_indices=time_mask_indices,
                    )

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                running_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

                # Log batch loss to W&B
                if self.wandb_logger.is_active:
                    self.wandb_logger.log(
                        {"train/batch_loss": loss.item()},
                        step=epoch * len(train_loader) + batch_idx
                    )

            avg_loss = running_loss / len(train_loader)
            self.logger.info(
                f"[Wav2Vec2 - Epoch {epoch+1}] Train Loss: {avg_loss:.4f}"
            )

            # Log epoch-level train loss to W&B
            if self.wandb_logger.is_active:
                self.wandb_logger.log(
                    {"train/epoch_loss": avg_loss},
                    step=epoch + 1 # Use epoch number as step for epoch-level metrics
                )

            if (
                epoch + 1
            ) % self.checkpoint_interval == 0:
                model_path = os.path.join(
                    self.checkpoint_path,
                    f"{self.method}_model_{self.timestamp}_epoch{epoch+1}.pth",
                )
                torch.save(self.model.state_dict(), model_path)
                self.logger.info(f"Model checkpoint saved: {model_path}")
                # Save model checkpoint as W&B artifact
                if self.wandb_logger.is_active:
                    self.wandb_logger.save_artifact(
                        model_path,
                        name=f"{self.method}-model-epoch-{epoch+1}",
                        type="model",
                        metadata={"epoch": epoch+1, "loss": avg_loss}
                    )

            if val_loader:
                self._validate_wav2vec2(val_loader, epoch)

            if hasattr(self, "_optuna_trial"):
                self._optuna_trial.report(loss.item(), epoch)
                if self._optuna_trial.should_prune():
                    raise optuna.TrialPruned()                

        # Final checkpoint after all epochs
        final_path = os.path.join(
            self.checkpoint_path,
            f"{self.method}_model_{self.timestamp}_final.pth",
        )
        torch.save(self.model.state_dict(), final_path)
        self.logger.info(f"Final model checkpoint saved: {final_path}")
        # Save final model as W&B artifact
        if self.wandb_logger.is_active:
            self.wandb_logger.save_artifact(
                final_path,
                name=f"{self.method}-model-final",
                type="model",
                metadata={"epochs_trained": max_epochs, "final_loss": avg_loss}
            )
        self.logger.info("Wav2Vec2 training complete.")


    def _validate_wav2vec2(self, val_loader: DataLoader, epoch: int):
        """
        Performs validation for the Wav2Vec2 model.

        Args:
            val_loader (DataLoader): PyTorch DataLoader for validation data.
            epoch (int): Current epoch number for logging.
        """
        self.model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(
                val_loader, desc=f"Validation Wav2Vec2 Epoch {epoch+1}"
            )
            for batch in pbar:
                audio = batch['audio'].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    (
                        context_features,
                        quantized_targets,
                        perplexity,
                        time_mask_indices,
                    ) = self.model(audio)

                    loss = self.loss(
                        context=context_features,
                        quantized=quantized_targets,
                        perplexity=perplexity,
                        time_mask_indices=time_mask_indices,
                    )

                val_running_loss += loss.item()

            avg_val_loss = val_running_loss / len(val_loader)
            self.logger.info(
                f"[Wav2Vec2 - Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}"
            )
            # Log validation loss to W&B
            if self.wandb_logger.is_active:
                self.wandb_logger.log(
                    {"val/loss": avg_val_loss},
                    step=epoch + 1 # Use epoch number as step for epoch-level metrics
                )
        self.model.train()


    def _train_simclr(
        self,
        train_loader,
        optimizer,
        max_epochs: int,
        start_epoch: int = 0,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """
        Trains the SimCLR model using the specified optimizer and data loader.
    
        Args:
            train_loader (DataLoader): PyTorch DataLoader for training data.
            optimizer (Optimizer): Optimizer instance for training.
            max_epochs (int): Total number of training epochs.
            start_epoch (int, optional): Epoch to start training from. Defaults to 0.
        """
        if self.transformation is None:
            self.logger.error("Transformation not given!")
            raise ValueError("Transformation not given!")
    
        self.model.train()
        # Watch the model with W&B if active
        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(self.model)

        for epoch in range(start_epoch, max_epochs):
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"SimCLR Epoch {epoch+1}/{max_epochs}") # Changed desc
    
            for batch_idx, batch in enumerate(pbar): # Added batch_idx for logging step
                audio = batch['audio'].to(self.device)
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

                # Log batch loss to W&B
                if self.wandb_logger.is_active:
                    self.wandb_logger.log(
                        {"train/batch_loss": loss.item()},
                        step=epoch * len(train_loader) + batch_idx
                    )
    
            avg_loss = running_loss / len(train_loader)
            self.logger.info(f"[SimCLR - Epoch {epoch+1}] Train Loss: {avg_loss:.4f}") # Changed log message

            # Log epoch-level train loss to W&B
            if self.wandb_logger.is_active:
                self.wandb_logger.log(
                    {"train/epoch_loss": avg_loss},
                    step=epoch + 1
                )
    
            # Save checkpoint
            if (epoch + 1) % self.checkpoint_interval == 0:
                model_path = os.path.join(
                    self.checkpoint_path,
                    f"{self.method}_model_{self.timestamp}_epoch{epoch+1}.pth",
                )
                torch.save(self.model.state_dict(), model_path)
                self.logger.info(f"Model checkpoint saved: {model_path}")
                # Save model checkpoint as W&B artifact
                if self.wandb_logger.is_active:
                    self.wandb_logger.save_artifact(
                        model_path,
                        name=f"{self.method}-model-epoch-{epoch+1}",
                        type="model",
                        metadata={"epoch": epoch+1, "loss": avg_loss}
                    )
            
            if val_loader:
                self._validate_simclr(val_loader, epoch)

            if hasattr(self, "_optuna_trial"):
                self._optuna_trial.report(loss.item(), epoch)
                if self._optuna_trial.should_prune():
                    raise optuna.TrialPruned()                

        final_path = os.path.join(
            self.checkpoint_path,
            f"{self.method}_model_{self.timestamp}_final.pth", # Changed to final
        )
        torch.save(self.model.state_dict(), final_path)
        self.logger.info(f"Final model checkpoint saved: {final_path}")
        # Save final model as W&B artifact
        if self.wandb_logger.is_active:
            self.wandb_logger.save_artifact(
                final_path,
                name=f"{self.method}-model-final",
                type="model",
                metadata={"epochs_trained": max_epochs, "final_loss": avg_loss}
            )
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

    def _train_cola(
        self,
        train_loader,
        optimizer,
        max_epochs: int,
        start_epoch: int = 0,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """
        Trains the COLA model using the specified optimizer and data loader.

        Args:
            train_loader (DataLoader): PyTorch DataLoader for training data.
            optimizer (Optimizer): Optimizer instance for training.
            max_epochs (int): Total number of training epochs.
            start_epoch (int, optional): Epoch to start training from. Defaults to 0.
        """
        if self.transformation is None:
            self.logger.error(f"Transformation not given!")
            raise ValueError("Transformation not given!")

        self.model.train()
        # Watch the model with W&B if active
        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(self.model)

        for epoch in range(start_epoch, max_epochs):
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"COLA Epoch {epoch+1}/{max_epochs}") # Changed desc

            for batch_idx, batch in enumerate(pbar): # Added batch_idx for logging step

                audio = batch['audio'].to(self.device)
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

                # Log batch loss to W&B
                if self.wandb_logger.is_active:
                    self.wandb_logger.log(
                        {"train/batch_loss": loss.item()},
                        step=epoch * len(train_loader) + batch_idx
                    )

            avg_loss = running_loss / len(train_loader)
            self.logger.info(f"[COLA - Epoch {epoch+1}] Train Loss: {avg_loss:.4f}") # Changed log message

            # Log epoch-level train loss to W&B
            if self.wandb_logger.is_active:
                self.wandb_logger.log(
                    {"train/epoch_loss": avg_loss},
                    step=epoch + 1
                )

            # Save checkpoint
            if (
                epoch + 1
            ) % self.checkpoint_interval == 0:
                model_path = os.path.join(
                    self.checkpoint_path,
                    f"{self.method}_model_{self.timestamp}_epoch{epoch+1}.pth",
                )
                torch.save(self.model.state_dict(), model_path)
                self.logger.info(f"Model checkpoint saved: {model_path}")
                # Save model checkpoint as W&B artifact
                if self.wandb_logger.is_active:
                    self.wandb_logger.save_artifact(
                        model_path,
                        name=f"{self.method}-model-epoch-{epoch+1}",
                        type="model",
                        metadata={"epoch": epoch+1, "loss": avg_loss}
                    )

            if val_loader:
                self._validate_cola(val_loader, epoch)

            if hasattr(self, "_optuna_trial"):
                self._optuna_trial.report(loss.item(), epoch)
                if self._optuna_trial.should_prune():
                    raise optuna.TrialPruned()                

        final_path = os.path.join(
            self.checkpoint_path,
            f"{self.method}_model_{self.timestamp}_final.pth", # Changed to final
        )
        torch.save(self.model.state_dict(), final_path)
        self.logger.info(f"Final model checkpoint saved: {final_path}")
        # Save final model as W&B artifact
        if self.wandb_logger.is_active:
            self.wandb_logger.save_artifact(
                final_path,
                name=f"{self.method}-model-final",
                type="model",
                metadata={"epochs_trained": max_epochs, "final_loss": avg_loss}
            )
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

    def _train_hubert(
        self,
        train_loader_for_training: DataLoader,
        train_loader_full_dataset: DataLoader,
        optimizer,
        max_epochs: int,
        start_epoch: int = 0,
        start_iteration: int = 0,
        val_loader: Optional[DataLoader] = None,
        num_hubert_iterations: int = 5,
        pseudo_label_sample_ratio: float = 0.1,
        **kwargs,
    ):
        transformer_layer = kwargs.get(
            "transformer_layer", getattr(self.model, "extractor_layer", None)
        )
        if transformer_layer is None:
            self.logger.warning(
                "No 'transformer_layer' specified for HuBERT. Defaulting to model's internal default (e.g., last layer of encoder)."
            )

        # Watch the model with W&B if active (for the entire HuBERT training process)
        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(self.model)

        for iteration in range(start_iteration, num_hubert_iterations):
            self.logger.info(
                f"--- Starting HuBERT Iteration {iteration + 1}/{num_hubert_iterations} ---"
            )
            # Log current iteration to W&B config or summary
            if self.wandb_logger.is_active:
                self.wandb_logger.log({"hubert_iteration": iteration + 1})


            iteration_pseudo_labels_path = os.path.join(
                self.pseudo_label_generator.save_dir,
                f"pseudo_labels_iter_{iteration}.npy",
            )

            pseudo_labels_dict = {}
            if os.path.exists(iteration_pseudo_labels_path):
                self.logger.info(
                    f"Loading existing pseudo-labels for iteration {iteration} from {iteration_pseudo_labels_path}"
                )
                pseudo_labels_dict = np.load(
                    iteration_pseudo_labels_path, allow_pickle=True
                ).item()
            else:
                self.logger.info(
                    f"Generating pseudo-labels for iteration {iteration + 1} (This may take a while)..."
                )

                dataloader_for_clustering = None
                if iteration == 0:
                    self.logger.info(
                        "Using full dataset for pseudo-label generation in iteration 0 (MFCCs)."
                    )
                    dataloader_for_clustering = train_loader_full_dataset
                else:
                    self.logger.info(
                        f"Sampling {pseudo_label_sample_ratio * 100}% of the dataset for pseudo-label generation in iteration {iteration + 1}."
                    )
                    wrapped_dataset = (
                        train_loader_full_dataset.dataset
                    )

                    num_samples_to_sample = int(
                        len(wrapped_dataset) * pseudo_label_sample_ratio
                    )
                    if num_samples_to_sample == 0 and len(wrapped_dataset) > 0:
                        self.logger.warning(
                            "Calculated sample size is 0. Using at least 1 sample if dataset is not empty."
                        )
                        num_samples_to_sample = 1
                    elif num_samples_to_sample > len(wrapped_dataset):
                        self.logger.warning(
                            f"Calculated sample size {num_samples_to_sample} is greater than dataset size {len(wrapped_dataset)}. Using full dataset for sampling."
                        )
                        num_samples_to_sample = len(wrapped_dataset)

                    sampler = RandomSampler(
                        wrapped_dataset,
                        num_samples=num_samples_to_sample,
                        replacement=False,
                    )

                    dataloader_for_clustering = DataLoader(
                        wrapped_dataset,
                        batch_size=train_loader_full_dataset.batch_size,
                        sampler=sampler,
                        num_workers=train_loader_full_dataset.num_workers,
                        pin_memory=train_loader_full_dataset.pin_memory,
                    )
                    self.logger.info(
                        f"Created dataloader for clustering with {len(dataloader_for_clustering.sampler)} samples."
                    )

                pseudo_labels_dict = self.pseudo_label_generator.generate_pseudo_labels(
                    dataloader=dataloader_for_clustering,
                    model=self.model,
                    is_mfcc=(iteration == 0),
                    transformer_layer=transformer_layer,
                    device=self.device,
                )
                np.save(iteration_pseudo_labels_path, pseudo_labels_dict)
                self.logger.info(
                    f"Generated and saved pseudo-labels for iteration {iteration + 1}."
                )

            train_loader_for_training.dataset.set_pseudo_labels(pseudo_labels_dict)
            self.logger.info(
                f"Updated train_dataset with pseudo-labels for iteration {iteration + 1}."
            )

            self.logger.info(
                f"Starting model training for HuBERT Iteration {iteration + 1} for {max_epochs} epochs."
            )
            self.model.train()

            current_iter_start_epoch = (
                start_epoch if iteration == start_iteration else 0
            )

            for epoch in range(current_iter_start_epoch, max_epochs):
                running_loss = 0.0
                pbar = tqdm(
                    train_loader_for_training,
                    desc=f"HuBERT Iter {iteration+1}, Epoch {epoch+1}/{max_epochs}",
                )

                for batch_idx, batch in enumerate(pbar): # Added batch_idx
                    audio = batch["audio"].to(self.device)
                    pseudo_labels = batch["pseudo_labels"].to(self.device)

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                        logits, mask_indices, _, _ = self.model(audio)
                        loss = self.loss(logits, pseudo_labels, mask_indices)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()

                    running_loss += loss.item()
                    pbar.set_postfix({"loss": loss.item()})

                    # Log batch loss to W&B
                    if self.wandb_logger.is_active:
                        self.wandb_logger.log(
                            {"train/batch_loss": loss.item(),
                             f"train/iter_{iteration+1}_batch_loss": loss.item()},
                            step=epoch * len(train_loader_for_training) + batch_idx
                        )


                avg_loss = running_loss / len(train_loader_for_training)
                self.logger.info(
                    f"[HuBERT Iter {iteration+1} - Epoch {epoch+1}] Train Loss: {avg_loss:.4f}"
                )
                # Log epoch-level train loss to W&B
                if self.wandb_logger.is_active:
                    self.wandb_logger.log(
                        {"train/epoch_loss": avg_loss,
                         f"train/iter_{iteration+1}_epoch_loss": avg_loss},
                        step=epoch + 1
                    )

                if (epoch + 1) % self.checkpoint_interval == 0:
                    model_path = os.path.join(
                        self.checkpoint_path,
                        f"{self.method}_iter{iteration+1}_model_{self.timestamp}_epoch{epoch+1}.pth",
                    )
                    torch.save(self.model.state_dict(), model_path)
                    self.logger.info(f"Model checkpoint saved: {model_path}")
                    # Save model checkpoint as W&B artifact
                    if self.wandb_logger.is_active:
                        self.wandb_logger.save_artifact(
                            model_path,
                            name=f"{self.method}-iter{iteration+1}-model-epoch-{epoch+1}",
                            type="model",
                            metadata={"iteration": iteration+1, "epoch": epoch+1, "loss": avg_loss}
                        )

                if val_loader:
                    self._validate_hubert(val_loader, iteration, epoch)

                if hasattr(self, "_optuna_trial"):
                    self._optuna_trial.report(loss.item(), epoch)
                    if self._optuna_trial.should_prune():
                        raise optuna.TrialPruned()

            final_iteration_model_path = os.path.join(
                self.checkpoint_path,
                f"{self.method}_iter{iteration+1}_final_model_{self.timestamp}.pth",
            )
            torch.save(self.model.state_dict(), final_iteration_model_path)
            self.logger.info(
                f"Final model for HuBERT Iteration {iteration+1} saved: {final_iteration_model_path}"
            )
            # Save final model for this iteration as W&B artifact
            if self.wandb_logger.is_active:
                self.wandb_logger.save_artifact(
                    final_iteration_model_path,
                    name=f"{self.method}-iter{iteration+1}-final-model",
                    type="model",
                    metadata={"iteration": iteration+1, "epochs_trained": max_epochs, "final_loss": avg_loss}
                )

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

    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 16,
        start_epoch: int = 0,
        max_epochs: int = 100,
        start_iteration: int = 0,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        optimizer: str = "adamw",
        use_optuna: bool = False,
        n_trials: int = 20,
        tuning_max_epochs: int = 5, 
        **kwargs,
    ) -> None:
        """
        Main training loop to train the model using the given dataset and hyperparameters.

        Args:
            train_dataset (Dataset): Dataset object for training.
            batch_size (int, optional): Mini-batch size. Defaults to 16.
            start_epoch (int, optional): Epoch to resume training from. Defaults to 0.
            max_epochs (int, optional): Total number of epochs. Defaults to 100.
            start_iteration (int, optional): Iteration to resume HuBERT training from. Defaults to 0.
            lr (float, optional): Learning rate. Defaults to 1e-4.
            weight_decay (float, optional): Weight decay (L2 regularization). Defaults to 1e-2.
            optimizer (str, optional): Optimizer to use ('adam', 'sgd', or 'adamw'). Defaults to 'adamw'.
            **kwargs: Additional keyword arguments passed to optimizer or loss, or HuBERT specific.
        """
        # Initialize W&B run at the very beginning of the main train method
        # This ensures console output is captured from the start and the run is properly set up.
        self.wandb_logger.init_run()

        # Log initial hyperparameters to W&B config if active
        if self.wandb_logger.is_active:
            # Update W&B config with dynamic training parameters
            self.wandb_logger.current_run.config.update({
                "batch_size": batch_size,
                "start_epoch": start_epoch,
                "max_epochs": max_epochs,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "optimizer": optimizer,
                **kwargs # Include any other kwargs passed to train method
            })
            self.logger.info(f"W&B run initialized. View run at: {self.wandb_logger.current_run.url}")
        else:
            self.logger.info("W&B logging is not active for this run.")


        # Auto hyperparameter tuning
        if use_optuna:
            self.logger.info("🧪 Running Optuna for hyperparameter tuning...")
            
            best_params = optimize_hyperparameters(
                trainer=self,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                n_trials=n_trials,
                max_epochs=tuning_max_epochs,
            )
            self.logger.info(f"🌟 Best hyperparameters found: {best_params}")
            
            lr = best_params.get("lr", lr)
            batch_size = best_params.get("batch_size", batch_size)
            weight_decay = best_params.get("weight_decay", weight_decay)
            # optimizer = best_params.get("optimizer", optimizer)

            kwargs.update({k: v for k, v in best_params.items() if k not in {"lr", "batch_size", "weight_decay", "optimizer"}})

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
            )

            val_loader = None
            if val_dataset:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )

            self._train_wav2vec2(
                train_loader,
                optimizer,
                max_epochs,
                start_epoch,
                val_loader,
            )

        elif self.method == "hubert":
            wrapped_train_dataset = HuBERTWrapperDataset(
                train_dataset, logger=self.logger
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

            val_loader = None
            if val_dataset:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )

            self._train_hubert(
                train_loader_for_training=train_loader_for_training,
                train_loader_full_dataset=train_loader_for_pseudo_label_gen,
                optimizer=optimizer,
                max_epochs=max_epochs,
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
            )

        elif self.method == "simclr": # Added simclr training
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            val_loader = None
            if val_dataset:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
            self._train_simclr(
                train_loader,
                optimizer,
                max_epochs,
                start_epoch,
                val_loader,
            )

        elif self.method == "cola": # Added cola training
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            val_loader = None
            if val_dataset:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
            self._train_cola(
                train_loader,
                optimizer,
                max_epochs,
                start_epoch,
                val_loader,
            )

        else:
            raise NotImplementedError(
                f"Training not implemented for method: {self.method}"
            )

        # Finish W&B run at the very end of the main train method
        self.wandb_logger.finish_run()
        self.logger.info("Main training process completed and W&B run finalized.")


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

    def __del__(self):
        """
        Destructor for the Trainer class.
        Closes the TensorBoard writer if it exists.
        """
        if hasattr(self, "writer"):
            self.writer.close()

