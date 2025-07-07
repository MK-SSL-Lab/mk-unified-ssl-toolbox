import os
import re
import torch
import numpy as np
from torch import nn
from tqdm.auto import tqdm # Used for general progress bars (can be replaced by inner tqdm if preferred)
from tqdm import tqdm # Used for specific inner loop progress bars
from datetime import datetime
from torch.utils.data import Subset, DataLoader, Dataset # Import Dataset
import logging
from torcheval.metrics.functional import multiclass_accuracy # Only import if directly used
from torch.optim import AdamW # Or Adam, as per paper
from typing import Optional # Import Optional

from MK_SSL.utils import configure_logging  # Assuming this exists


from MK_SSL.audio.models.utils import get_method
from MK_SSL.audio.models.modules.tools import PseudoLabelGenerator


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

        **kwargs,
    ) -> None:
        """
        Initializes the Trainer class for audio self-supervised learning.

        Args:
            method (str): SSL method name (e.g., 'wav2vec2').
            backbone (nn.Module): Model backbone architecture (e.g., ConvNet, Transformer).
            variant (str): Architecture variant (e.g., 'base', 'large') used for model config.
            temperature (float): Temperature parameter for contrastive loss.
            diversity_loss_weight (float): Weight for the diversity loss component.
            num_negatives (int): Number of negative samples used in contrastive loss.
            save_dir (str, optional): Directory to save checkpoints and logs. Defaults to ".".
            checkpoint_interval (int, optional): Frequency (in epochs) to save model checkpoints. Defaults to 10.
            reload_checkpoint (bool, optional): Whether to reload the most recent checkpoint. Defaults to False.
            configure_logger (bool, optional): Whether to initialize logging. Defaults to True.
            verbose (bool, optional): Verbosity flag for logger level. Defaults to True.
            mixed_precision_training (bool, optional): Enable AMP mixed precision training. Defaults to True.
            **kwargs: Additional keyword arguments passed to the model or loss.
        """
        if configure_logger:
            configure_logging()

        self.logger = logging.getLogger(self.__class__.__name__)
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

        loss_args = {

        }
        if "params" in method_cfg:
            loss_args.update(method_cfg["default_params"])
        
        loss_args.update(kwargs)


        # --- Create Generic Model ---
        self.model = method_cfg["model"](
            **model_args
        ).to(self.device)

        # --- Create Generic Loss ---
        self.loss = method_cfg["loss"](
            **loss_args
        ).to(self.device)



        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision_training)



    

    def _train_wav2vec2(
            self,
            train_loader,
            optimizer,
            max_epochs: int,
            start_epoch: int = 0
        ) -> None:
        """
        Trains the Wav2Vec2 model using the specified optimizer and data loader.

        Args:
            train_loader (DataLoader): PyTorch DataLoader for training data.
            optimizer (Optimizer): Optimizer instance for training.
            max_epochs (int): Total number of training epochs.
            start_epoch (int, optional): Epoch to start training from. Defaults to 1.
        """
        self.model.train()
        for epoch in range(start_epoch, max_epochs):
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}")

            for batch in pbar:
                audio = batch["audio"].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    features, quantized = self.model(audio)
                    loss = self.loss(features, quantized)

                optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                running_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

            avg_loss = running_loss / len(train_loader)
            self.logger.info(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")

            if epoch % self.checkpoint_interval == 0:
                model_path = os.path.join(
                    self.checkpoint_path,
                    f"{self.method}_model_{self.timestamp}_epoch{epoch}.pth",
                )
                torch.save(self.model.state_dict(), model_path)
                self.logger.info(f"Model checkpoint saved: {model_path}")

        final_path = os.path.join(
            self.checkpoint_path,
            f"{self.method}_model_{self.timestamp}_epoch{max_epochs}.pth",
        )
        torch.save(self.model.state_dict(), final_path)
        self.logger.info(f"Final model checkpoint saved: {final_path}")




    def _train_hubert(
        self, 
        audio_paths_for_kmeans: list,
        train_loader: DataLoader,
        optimizer ,
        max_epochs: int,
        start_epoch: int = 0,
        start_iteration: int = 0,
        val_loader: Optional[DataLoader] = None,  
    ):
        """
        Trains the HuBERT model iteratively.

        Args:
            config (HubertConfig): Configuration object for HuBERT and training parameters.
            train_dataloader (DataLoader): DataLoader for the training audio data.
            val_dataloader (Optional[DataLoader]): DataLoader for validation data (optional).
            audio_paths_for_kmeans (list): List of audio file paths used for K-means clustering.
                                           This could be a subset of the training data.
        """

        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.logger.info("Initializing HuBERT model...")



        num_iterations = self.model.config['iterations']

        # Main Iteration Loop
        for iteration in range(start_iteration, num_iterations):
            self.logger.info(f"\n--- Starting Iteration {iteration + 1}/{num_iterations} ---")

            # --- Phase 1: Offline Clustering (Pseudo-label Generation) ---
            self.logger.info("Generating pseudo-labels...")

            # Determine input type for K-means based on iteration
            if iteration == 0 and self.model.init_from_mfcc:
                kmeans_input_type = "mfcc"
                kmeans_model = None
                kmeans_layer = None
            else:
                kmeans_input_type = "transformer"
                kmeans_model = self.model # Use the current HuBERT model (self.model)
                kmeans_layer = self.model.model_config['extractor_layer']
                self.model.eval() # Ensure model is in eval mode during feature extraction for clustering


            pseudo_label_generator = PseudoLabelGenerator(
                input_type=kmeans_input_type,
                model=kmeans_model,
                transformer_layer=kmeans_layer,
                sample_rate=self.model.sample_rate,
                kmeans_clusters=self.model.num_clusters,
                save_dir=os.path.join(self.save_dir, f"pseudo_labels_iter_{iteration}") # Use self.save_dir
            )

            self.logger.info("Fitting K-means...")
            pseudo_label_generator.fit_kmeans(audio_paths_for_kmeans)

            self.logger.info("Generating and saving pseudo-labels for training data...")
            # Note: Your DataLoader/Dataset needs a mechanism to load these newly generated labels.
            # This is crucial for the next training phase.
            pseudo_label_generator.generate_labels(list(train_loader.dataset.audio_paths))


            # --- Phase 2: HuBERT Model Training ---
            self.logger.info(f"Training HuBERT model for {max_epochs} epochs...")
            self.model.train() # Set model to training mode

            for epoch in range(start_epoch, max_epochs):
                total_loss = 0
                num_batches = 0
                pbar = tqdm(train_loader, desc=f"Iteration {iteration+1}, Epoch {epoch+1}")

                for batch_idx, batch in enumerate(pbar):
                    audio = batch["audio"].to(self.device)
                    # Assumes your dataset now provides 'labels' generated by PseudoLabelGenerator
                    pseudo_labels = batch["labels"].to(self.device)
                    padding_mask = batch.get("padding_mask", None) # Get padding_mask if present

                    optimizer.zero_grad() # Reset gradients

                    with torch.cuda.amp.autocast(enabled=self.mixed_precision_training): # Enable AMP
                        # Forward pass
                        logits, mask_indices, _, _ = self.model(audio, padding_mask=padding_mask) # Pass padding_mask

                        # Compute loss only on masked positions
                        loss = self.loss(logits, pseudo_labels, mask_indices) # Use self.loss

                    self.scaler.scale(loss).backward() # Scale loss and backpropagate
                    self.scaler.step(optimizer) # Update optimizer weights
                    self.scaler.update() # Update the scaler for next iteration

                    total_loss += loss.item()
                    num_batches += 1
                    pbar.set_postfix({"loss": loss.item()}) # Update tqdm postfix

                avg_loss = total_loss / num_batches
                self.logger.info(f"Iteration {iteration+1}, Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")

                # --- Validation/Evaluation ---
                if val_loader:
                    self.logger.info(f"Running validation for Iteration {iteration+1}, Epoch {epoch+1}...")
                    self.model.eval() # Set model to eval mode for validation
                    val_total_loss = 0
                    with torch.no_grad():
                        for val_batch in tqdm(val_loader, desc=f"Validation Iteration {iteration+1}, Epoch {epoch+1}"):
                            val_audio = val_batch["audio"].to(self.device)
                            val_labels = val_batch["labels"].to(self.device)
                            val_padding_mask = val_batch.get("padding_mask", None) # Get padding_mask if present

                            val_logits, val_mask_indices, _, _ = self.model(val_audio, padding_mask=val_padding_mask)
                            val_loss = self.loss(val_logits, val_labels, val_mask_indices).item() # Use self.loss
                            val_total_loss += val_loss
                    avg_val_loss = val_total_loss / len(val_loader)
                    self.logger.info(f"Iteration {iteration+1}, Epoch {epoch+1}: Average Validation Loss = {avg_val_loss:.4f}")
                    self.model.train() # Set back to train mode

                # Save checkpoint
                model_path = os.path.join(
                    self.checkpoint_path,
                    f"hubert_iter_{iteration}_epoch_{epoch}.pth", # Consistent naming
                )
                torch.save(self.model.state_dict(), model_path)
                self.logger.info(f"Model checkpoint saved: {model_path}")

        self.logger.info("\nHuBERT pre-training complete!")
        # The trained model is now stored in self.model
        # return self.model # No need to return here, as it's self.model



    def train(
        self,
        train_dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 16,
        start_epoch: int = 0,
        max_epochs: int = 100,
        start_iteration: int = 0,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        optimizer: str = "adamw",
        **kwargs,
    ) -> None:
        """
        Main training loop to train the model using the given dataset and hyperparameters.

        Args:
            train_dataset (Dataset): Dataset object for training.
            batch_size (int, optional): Mini-batch size. Defaults to 16.
            start_epoch (int, optional): Epoch to resume training from. Defaults to 1.
            max_epochs (int, optional): Total number of epochs. Defaults to 100.
            lr (float, optional): Learning rate. Defaults to 1e-4.
            weight_decay (float, optional): Weight decay (L2 regularization). Defaults to 1e-2.
            optimizer (str, optional): Optimizer to use ('adam', 'sgd', or 'adamw'). Defaults to 'adamw'.
            **kwargs: Additional keyword arguments passed to optimizer or loss.
        """
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
                batch_size=batch_size, # Could be separate val_batch_size if needed
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

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
            self._train_wav2vec2(
                train_loader,
                optimizer,
                max_epochs,
                start_epoch,
            )

        elif self.method == "hubert":


            if not hasattr(train_dataset, 'audio_paths'):
                self.logger.error("`train_dataset` must have an `audio_paths` attribute (list of file paths) for HuBERT's K-means clustering.")
                raise ValueError("`train_dataset` must have an `audio_paths` attribute (list of file paths) for HuBERT's K-means clustering.")
            audio_paths_for_kmeans = list(train_dataset.audio_paths)

            self._train_hubert(
                train_loader=train_loader,
                optimizer=optimizer,
                max_epochs=max_epochs,
                start_epoch=start_epoch,
                val_loader=val_loader, # Pass the new val_loader
                audio_paths_for_kmeans=audio_paths_for_kmeans,
            )


        else:
            raise NotImplementedError(f"Training not implemented for method: {self.method}")



    def load_checkpoint(self, checkpoint_path: str):
        """
        Loads a model checkpoint from the given path.
        Assumes self.model is already initialized and matches the checkpoint's state_dict.

        Args:
            checkpoint_path (str): Path to the checkpoint file (.pth).
        """
        if self.model is None:
            self.logger.error("Model must be initialized before loading a checkpoint.")
            raise RuntimeError("Model must be initialized before loading a checkpoint.")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
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
        # Filter for files ending with .pth and potentially related to the current method
        method_prefix = self.method + "_model_"
        filtered_checkpoints = [
            ckpt for ckpt in checkpoints if ckpt.endswith(".pth") and ckpt.startswith(method_prefix)
        ]

        if not filtered_checkpoints:
            self.logger.warning(f"No valid checkpoints found for method '{self.method}' in {self.checkpoint_path}. Starting from scratch.")
            return 0 # Indicate starting from epoch 0 or 1, depending on convention

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
            self.logger.warning(f"No epoch number found in the checkpoint name '{latest_ckpt}'. Resuming from epoch 1.")
            epoch = 0 # Default to epoch 1 if not found

        return epoch


    def __del__(self):
        """
        Destructor for the Trainer class.
        Closes the TensorBoard writer if it exists.
        """
        # Assuming writer is an attribute if used
        if hasattr(self, "writer"):
            self.writer.close()