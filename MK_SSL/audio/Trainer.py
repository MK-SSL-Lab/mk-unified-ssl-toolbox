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
# from torcheval.metrics.functional import multiclass_accuracy # Only import if directly used
from torch.optim import AdamW # Or Adam, as per paper
from typing import Optional # Import Optional

from MK_SSL.utils import configure_logging  # Assuming this exists


from MK_SSL.audio.models.utils import get_method
from MK_SSL.audio.models.hubert import HuBERT, HubertConfig
from MK_SSL.audio.models.modules.losses import HuBERTLoss # Explicitly import HuBERTLoss
from MK_SSL.audio.models.modules.tools import PseudoLabelGenerator # Explicitly import PseudoLabelGenerator


class Trainer:
    def __init__(
        self,
        method: str,
        # Removed method-specific parameters like backbone, variant, projection_dim etc.
        # These will be passed either via hubert_config or directly to the train method.
        save_dir: str = ".",
        checkpoint_interval: int = 10,
        reload_checkpoint: bool = False,
        configure_logger: bool = True,
        verbose: bool = True,
        mixed_precision_training: bool = True,
        **kwargs, # Catch any extra parameters for flexibility
    ) -> None:
        """
        Initializes the Trainer class for audio self-supervised learning.

        Args:
            method (str): SSL method name (e.g., 'hubert', 'wav2vec2').
            save_dir (str, optional): Directory to save checkpoints and logs. Defaults to ".".
            checkpoint_interval (int, optional): Frequency (in epochs) to save model checkpoints. Defaults to 10.
            reload_checkpoint (bool, optional): Whether to reload the most recent checkpoint. Defaults to False.
            configure_logger (bool, optional): Whether to initialize logging. Defaults to True.
            verbose (bool, optional): Verbosity flag for logger level. Defaults to True.
            mixed_precision_training (bool, optional): Enable AMP mixed precision training. Defaults to True.
            **kwargs: Additional keyword arguments.
        """
        if configure_logger:
            configure_logging()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self.logger.info("Audio Trainer initialized.")

        self.method = method.lower()
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
        
        # Model and Loss are now initialized within the specific _train_ methods
        self.model: Optional[nn.Module] = None 
        self.loss: Optional[nn.Module] = None

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision_training)


    def _train_wav2vec2(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader], # Added val_loader for consistency
        optimizer_instance: torch.optim.Optimizer, # Renamed to avoid conflict
        max_epochs: int,
        start_epoch: int = 1,
        # Pass model-specific parameters to initialize self.model and self.loss
        variant: str = "base", # Example parameter, adjust based on actual Wav2Vec2 needs
        projection_dim: int = 256, # Example parameter
        temperature: float = 0.1, # Example parameter
        diversity_loss_weight: float = 0.1, # Example parameter
        num_negatives: int = 100, # Example parameter
        **kwargs,
    ) -> None:
        """
        Trains the Wav2Vec2 model using the specified optimizer and data loader.

        Args:
            train_loader (DataLoader): PyTorch DataLoader for training data.
            val_loader (Optional[DataLoader]): PyTorch DataLoader for validation data (optional).
            optimizer_instance (Optimizer): Optimizer instance for training.
            max_epochs (int): Total number of training epochs.
            start_epoch (int, optional): Epoch to start training from. Defaults to 1.
            variant (str): Wav2Vec2 model variant (e.g., 'base').
            projection_dim (int): Output dimension of the projection head.
            temperature (float): Temperature for contrastive loss.
            diversity_loss_weight (float): Weight for diversity loss.
            num_negatives (int): Number of negative samples.
            **kwargs: Additional keyword arguments passed to model or loss.
        """
        self.logger.info("Initializing Wav2Vec2 model and loss...")
        try:
            method_cfg = get_method(self.method)
        except ValueError as e:
            self.logger.error(f"Method {self.method} not found in registry.")
            raise e
        
        # Initialize self.model and self.loss for Wav2Vec2
        self.model = method_cfg["model"](
            variant=variant,
            projection_dim=projection_dim,
            **kwargs
        ).to(self.device)

        self.loss = method_cfg["loss"](
            temperature=temperature,
            diversity_loss_weight=diversity_loss_weight,
            num_negatives=num_negatives,
            **kwargs
        ).to(self.device)

        self.model.train()
        for epoch in range(start_epoch, max_epochs + 1):
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}")

            for batch in pbar:
                audio = batch["audio"].to(self.device)
                padding_mask = batch.get("padding_mask", None) # Get padding_mask if present

                self.model.zero_grad() # Use self.model.zero_grad() instead of optimizer.zero_grad() before scaler.scale

                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    features, quantized = self.model(audio, padding_mask=padding_mask) # Pass padding_mask
                    loss = self.loss(features, quantized)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer_instance)
                self.scaler.update()

                running_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

            avg_loss = running_loss / len(train_loader)
            self.logger.info(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")

            # --- Validation for Wav2Vec2 ---
            if val_loader:
                self.logger.info(f"Running validation for Epoch {epoch}...")
                self.model.eval()
                val_running_loss = 0.0
                with torch.no_grad():
                    for val_batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                        val_audio = val_batch["audio"].to(self.device)
                        val_padding_mask = val_batch.get("padding_mask", None)
                        with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                            val_features, val_quantized = self.model(val_audio, padding_mask=val_padding_mask)
                            val_loss = self.loss(val_features, val_quantized).item()
                        val_running_loss += val_loss
                avg_val_loss = val_running_loss / len(val_loader)
                self.logger.info(f"Epoch {epoch}: Average Validation Loss = {avg_val_loss:.4f}")
                self.model.train() # Set back to train mode

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
        self, # Add self as it's a method
        config: HubertConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader], # Make optional
        audio_paths_for_kmeans: list, # List of audio paths for K-means fitting
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
        # Ensure output directories exist (Trainer __init__ already does this for self.save_dir)
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.logger.info("Initializing HuBERT model...")
        # Initialize HuBERT model and store it as self.model
        self.model = HuBERT(
            variant=config.variant,
            mask_prob=config.mask_prob,
            mask_length=config.mask_length,
            mask_channel_prob=config.mask_channel_prob,
            mask_channel_length=config.mask_channel_length,
            num_clusters=config.num_clusters
        ).to(self.device) # Use self.device

        # Initialize HuBERT loss and store it as self.loss
        self.loss = HuBERTLoss(reduction="mean") # HuBERT loss is specific and directly imported
        optimizer = AdamW(self.model.parameters(), lr=config.lr) # Optimizer for HuBERT


        # Main Iteration Loop
        for iteration in range(config.iterations):
            self.logger.info(f"\n--- Starting Iteration {iteration + 1}/{config.iterations} ---")

            # --- Phase 1: Offline Clustering (Pseudo-label Generation) ---
            self.logger.info("Generating pseudo-labels...")

            # Determine input type for K-means based on iteration
            if iteration == 0 and config.init_from_mfcc:
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
                sample_rate=config.sample_rate,
                kmeans_clusters=config.num_clusters,
                save_dir=os.path.join(self.save_dir, f"pseudo_labels_iter_{iteration}") # Use self.save_dir
            )

            self.logger.info("Fitting K-means...")
            pseudo_label_generator.fit_kmeans(audio_paths_for_kmeans)

            self.logger.info("Generating and saving pseudo-labels for training data...")
            # Note: Your DataLoader/Dataset needs a mechanism to load these newly generated labels.
            # This is crucial for the next training phase.
            pseudo_label_generator.generate_labels(list(train_dataloader.dataset.audio_paths))


            # --- Phase 2: HuBERT Model Training ---
            self.logger.info(f"Training HuBERT model for {config.epochs} epochs...")
            self.model.train() # Set model to training mode

            for epoch in range(config.epochs):
                total_loss = 0
                num_batches = 0
                pbar = tqdm(train_dataloader, desc=f"Iteration {iteration+1}, Epoch {epoch+1}")

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
                if val_dataloader:
                    self.logger.info(f"Running validation for Iteration {iteration+1}, Epoch {epoch+1}...")
                    self.model.eval() # Set model to eval mode for validation
                    val_total_loss = 0
                    with torch.no_grad():
                        for val_batch in tqdm(val_dataloader, desc=f"Validation Iteration {iteration+1}, Epoch {epoch+1}"):
                            val_audio = val_batch["audio"].to(self.device)
                            val_labels = val_batch["labels"].to(self.device)
                            val_padding_mask = val_batch.get("padding_mask", None) # Get padding_mask if present

                            val_logits, val_mask_indices, _, _ = self.model(val_audio, padding_mask=val_padding_mask)
                            val_loss = self.loss(val_logits, val_labels, val_mask_indices).item() # Use self.loss
                            val_total_loss += val_loss
                    avg_val_loss = val_total_loss / len(val_dataloader)
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
        train_dataset: Dataset, # Explicitly type as Dataset
        batch_size: int = 16,
        start_epoch: int = 1, # More relevant for fine-tuning or fixed-epoch training
        max_epochs: int = 100, # More relevant for fine-tuning or fixed-epoch training
        lr: float = 1e-4, # Default LR for optimizers if not HuBERT
        weight_decay: float = 1e-2,
        optimizer: str = "adamw", # Default optimizer type if not HuBERT
        hubert_config: Optional[HubertConfig] = None, # Configuration for HuBERT pretraining
        val_dataset: Optional[Dataset] = None, # Added optional validation dataset
        # Pass Wav2Vec2 specific parameters directly to train, then to _train_wav2vec2
        wav2vec2_variant: str = "base",
        wav2vec2_projection_dim: int = 256,
        wav2vec2_temperature: float = 0.1,
        wav2vec2_diversity_loss_weight: float = 0.1,
        wav2vec2_num_negatives: int = 100,
        **kwargs, # Catch additional keyword arguments
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
            hubert_config (Optional[HubertConfig]): Configuration for HuBERT pretraining.
            val_dataset (Optional[Dataset]): Dataset object for validation. Defaults to None.
            wav2vec2_variant (str): Wav2Vec2 model variant (e.g., 'base').
            wav2vec2_projection_dim (int): Output dimension of Wav2Vec2 projection head.
            wav2vec2_temperature (float): Temperature for Wav2Vec2 contrastive loss.
            wav2vec2_diversity_loss_weight (float): Weight for Wav2Vec2 diversity loss.
            wav2vec2_num_negatives (int): Number of negative samples for Wav2Vec2 loss.
            **kwargs: Additional keyword arguments.
        """
        self.logger.info(f"Starting training for method: {self.method}")

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

        if self.method == "hubert":
            if hubert_config is None:
                self.logger.info("No HuBERTConfig provided, using default configuration.")
                hubert_config = HubertConfig() # Use default config if not provided

            # Ensure train_dataset provides audio_paths for K-means
            if not hasattr(train_dataset, 'audio_paths'):
                self.logger.error("`train_dataset` must have an `audio_paths` attribute (list of file paths) for HuBERT's K-means clustering.")
                raise ValueError("`train_dataset` must have an `audio_paths` attribute (list of file paths) for HuBERT's K-means clustering.")
            audio_paths_for_kmeans = list(train_dataset.audio_paths)

            self._train_hubert(
                config=hubert_config,
                train_dataloader=train_loader,
                val_dataloader=val_loader, # Pass the new val_loader
                audio_paths_for_kmeans=audio_paths_for_kmeans,
                # output_dir and device are now self attributes, no need to pass explicitly
            )
            self.logger.info("HuBERT training complete.")

        elif self.method == "wav2vec2":
            # Model and Loss for Wav2Vec2 are initialized inside _train_wav2vec2
            # Optimizer is also created inside train method here for wav2vec2, then passed.
            optimizer_instance = None
            try:
                match optimizer.lower():
                    case "adam":
                        optimizer_instance = torch.optim.Adam(
                            list(self.model.parameters()), # Assumes self.model is initialized elsewhere or by _train_wav2vec2 itself
                            lr=lr,
                            weight_decay=weight_decay,
                        )
                    case "sgd":
                        optimizer_instance = torch.optim.SGD(
                            list(self.model.parameters()),
                            lr=lr,
                            weight_decay=weight_decay,
                        )
                    case "adamw":
                        optimizer_instance = torch.optim.AdamW(
                            list(self.model.parameters()),
                            lr=lr,
                            weight_decay=weight_decay,
                        )
                    case _:
                        self.logger.error(f"Unsupported Optimizer: {optimizer}")
                        raise ValueError(f"Optimizer {optimizer} not supported")
            except Exception as e:
                self.logger.error(f"Error initializing optimizer for {self.method}: {e}")
                raise

            if self.reload_checkpoint:
                start_epoch = self._reload_latest_checkpoint() + 1

            self._train_wav2vec2(
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer_instance=optimizer_instance,
                max_epochs=max_epochs,
                start_epoch=start_epoch,
                # Pass Wav2Vec2 specific parameters
                variant=wav2vec2_variant,
                projection_dim=wav2vec2_projection_dim,
                temperature=wav2vec2_temperature,
                diversity_loss_weight=wav2vec2_diversity_loss_weight,
                num_negatives=wav2vec2_num_negatives,
                **kwargs,
            )
            self.logger.info("Wav2Vec2 training complete.")
        else:
            self.logger.error(f"Training not implemented for method: {self.method}")
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
            self.logger.info(f"Reloaded checkpoint from epoch {epoch}")
        else:
            self.logger.warning(f"No epoch number found in the checkpoint name '{latest_ckpt}'. Resuming from epoch 1.")
            epoch = 1 # Default to epoch 1 if not found

        return epoch


    def __del__(self):
        """
        Destructor for the Trainer class.
        Closes the TensorBoard writer if it exists.
        """
        # Assuming writer is an attribute if used
        if hasattr(self, "writer"):
            self.writer.close()