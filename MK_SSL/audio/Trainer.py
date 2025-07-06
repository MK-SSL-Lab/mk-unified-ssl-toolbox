import os
import re
import torch
import numpy as np
from torch import nn
from tqdm.auto import tqdm
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Subset, DataLoader
import logging
from torcheval.metrics.functional import multiclass_accuracy
from torch.optim import AdamW # Or Adam, as per paper

from MK_SSL.utils import configure_logging  


from MK_SSL.audio.models.utils import get_method
from MK_SSL.audio.models.hubert import HuBERT, HubertConfig
from MK_SSL.audio.models.modules.losses import HuBERTLoss
from MK_SSL.audio.models.modules.tools import PseudoLabelGenerator

\







class Trainer:
    def __init__(
        self,
        method: str,
        backbone: nn.Module,
        variant: str,
        projection_dim: int,
        temperature: float,
        diversity_loss_weight: float,
        num_negatives: int,
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
            projection_dim (int): Output dimension of the projection head.
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

        try:
            method_cfg = get_method(self.method)
        except ValueError as e:
            self.logger.error(f"Method {self.method} not found in registry.")
            raise e

        self.model = method_cfg["model"](
            variant=variant,
            projection_dim=projection_dim,
            
            variant=config.variant,
            mask_prob=config.mask_prob,
            mask_length=config.mask_length,
            mask_channel_prob=config.mask_channel_prob,
            mask_channel_length=config.mask_channel_length,
            num_clusters=config.num_clusters # This is for the projection head output dim
            **kwargs
        ).to(self.device)

        self.loss = method_cfg["loss"](
            temperature=temperature,
            diversity_loss_weight=diversity_loss_weight,
            num_negatives=num_negatives,
            **kwargs
        ).to(self.device)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision_training)


    



    def _train_wav2vec2(self, train_loader, optimizer, max_epochs: int, start_epoch: int = 1) -> None:
        """
        Trains the Wav2Vec2 model using the specified optimizer and data loader.

        Args:
            train_loader (DataLoader): PyTorch DataLoader for training data.
            optimizer (Optimizer): Optimizer instance for training.
            max_epochs (int): Total number of training epochs.
            start_epoch (int, optional): Epoch to start training from. Defaults to 1.
        """
        self.model.train()
        for epoch in range(start_epoch, max_epochs + 1):
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
        config: HubertConfig,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader, # Optional, for evaluation during training
        audio_paths_for_kmeans: list, # List of audio paths for K-means fitting
        output_dir: str,
        device: torch.device,
    ):
        """
        Trains the HuBERT model iteratively.
    
        Args:
            config (HubertConfig): Configuration object for HuBERT and training parameters.
            train_dataloader (DataLoader): DataLoader for the training audio data.
            val_dataloader (DataLoader): DataLoader for validation data (optional).
            audio_paths_for_kmeans (list): List of audio file paths used for K-means clustering.
                                           This could be a subset of the training data.
            output_dir (str): Directory to save models, pseudo-labels, and logs.
            device (torch.device): Device to run training on (e.g., 'cuda' or 'cpu').
        """
    
        os.makedirs(output_dir, exist_ok=True)
        model = HuBERT(
            variant=config.variant,
            mask_prob=config.mask_prob,
            mask_length=config.mask_length,
            mask_channel_prob=config.mask_channel_prob,
            mask_channel_length=config.mask_channel_length,
            num_clusters=config.num_clusters # This is for the projection head output dim
        ).to(device)
    
        loss_fn = HuBERTLoss(reduction="mean") # From hubert_loss.py
        optimizer = AdamW(model.parameters(), lr=config.lr) # Using AdamW, common for transformers
    
    
        # Main Iteration Loop (Offline Clustering + Model Training)
        for iteration in range(config.iterations):
            print(f"\n--- Starting Iteration {iteration + 1}/{config.iterations} ---")
    
            # --- Phase 1: Offline Clustering (Pseudo-label Generation) ---
            print("Generating pseudo-labels...")
    
            # Determine input type for K-means based on iteration
            if iteration == 0 and config.init_from_mfcc:
                # First iteration uses MFCC features for K-means
                kmeans_input_type = "mfcc"
                kmeans_model = None # No HuBERT model needed for MFCC extraction
                kmeans_layer = None
            else:
                # Subsequent iterations use features from a trained HuBERT model
                kmeans_input_type = "transformer"
                kmeans_model = model # Use the current HuBERT model
                kmeans_layer = config.extractor_layer # The layer to extract features from (e.g., 6 for base, 9 for large)
                # Ensure model is in eval mode during feature extraction for clustering
                model.eval()
    
    
            pseudo_label_generator = PseudoLabelGenerator(
                input_type=kmeans_input_type,
                model=kmeans_model,
                transformer_layer=kmeans_layer,
                sample_rate=config.sample_rate,
                kmeans_clusters=config.num_clusters,
                save_dir=os.path.join(output_dir, f"pseudo_labels_iter_{iteration}")
            )
    
            # Fit K-means (randomly sample subset of audio_paths_for_kmeans if dataset is very large)
            print("Fitting K-means...")
            # Note: For large datasets, you would only sample a subset of audio_paths_for_kmeans here
            pseudo_label_generator.fit_kmeans(audio_paths_for_kmeans)
    
            # Generate and save pseudo-labels for the entire training dataset
            print("Generating and saving pseudo-labels for training data...")
            # This function would save the generated labels for each audio path.
            # You'd then need to modify your AudioDataset to load these labels.
            pseudo_label_generator.generate_labels(list(train_dataloader.dataset.audio_paths))
            # Important: Your DataLoader/Dataset needs to be updated here to load these newly generated labels.
            # This implies either re-initializing the DataLoader or having a mechanism to update labels.
            # For simplicity in this outline, assume the dataset can load labels from the save_dir.
    
    
            # --- Phase 2: HuBERT Model Training ---
            print(f"Training HuBERT model for {config.epochs} epochs...")
            model.train() # Set model to training mode
    
            for epoch in range(config.epochs):
                total_loss = 0
                num_batches = 0
                # Wrap train_dataloader with tqdm for progress bar
                for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
                    audio = batch["audio"].to(device) # Shape (B, T_audio_samples)
                    # Assumes your dataset now provides 'labels' generated by PseudoLabelGenerator
                    pseudo_labels = batch["labels"].to(device) # Shape (B, T_features)
    
                    optimizer.zero_grad()
    
                    # Forward pass
                    # model_output contains (logits, masked_indices, features_lengths, masked_lengths)
                    # as per the HuBERT forward method signature
                    logits, mask_indices, _, _ = model(audio)
    
                    # Compute loss only on masked positions
                    # logits: (B, T_features, num_clusters)
                    # pseudo_labels: (B, T_features)
                    # mask_indices: (B, T_features) boolean mask
                    loss = loss_fn(logits, pseudo_labels, mask_indices)
    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
    
                    total_loss += loss.item()
                    num_batches += 1
    
                avg_loss = total_loss / num_batches
                print(f"Iteration {iteration+1}, Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
    
                # --- (Optional) Validation/Evaluation ---
                # You would typically add validation here after each epoch or every few epochs
                # if val_dataloader is not None:
                #     model.eval()
                #     val_loss = 0
                #     with torch.no_grad():
                #         for batch in val_dataloader:
                #             audio = batch["audio"].to(device)
                #             pseudo_labels = batch["labels"].to(device)
                #             logits, mask_indices, _, _ = model(audio)
                #             loss = loss_fn(logits, pseudo_labels, mask_indices)
                #             val_loss += loss.item()
                #     avg_val_loss = val_loss / len(val_dataloader)
                #     print(f"Iteration {iteration+1}, Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")
                #     model.train()
    
                # Save checkpoint (optional)
                torch.save(model.state_dict(), os.path.join(output_dir, f"hubert_iter_{iteration}_epoch_{epoch}.pt"))
    
        print("\nHuBERT pre-training complete!")
        return model
    




    def train(
        self,
        train_dataset,
        batch_size: int = 16,
        start_epoch: int = 1,
        max_epochs: int = 100,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        optimizer: str = "adamw",
        hubert_config: HubertConfig | None = None,
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
            hubert_config (HubertConfig, optional): Configuration for HuBERT pretraining.
            **kwargs: Additional keyword arguments passed to optimizer or loss.
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
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
            start_epoch = self._reload_latest_checkpoint() + 1

        if self.method == "hubert":
            if hubert_config is None:
                hubert_config = HubertConfig()
            self._train_hubert(train_loader, hubert_config)
        elif self.method == "wav2vec2":
            self._train_wav2vec2(train_loader, optimizer, max_epochs, start_epoch)
        else:
            raise NotImplementedError(f"Training not implemented for method: {self.method}")


    def load_checkpoint(self, checkpoint_path: str):
        """
        Loads a model checkpoint from the given path.

        Args:
            checkpoint_path (str): Path to the checkpoint file (.pth).
        """
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.logger.info("Checkpoint loaded from: {}".format(checkpoint_path))


    def _reload_latest_checkpoint(self):
        """
        Reloads the most recent model checkpoint from the checkpoint directory.

        Returns:
            int: The epoch number from which training should resume.

        Raises:
            ValueError: If no valid checkpoint or epoch information is found.
        """
        checkpoints = os.listdir(self.checkpoint_path)
        sorted_checkpoints = sorted(
            [os.path.join(self.checkpoint_path, ckpt) for ckpt in checkpoints],
            key=os.path.getmtime,
        )

        if len(sorted_checkpoints) == 0:
            self.logger.error("No checkpoints found in the directory")
            raise ValueError("No checkpoints found in the directory")

        latest_ckpt = sorted_checkpoints[-1]
        self.load_checkpoint(latest_ckpt)

        match = re.search(r"epoch(\d+)", latest_ckpt)
        if match:
            epoch = int(match.group(1))
            self.logger.info(f"Reloaded checkpoint from epoch {epoch}")
        else:
            self.logger.error("No epoch number found in the checkpoint name.")
            raise ValueError("No epoch number found in the checkpoint name.")

        return epoch


    def __del__(self):
        """
        Destructor for the Trainer class.
        Closes the TensorBoard writer if it exists.
        """
        if hasattr(self, "writer"):
            self.writer.close()
