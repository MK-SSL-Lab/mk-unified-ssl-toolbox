import os
import re
import torch
import numpy as np
from torch import nn
from tqdm.auto import tqdm # Used for general progress bars (can be replaced by inner tqdm if preferred)
from tqdm import tqdm # Used for specific inner loop progress bars
from datetime import datetime
from torch.utils.data import Subset, DataLoader, Dataset, RandomSampler
import logging
from torcheval.metrics.functional import multiclass_accuracy # Only import if directly used
from torch.optim import AdamW # Or Adam, as per paper
from typing import Optional # Import Optional

from MK_SSL.utils import configure_logging  # Assuming this exists


from MK_SSL.audio.models.utils import get_method
from MK_SSL.audio.models.modules.tools import PseudoLabelGenerator
from MK_SSL.audio.models.modules.utils import HuBERTWrapperDataset # Import the new wrapper


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

        loss_args = {}

        if "params" in method_cfg:
            loss_args.update(method_cfg["default_params"])
        
        loss_args.update(kwargs)


        # --- Create Generic Model ---
        self.model = method_cfg["model"](**model_args)

        # --- Create Generic Loss ---
        self.loss = method_cfg["loss"](**loss_args)



        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision_training)
        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)


        kmeans_clusters = kwargs.get('kmeans_clusters', getattr(self.model, 'num_clusters', 100))
        sample_rate = kwargs.get('sample_rate', 16000)
        self.pseudo_label_generator = PseudoLabelGenerator(
            kmeans_clusters=kmeans_clusters,
            sample_rate=sample_rate,
            save_dir=os.path.join(self.save_dir, "hubert_pseudo_labels"),
            logger=self.logger # Pass logger to the generator
        )

    

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
        train_loader_for_training: DataLoader, # For actual training epochs
        train_loader_full_dataset: DataLoader, # For feature extraction for K-means (full dataset)
        optimizer,
        max_epochs: int,
        start_epoch: int = 0,
        start_iteration: int = 0,
        val_loader: Optional[DataLoader] = None,
        num_hubert_iterations: int = 5,
        pseudo_label_sample_ratio: float = 0.1, # New argument for sampling ratio
        **kwargs
    ):
        transformer_layer = kwargs.get('transformer_layer', getattr(self.model, 'extractor_layer', None))
        if transformer_layer is None:
            self.logger.warning("No 'transformer_layer' specified for HuBERT. Defaulting to model's internal default (e.g., last layer of encoder).")

        for iteration in range(start_iteration, num_hubert_iterations):
            self.logger.info(f"--- Starting HuBERT Iteration {iteration + 1}/{num_hubert_iterations} ---")

            iteration_pseudo_labels_path = os.path.join(
                self.pseudo_label_generator.save_dir, f"pseudo_labels_iter_{iteration}.npy"
            )

            pseudo_labels_dict = {}
            if os.path.exists(iteration_pseudo_labels_path):
                self.logger.info(f"Loading existing pseudo-labels for iteration {iteration} from {iteration_pseudo_labels_path}")
                pseudo_labels_dict = np.load(iteration_pseudo_labels_path, allow_pickle=True).item()
            else:
                self.logger.info(f"Generating pseudo-labels for iteration {iteration + 1} (This may take a while)...")

                dataloader_for_clustering = None
                if iteration == 0:
                    # Use the full dataset for the first iteration (MFCC-based clustering)
                    self.logger.info("Using full dataset for pseudo-label generation in iteration 0 (MFCCs).")
                    dataloader_for_clustering = train_loader_full_dataset
                else:
                    # For subsequent iterations, sample a subset of the dataset
                    self.logger.info(f"Sampling {pseudo_label_sample_ratio * 100}% of the dataset for pseudo-label generation in iteration {iteration + 1}.")
                    wrapped_dataset = train_loader_full_dataset.dataset # Get the HuBERTWrapperDataset
                    
                    # Determine subset size
                    num_samples_to_sample = int(len(wrapped_dataset) * pseudo_label_sample_ratio)
                    if num_samples_to_sample == 0 and len(wrapped_dataset) > 0:
                        self.logger.warning("Calculated sample size is 0. Using at least 1 sample if dataset is not empty.")
                        num_samples_to_sample = 1
                    elif num_samples_to_sample > len(wrapped_dataset):
                        self.logger.warning(f"Calculated sample size {num_samples_to_sample} is greater than dataset size {len(wrapped_dataset)}. Using full dataset for sampling.")
                        num_samples_to_sample = len(wrapped_dataset)

                    # Create a RandomSampler to select a subset of indices
                    # Ensure reproducibility if needed by setting a random seed before sampling
                    sampler = RandomSampler(wrapped_dataset, num_samples=num_samples_to_sample, replacement=False)
                    
                    # Create a DataLoader from the Subset
                    # This dataloader must use the original_idx for mapping
                    dataloader_for_clustering = DataLoader(
                        wrapped_dataset,
                        batch_size=train_loader_full_dataset.batch_size, # Use same batch size
                        sampler=sampler, # Use the sampler to get the subset
                        num_workers=train_loader_full_dataset.num_workers,
                        pin_memory=train_loader_full_dataset.pin_memory,
                    )
                    self.logger.info(f"Created dataloader for clustering with {len(dataloader_for_clustering.sampler)} samples.")


                pseudo_labels_dict = self.pseudo_label_generator.generate_pseudo_labels(
                    dataloader=dataloader_for_clustering, # Use the conditionally selected dataloader
                    model=self.model,
                    is_mfcc=(iteration == 0),
                    transformer_layer=transformer_layer,
                    device=self.device
                )
                np.save(iteration_pseudo_labels_path, pseudo_labels_dict)
                self.logger.info(f"Generated and saved pseudo-labels for iteration {iteration + 1}.")

            train_loader_for_training.dataset.set_pseudo_labels(pseudo_labels_dict)
            self.logger.info(f"Updated train_dataset with pseudo-labels for iteration {iteration + 1}.")

            self.logger.info(f"Starting model training for HuBERT Iteration {iteration + 1} for {max_epochs} epochs.")
            self.model.train()

            current_iter_start_epoch = start_epoch if iteration == start_iteration else 0

            for epoch in range(current_iter_start_epoch, max_epochs):
                running_loss = 0.0
                pbar = tqdm(train_loader_for_training, desc=f"HuBERT Iter {iteration+1}, Epoch {epoch+1}/{max_epochs}")

                for batch in pbar:
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

                avg_loss = running_loss / len(train_loader_for_training)
                self.logger.info(f"[HuBERT Iter {iteration+1} - Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

                if (epoch + 1) % self.checkpoint_interval == 0:
                    model_path = os.path.join(
                        self.checkpoint_path,
                        f"{self.method}_iter{iteration+1}_model_{self.timestamp}_epoch{epoch+1}.pth",
                    )
                    torch.save(self.model.state_dict(), model_path)
                    self.logger.info(f"Model checkpoint saved: {model_path}")

                if val_loader:
                    self._validate_hubert(val_loader, iteration, epoch)

            final_iteration_model_path = os.path.join(
                self.checkpoint_path,
                f"{self.method}_iter{iteration+1}_final_model_{self.timestamp}.pth",
            )
            torch.save(self.model.state_dict(), final_iteration_model_path)
            self.logger.info(f"Final model for HuBERT Iteration {iteration+1} saved: {final_iteration_model_path}")

        self.logger.info("HuBERT training complete across all specified iterations.")

 
    def _validate_hubert(self, val_loader: DataLoader, iteration: int, epoch: int):
        self.model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Validation HuBERT Iter {iteration+1}, Epoch {epoch+1}")
            for batch in pbar:
                audio = batch["audio"].to(self.device)
                # If validation also requires pseudo_labels from the dataset, the val_dataset would need a similar wrapper.
                # For simplicity, assuming validation uses fixed labels or is handled differently if no pseudo_labels are needed.
                # If validation involves pseudo-labels, make sure the val_loader also provides 'pseudo_labels'.
                # For now, fetching 'pseudo_labels' for validation, assuming it's available.
                pseudo_labels = batch["pseudo_labels"].to(self.device) 

                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    logits, mask_indices, _, _ = self.model(audio)
                    loss = self.loss(logits, pseudo_labels, mask_indices)

                val_running_loss += loss.item()

            avg_val_loss = val_running_loss / len(val_loader)
            self.logger.info(f"[HuBERT Iter {iteration+1} - Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}")
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
                    batch_size=batch_size, # Could be separate val_batch_size if needed
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
            )

            self._train_wav2vec2(
                train_loader,
                optimizer,
                max_epochs,
                start_epoch,
            )

        elif self.method == "hubert":
            # Wrap the user's dataset for HuBERT, without requiring specific __getitem__ output beyond audio tensor
            wrapped_train_dataset = HuBERTWrapperDataset(train_dataset, logger=self.logger)

            # Create a DataLoader for the initial pseudo-label generation phase
            # This DataLoader will return {"audio": audio_tensor, "original_idx": idx}
            train_loader_for_pseudo_label_gen = DataLoader(
                wrapped_train_dataset,
                batch_size=batch_size, # Use training batch size, or a larger one for faster feature extraction
                shuffle=False, # Order matters for consistent index mapping for pseudo-label gen
                num_workers=self.num_workers,
                pin_memory=True,
            )
            
            # The final DataLoader for actual model training. It will receive pseudo_labels later.
            # Keep shuffle=True for actual training.
            train_loader_for_training = DataLoader(
                wrapped_train_dataset,
                batch_size=batch_size,
                shuffle=True, # Shuffle for training epochs
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


            self._train_hubert(
                train_loader_for_training=train_loader_for_training, # Used for actual training
                train_loader_for_pseudo_label_gen=train_loader_for_pseudo_label_gen, # Used for pseudo-label generation
                optimizer=optimizer,
                max_epochs=max_epochs,
                start_epoch=start_epoch,
                val_loader=val_loader,
                start_iteration=start_iteration,
                num_hubert_iterations=kwargs.get("num_hubert_iterations", self.model.config.get('max_iterations', 2)),
                transformer_layer=kwargs.get("transformer_layer", self.model.config.get('extractor_layer', None)),
                pseudo_label_sample_ratio=kwargs.get("pseudo_label_sample_ratio", self.model.config.get('pseudo_label_sample_ratio', 0.1)) 
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