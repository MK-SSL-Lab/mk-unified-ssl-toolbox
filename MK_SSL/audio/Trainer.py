import os
import re
import torch
import numpy as np
from torch import nn
from tqdm.auto import tqdm
from datetime import datetime
from torch.utils.data import Subset, DataLoader
import logging
from torcheval.metrics.functional import multiclass_accuracy
from MK_SSL.utils import configure_logging  

from MK_SSL.audio.models.utils import get_method
from MK_SSL.audio.models.hubert import HuBERT, HubertConfig
from MK_SSL.audio.models.modules.losses import HuBERTLoss
from MK_SSL.audio.models.modules.tools import PseudoLabelGenerator


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

    def _train_hubert(self, dataloader, config: HubertConfig) -> None:
        """Run multi-iteration HuBERT pretraining."""
        hubert_dir = os.path.join(self.checkpoint_path, "hubert")
        os.makedirs(hubert_dir, exist_ok=True)

        model: HuBERT | None = None

        for itr in range(1, config.iterations + 1):
            if itr == 1:
                clusters = 100
                label_gen = PseudoLabelGenerator(
                    input_type="mfcc",
                    sample_rate=config.sample_rate,
                    kmeans_clusters=clusters,
                )
                model = HuBERT(
                    variant=config.variant,
                    mask_prob=config.mask_prob,
                    mask_length=config.mask_length,
                    mask_channel_prob=config.mask_channel_prob,
                    mask_channel_length=config.mask_channel_length,
                ).to(self.device)
            else:
                clusters = 500
                label_gen = PseudoLabelGenerator(
                    input_type="transformer",
                    model=model,
                    transformer_layer=config.extractor_layer,
                    sample_rate=config.sample_rate,
                    kmeans_clusters=clusters,
                )

            features = []
            for batch in dataloader:
                wave = batch[0].to(self.device) if not isinstance(batch, dict) else batch.get("audio").to(self.device)
                with torch.no_grad():
                    feat = label_gen.extract_features(wave)
                features.append(feat.cpu().numpy())

            flat = np.concatenate(features, axis=0)
            if itr > 1:
                keep = np.random.choice(len(flat), max(1, int(0.1 * len(flat))), replace=False)
                flat = flat[keep]
            label_gen.kmeans.fit(flat)
            label_gen.fitted = True

            head = nn.Linear(model.encoder.embed_dim, clusters).to(self.device)
            criterion = HuBERTLoss().to(self.device)
            optimizer = torch.optim.AdamW(
                list(model.parameters()) + list(head.parameters()), lr=config.lr
            )

            model.train()
            head.train()
            for epoch in range(1, config.epochs + 1):
                running_loss = 0.0
                pbar = tqdm(dataloader, desc=f"Iter {itr} Epoch {epoch}/{config.epochs}")
                for batch in pbar:
                    wave = batch[0].to(self.device) if not isinstance(batch, dict) else batch.get("audio").to(self.device)
                    with torch.no_grad():
                        feats = label_gen.extract_features(wave)
                        t_np = label_gen.kmeans.predict(feats.cpu().numpy())
                        targets = torch.from_numpy(t_np).long().to(self.device)

                    with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                        context, mask_idx, _ = model(wave)
                        logits = head(context)
                        loss = criterion(logits, targets, mask_idx)

                    optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()

                    running_loss += loss.item()
                    pbar.set_postfix({"loss": loss.item()})

                avg_loss = running_loss / len(dataloader)
                self.logger.info(
                    f"[Iter {itr} Epoch {epoch}] Loss: {avg_loss:.4f}"
                )

                if epoch % self.checkpoint_interval == 0:
                    ckpt_path = os.path.join(
                        hubert_dir,
                        f"iter{itr}_{self.timestamp}_epoch{epoch}.pth",
                    )
                    torch.save(
                        {"model": model.state_dict(), "head": head.state_dict()},
                        ckpt_path,
                    )
                    self.logger.info(f"Model checkpoint saved: {ckpt_path}")

            final_path = os.path.join(
                hubert_dir, f"iter{itr}_{self.timestamp}_epoch{config.epochs}.pth"
            )
            torch.save({"model": model.state_dict(), "head": head.state_dict()}, final_path)
            self.logger.info(f"Final model checkpoint saved: {final_path}")

        self.model = model


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
