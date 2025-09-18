import os
import re
import numpy as np
import torch
import logging
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import Subset, DataLoader, Dataset, RandomSampler

from typing import Optional, Type, Dict, Any
from transformers import BertTokenizer
import optuna

from sklearn.metrics import classification_report
import wandb

from MK_SSL.multimodal.models import *

from MK_SSL.vision.models.modules.losses.nt_xent import NT_Xent

from MK_SSL.multimodal.models.utils.registry import get_method
from MK_SSL.multimodal.models.utils.clap_audio_text_collate import AudioTextCollator
from MK_SSL.multimodal.models.utils.audio_clip_collator import AudioMultimodalCollator



# from MK_SSL.multimodal.models.modules.clap_backbone import CLAPAudioBackbone
# from MK_SSL.multimodal.models.modules.clap_backbone import CLAPTextBackbone
# from MK_SSL.multimodal.models.modules.audio_clip_backbone import AudioCLIPAudioBackbone
# from MK_SSL.multimodal.models.modules.wav2clip_backbone import Wav2CLIPAudioBackbone

from MK_SSL.multimodal.models.modules import CLAPAudioBackbone
from MK_SSL.multimodal.models.modules import CLAPTextBackbone
from MK_SSL.multimodal.models.modules import AudioCLIPAudioBackbone
from MK_SSL.multimodal.models.modules import Wav2CLIPAudioBackbone


# from MK_SSL.utils import EvaluateNet
from MK_SSL.utils import EmbeddingLogger
from MK_SSL.utils import optimize_hyperparameters
from MK_SSL.utils import WandbLogger
from MK_SSL.utils import get_logger_handler


class Trainer:

    def __init__(
        self,
        method: str,
        image_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        audio_encoder: Optional[nn.Module] = None,
        mixed_precision_training: bool = True,
        save_dir: str = ".",
        checkpoint_interval: int = 10,
        reload_checkpoint: bool = False,
        verbose: bool = True,
        # W&B specific arguments
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_mode: str = "online", # "online", "offline", "disabled"
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        wandb_notes: Optional[str] = None,
        wandb_tags: Optional[list[str]] = None,
        use_data_parallel: bool = False,
        audio_clip_text_template: Optional[str] = "{}",
        **kwargs,
    ) -> None:
        """
        Description:
            Initializes the Trainer class for self-supervised training of vision-language models.

        Args:
            method (str): The training method or framework to be used.
                          Options include ["CLIP", "ALBEF", "SimVLM", "SLIP", "UNITER", "VSE"].
            image_encoder (nn.Module): The neural network module responsible for extracting features from images.
            text_encoder (nn.Module): The neural network module responsible for extracting features from text.
            mixed_precision_training (bool, optional): If True, enables mixed precision training to reduce memory usage
                                                       and potentially speed up training. Defaults to True.
            save_dir (str, optional): The directory path where model checkpoints will be saved during training.
                                      Defaults to the current directory ("./").
            checkpoint_interval (int, optional): The number of training epochs between saving model checkpoints.
                                                 Defaults to 10.
            reload_checkpoint (bool, optional): If True, attempts to reload the most recent checkpoint from `save_dir`
                                                at the start of training, allowing continuation from a previous run.
                                                Defaults to False.
            verbose (bool, optional): If True, enables detailed logging and progress information during training.
                                      Defaults to True.
            wandb_project (str, optional): W&B project name. If None, uses default from W&B.
            wandb_entity (str, optional): W&B entity (username or team name). If None, uses default.
            wandb_mode (str, optional): W&B logging mode ("online", "offline", "disabled"). Defaults to "online".
            wandb_run_name (str, optional): Custom name for the W&B run.
            wandb_config (Dict[str, Any], optional): Dictionary of hyperparameters/settings for W&B.
            wandb_notes (str, optional): Notes for the W&B run.
            wandb_tags (list[str], optional): Tags for the W&B run.
            **kwargs: Additional keyword arguments that can be passed to the image and text encoder models,
                      or used to customize the training process.
        """



        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.propagate = False

        if not self.logger.hasHandlers():
            self.logger.addHandler(get_logger_handler())
            
        self.logger.info("Multimodal Trainer initialized.")

        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)


        self.method = method
        self.checkpoint_interval = checkpoint_interval
        self.reload_checkpoint = reload_checkpoint
        self.mixed_precision_training = mixed_precision_training
        
        self.save_dir = os.path.join(save_dir, self.method)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

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
            "---------------- MK_SSL: Multimodal ----------------\n"
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

        # --- Model Args ---
        
        model_args = {
            "image_encoder": image_encoder,
            "text_encoder": text_encoder,
            "audio_encoder": audio_encoder,
            "text_template" : audio_clip_text_template,
            "transformer_encoder": image_encoder,
            "transformer_decoder": text_encoder,
            "device": self.device,
        }

        if "params" in method_cfg:
            model_args.update(method_cfg["default_params"])
        
        if use_data_parallel:
            self.logger.info(f"Wrapping model with DataParallel using {torch.cuda.device_count()} GPUs.")
            self.model = nn.DataParallel(self.model)

        model_args.update(kwargs)

        self.model = method_cfg["model"](**model_args)

        self.logger.info(method_cfg["logs"](self.model))

        self.model = self.model.to(self.device)

        self.logger.info(
            "\n"
            "---------------- Model Parameters ----------------\n"
            f"Total Parameters : {np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}\n"
            "--------------------------------------------------"
        )

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.writer = SummaryWriter("{}/Logs/{}".format(self.save_dir, self.timestamp)) # Commented out: Replaced by WandbLogger
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision_training)

        # --- W&B Logger Initialization ---
        # Combine trainer_config with any specific wandb_config provided
        trainer_internal_config = {
            "method": self.method,
            "save_dir": save_dir,
            "checkpoint_interval": checkpoint_interval,
            "reload_checkpoint": reload_checkpoint,
            "mixed_precision_training": mixed_precision_training,
            "device": str(self.device),
            "num_gpus" : torch.cuda.device_count(),
            "num_workers": self.num_workers,
            **kwargs # Include any other kwargs passed to Trainer init
        }
        full_wandb_config = {**trainer_internal_config, **(wandb_config if wandb_config else {})}

        self.wandb_logger = WandbLogger(
            project_name=wandb_project if wandb_project else f"MK_SSL_Multimodal_{self.method}", # Default project name
            entity=wandb_entity,
            mode=wandb_mode,
            run_name=wandb_run_name,
            config=full_wandb_config,
            notes=wandb_notes if wandb_notes else f"Training {self.method} multimodal model with MK_SSL.",
            tags=wandb_tags if wandb_tags else [self.method, "multimodal", "training"],
        )


        # This logic mirrors the WandbLogger initialization to accurately log the project name.
        effective_wandb_project = wandb_project if wandb_project else f"MK_SSL_Multimodal_{self.method}"

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

    def __del__(self):
        # if hasattr(self, "writer"): # Commented out
        #     self.writer.close() # Commented out
        pass # No need for TensorBoard writer close if not used

    def _train_clip(self, tepoch, optimizer, epoch_idx, total_batches_per_epoch): # Added epoch_idx, total_batches_per_epoch
        epoch_loss = 0.0
        # Watch the model with W&B if active
        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(self.model)

        for step, (batch) in enumerate(tepoch):
            batch = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask", "image"]
            }

            with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                logits = self.model(**batch)
                if self.model.use_siglip:
                    loss = self.model.criterion_siglip_loss(logits)
                else:
                    loss = self.model.criterion_contrastive_loss(logits)

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            # Log batch-level metrics to W&B
            if self.wandb_logger.is_active:
                global_batch_step = (epoch_idx * total_batches_per_epoch) + step
                self.wandb_logger.log({
                    "train/batch_loss": loss.item(),
                    "train/temp": self.model.t_prime.exp().item(),
                    "train/bias": self.model.b.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }, step=global_batch_step)

            tepoch.set_postfix(
                loss=loss.item(),
                temp=self.model.t_prime.exp().item(),
                bias=self.model.b.item(),
                lr=optimizer.param_groups[0]["lr"],
            )
                  

        return epoch_loss

    def _train_slip(self, tepoch, optimizer, epoch_idx, total_batches_per_epoch): # Added epoch_idx, total_batches_per_epoch
        epoch_loss = 0.0
        # Watch the model with W&B if active
        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(self.model)

        for step, (batch) in enumerate(tepoch):
            batch = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask", "image"]
            }

            with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                logits = self.model(**batch)
                ssl_loss = NT_Xent(temperature=0.1)
                ssl_loss = ssl_loss(logits["aug1_embed"], logits["aug2_embed"])
                clip_loss = self.model.clip.criterion_contrastive_loss(
                    logits["clip_output"]
                )

                loss = self.model.criterion(
                    ssl_scale=1.0, ssl_loss=ssl_loss, clip_loss=clip_loss
                )

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            # Log batch-level metrics to W&B
            if self.wandb_logger.is_active:
                global_batch_step = (epoch_idx * total_batches_per_epoch) + step
                self.wandb_logger.log({
                    "train/batch_loss": loss.item(),
                    "train/ssl_loss": ssl_loss.item(),
                    "train/clip_loss": clip_loss.item(),
                    "train/temp": self.model.clip.t_prime.exp().item(),
                    "train/bias": self.model.clip.b.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }, step=global_batch_step)

            tepoch.set_postfix(
                loss=loss.item(),
                temp=self.model.clip.t_prime.exp().item(),
                bias=self.model.clip.b.item(),
                lr=optimizer.param_groups[0]["lr"],
            )

          

        return epoch_loss

    def _train_simvlm(self, tepoch, optimizer, epoch_idx, total_batches_per_epoch): # Added epoch_idx, total_batches_per_epoch
        epoch_loss = 0.0
        # Watch the model with W&B if active
        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(self.model)

        for step, (batch) in enumerate(tepoch):
            batch = {
                k: v.to(self.device) for k, v in batch.items() if k in ["text", "image"]
            }

            with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                logits, labels = self.model(**batch)
                loss = self.model.criterion(logits, labels)

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            # Log batch-level metrics to W&B
            if self.wandb_logger.is_active:
                global_batch_step = (epoch_idx * total_batches_per_epoch) + step
                self.wandb_logger.log({
                    "train/batch_loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }, step=global_batch_step)

            tepoch.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
          

        return epoch_loss

    def _train_vse(self, tepoch, optimizer, epoch_idx, total_batches_per_epoch): # Added epoch_idx, total_batches_per_epoch
        epoch_loss = 0.0
        num_negs = []
        # Watch the model with W&B if active
        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(self.model)

        for step, (batch) in enumerate(tepoch):
            batch = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k in ["image", "image_lengths", "text", "text_lengths"]
            }

            with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                img_emb, txt_emb, txt_lens = self.model(**batch)
                loss, tmp_num_negs = self.model.conterastive_loss(
                    img_emb, txt_emb, txt_lens
                )
                num_negs.extend(tmp_num_negs)

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            clip_grad_norm_(self.model.enc_params, 2.0)
            self.scaler.step(optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            # Log batch-level metrics to W&B
            if self.wandb_logger.is_active:
                global_batch_step = (epoch_idx * total_batches_per_epoch) + step
                self.wandb_logger.log({
                    "train/batch_loss": loss.item(),
                    "train/num_negatives": np.mean(tmp_num_negs), # Log current batch's average negs
                }, step=global_batch_step)

            tepoch.set_postfix(loss=loss.item(), epoch_negs=np.mean(num_negs)) # Keep for TQDM
           

        return epoch_loss

    def _train_albef(self, tepoch, optimizer, epoch, total_batches_per_epoch): # Adjusted epoch param and added total_batches_per_epoch
        epoch_loss = 0.0
        # Watch the model with W&B if active
        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(self.model)

        for step, (batch) in enumerate(tepoch):
            batch = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask", "image"]
            }
            if epoch > 0: # Note: 'epoch' here is the 0-indexed loop variable
                alpha = self.model.alpha
            else:
                alpha = self.model.alpha * min(1, step / len(tepoch))
            with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                loss_mlm, loss_ita, loss_itm = self.model(
                    image=batch["image"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    alpha=alpha,
                )
                loss = loss_mlm + loss_ita + loss_itm

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            # Log batch-level metrics to W&B
            if self.wandb_logger.is_active:
                global_batch_step = (epoch * total_batches_per_epoch) + step
                self.wandb_logger.log({
                    "train/batch_loss": loss.item(),
                    "train/mlm_loss": loss_mlm.item(),
                    "train/ita_loss": loss_ita.item(),
                    "train/itm_loss": loss_itm.item(),
                    "train/alpha": alpha,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }, step=global_batch_step)

            tepoch.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])


        return epoch_loss

    def _train_unitervqa(self, tepoch, optimizer, epoch_idx, total_batches_per_epoch): # Added epoch_idx, total_batches_per_epoch
        epoch_loss = 0.0
        # Watch the model with W&B if active
        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(self.model)

        for step, (batch) in enumerate(tepoch):
            # Assumes batch already on device or handles it internally for UNITER
            with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                logits = self.model(**batch)
                loss = self.model.criterion(batch["targets"], logits)

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            # Log batch-level metrics to W&B
            if self.wandb_logger.is_active:
                global_batch_step = (epoch_idx * total_batches_per_epoch) + step
                self.wandb_logger.log({
                    "train/batch_loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }, step=global_batch_step)


            tepoch.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

                          
        return epoch_loss


    def _train_clap(
        self,
        tepoch,
        optimizer,
        epoch_idx,
        total_batches_per_epoch,
        use_embedding_logger: bool = False,
        logger_loader: Optional[DataLoader] = None,  # NEW
    ):
        """
        Trains the CLAP model for one epoch.

        Args:
            tepoch: tqdm-wrapped DataLoader (or iterable) over batches.
            optimizer: Optimizer instance.
            epoch_idx: Index of the current epoch.
            total_batches_per_epoch: Total number of batches per epoch.
            use_embedding_logger (bool): Whether to enable embedding visualization.
            logger_loader (Optional[DataLoader]): Third dataset with labels for embedding logging.
        """
        epoch_loss = 0.0

        # === Step 0: Log pre-training embeddings ===
        if use_embedding_logger:
            assert logger_loader is not None, "logger_loader must be provided when use_embedding_logger=True"
            embedding_log_dir = os.path.join(self.save_dir, "embedding_logs")
            self.embedding_logger = EmbeddingLogger(
                log_dir=embedding_log_dir,
                method_name=self.method,
                reduce_method="tsne",
                log_interval=1,
            )
            self.logger.info(f"Embedding logger initialized at {embedding_log_dir}")

            self.logger.info("[CLAP - Step 0] Logging pre-training embeddings...")
            backbone = CLAPAudioBackbone(self.model).to(self.device)
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
            self.embedding_logger.log_step(step=0, embeddings=embeddings, labels=labels)
            self.logger.info("[CLAP - Step 0] Pre-training embeddings logged.")
            self.model.train()

        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(self.model)

        # === Training loop ===
        for step, batch in enumerate(tepoch):
            batch = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k in ["audio", "text"]
            }

            with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                audio_embeds, text_embeds, sim_matrix = self.model(
                    audio_input=batch["audio"], text_input=batch["text"]
                )
                loss = self.model.criterion(sim_matrix)

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            epoch_loss += loss.item()

            global_step = (epoch_idx * total_batches_per_epoch) + step

            # === Log to W&B ===
            if self.wandb_logger.is_active:
                self.wandb_logger.log(
                    {
                        "train/batch_loss": loss.item(),
                        "train/temperature": self.model.temperature.exp().item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step
                )

            tepoch.set_postfix(
                loss=loss.item(),
                temp=self.model.temperature.exp().item(),
                lr=optimizer.param_groups[0]["lr"],
            )

        # === Post-epoch embedding logging ===
        if use_embedding_logger:
            self.logger.info(f"[CLAP - Epoch {epoch_idx+1}] Logging embeddings...")
            backbone = CLAPAudioBackbone(self.model).to(self.device)
            backbone.eval()

            all_embeddings, all_labels = [], []
            with torch.no_grad():
                for batch in tqdm(logger_loader, desc=f"EmbeddingLogger Epoch {epoch_idx+1}"):
                    audio = batch["audio"].to(self.device)
                    labels = batch["label"].to(self.device)
                    embeddings = backbone(audio)
                    all_embeddings.append(embeddings)
                    all_labels.append(labels)

            embeddings = torch.cat(all_embeddings, dim=0)
            labels = torch.cat(all_labels, dim=0)
            self.embedding_logger.log_step(step=epoch_idx + 1, embeddings=embeddings, labels=labels)
            self.logger.info(f"[CLAP - Epoch {epoch_idx+1}] Embeddings logged.")
            self.model.train()

        return epoch_loss



        
    def _train_audio_clip(
        self,
        tepoch,
        optimizer,
        epoch_idx,
        total_batches_per_epoch,
        use_embedding_logger: bool = False,
        logger_loader: Optional[DataLoader] = None,  # NEW: third dataloader
    ):
        """
        Trains the AudioCLIP model for one epoch.

        Args:
            tepoch: tqdm-wrapped DataLoader (or iterable).
            optimizer: Optimizer instance.
            epoch_idx: Index of the current epoch.
            total_batches_per_epoch: Total number of steps per epoch.
            use_embedding_logger (bool): Whether to enable embedding visualization.
            logger_loader (Optional[DataLoader]): Dataset for embedding evaluation.
        """
        epoch_loss = 0.0

        # === Step 0: Log pre-training embeddings ===
        if use_embedding_logger:
            assert logger_loader is not None, "logger_loader must be provided when use_embedding_logger=True"
            embedding_log_dir = os.path.join(self.save_dir, "embedding_logs")
            self.embedding_logger = EmbeddingLogger(
                log_dir=embedding_log_dir,
                method_name=self.method,
                reduce_method="tsne",
                log_interval=1,
            )
            self.logger.info(f"Embedding logger initialized at {embedding_log_dir}")

            self.logger.info("[AudioCLIP - Step 0] Logging pre-training embeddings...")
            backbone = AudioCLIPAudioBackbone(self.model).to(self.device)
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
            self.embedding_logger.log_step(step=0, embeddings=embeddings, labels=labels)
            self.logger.info("[AudioCLIP - Step 0] Pre-training embeddings logged.")
            self.model.train()

        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(self.model)

        # === Training loop ===
        for step, batch in enumerate(tepoch):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k in ["audio", "image", "text"]
            }

            with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                (
                    audio_embeds,
                    image_embeds,
                    text_embeds,
                    sim_text_audio,
                    sim_text_image,
                    sim_audio_image,
                ) = self.model(
                    audio_input=batch.get("audio", None),
                    image_input=batch.get("image", None),
                    text_input=batch.get("text", None),
                )

                loss = self.model.criterion(
                    sim_text_audio=sim_text_audio,
                    sim_text_image=sim_text_image,
                    sim_audio_image=sim_audio_image,
                )

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            global_step = (epoch_idx * total_batches_per_epoch) + step

            # === W&B batch logging ===
            if self.wandb_logger.is_active:
                log_data = {
                    "train/batch_loss": loss.item(),
                    "train/temperature": self.model.temperature.exp().item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }

                if sim_text_audio is not None:
                    log_data["train/sim_text_audio_loss"] = sim_text_audio.mean().item()
                if sim_text_image is not None:
                    log_data["train/sim_text_image_loss"] = sim_text_image.mean().item()
                if sim_audio_image is not None:
                    log_data["train/sim_audio_image_loss"] = sim_audio_image.mean().item()

                self.wandb_logger.log(log_data, step=global_step)

            tepoch.set_postfix(
                loss=loss.item(),
                temp=self.model.temperature.exp().item(),
                lr=optimizer.param_groups[0]["lr"],
            )

        # === Post-epoch embedding logging ===
        if use_embedding_logger:
            self.logger.info(f"[AudioCLIP - Epoch {epoch_idx+1}] Logging embeddings...")
            backbone = AudioCLIPAudioBackbone(self.model).to(self.device)
            backbone.eval()

            all_embeddings, all_labels = [], []
            with torch.no_grad():
                for batch in tqdm(logger_loader, desc=f"EmbeddingLogger Epoch {epoch_idx+1}"):
                    audio = batch["audio"].to(self.device)
                    labels = batch["label"].to(self.device)
                    embeddings = backbone(audio)
                    all_embeddings.append(embeddings)
                    all_labels.append(labels)

            embeddings = torch.cat(all_embeddings, dim=0)
            labels = torch.cat(all_labels, dim=0)
            self.embedding_logger.log_step(step=epoch_idx + 1, embeddings=embeddings, labels=labels)
            self.logger.info(f"[AudioCLIP - Epoch {epoch_idx+1}] Embeddings logged.")
            self.model.train()

        return epoch_loss



    def _train_wav2clip(
        self,
        tepoch,
        optimizer,
        epoch_idx,
        total_batches_per_epoch,
        use_embedding_logger: bool = False,
        logger_loader: Optional[DataLoader] = None,  # NEW: for post-epoch embedding eval
    ):
        """
        Training loop for Wav2CLIP (contrastive learning between audio and image).

        Args:
            tepoch: tqdm-wrapped training dataloader
            optimizer: optimizer instance
            epoch_idx: current epoch index
            total_batches_per_epoch: number of batches in the epoch
            use_embedding_logger (bool): whether to use EmbeddingLogger for logging embeddings
            logger_loader (Optional[DataLoader]): dataset used for post-epoch embedding logging
        """
        epoch_loss = 0.0

        # === Step 0: Log pre-training embeddings ===
        if use_embedding_logger:
            assert logger_loader is not None, "logger_loader must be provided when use_embedding_logger=True"
            embedding_log_dir = os.path.join(self.save_dir, "embedding_logs")
            self.embedding_logger = EmbeddingLogger(
                log_dir=embedding_log_dir,
                method_name=self.method,
                reduce_method="tsne",
                log_interval=1,
            )
            self.logger.info(f"Embedding logger initialized at {embedding_log_dir}")

            self.logger.info("[Wav2CLIP - Step 0] Logging pre-training embeddings...")
            backbone = Wav2CLIPAudioBackbone(self.model).to(self.device)
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
            self.embedding_logger.log_step(step=0, embeddings=embeddings, labels=labels)
            self.logger.info("[Wav2CLIP - Step 0] Pre-training embeddings logged.")
            self.model.train()

        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(self.model)

        for step, batch in enumerate(tepoch):
            batch = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k in ["audio", "image"]
            }

            with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                audio_embeds, image_embeds = self.model(
                    audio_waveform=batch["audio"], image_input=batch["image"]
                )
                loss = self.model.criterion(
                    image_embeddings=image_embeds,
                    audio_embeddings=audio_embeds,
                )

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            global_step = (epoch_idx * total_batches_per_epoch) + step

            # === W&B logging ===
            if self.wandb_logger.is_active:
                self.wandb_logger.log(
                    {
                        "train/batch_loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

            tepoch.set_postfix(
                loss=loss.item(),
                lr=optimizer.param_groups[0]["lr"]
            )

        # === Embedding logger ===
        if use_embedding_logger:
            self.logger.info(f"[Wav2CLIP - Epoch {epoch_idx+1}] Logging embeddings...")
            backbone = Wav2CLIPAudioBackbone(self.model).to(self.device)
            backbone.eval()
            all_embeddings, all_labels = [], []
            with torch.no_grad():
                for batch in tqdm(logger_loader, desc=f"EmbeddingLogger Epoch {epoch_idx+1}"):
                    audio = batch["audio"].to(self.device)
                    labels = batch["label"].to(self.device)
                    embeddings = backbone(audio)  
                    all_embeddings.append(embeddings)
                    all_labels.append(labels)
            embeddings = torch.cat(all_embeddings, dim=0)
            labels = torch.cat(all_labels, dim=0)
            self.embedding_logger.log_step(step=epoch_idx + 1, embeddings=embeddings, labels=labels)
            self.logger.info(f"[Wav2CLIP - Epoch {epoch_idx+1}] Embeddings logged.")
            self.model.train()

        return epoch_loss


 

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
        use_embedding_logger: bool = False,
        logger_loader: Optional[DataLoader] = None,  # NEW

        **kwargs,
    ):

        if not hasattr(self, "_optuna_trial"):
            self.wandb_logger.init_run()
        else:
            self.wandb_logger.mode = 'disabled'

        if self.wandb_logger.is_active:

            self.wandb_logger.current_run.config.update({
                "batch_size": batch_size,
                "start_epoch": start_epoch,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "optimizer": optimizer,
                **kwargs,
            })
            self.logger.info(f"W&B run initialized. View run at: {self.wandb_logger.current_run.url}")
        else:
            self.logger.info("W&B logging is not active for this run.")

        number_of_epochs = epochs - start_epoch + 1

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

            kwargs.update({k: v for k, v in best_params.items() if k not in {"lr", "batch_size", "weight_decay", "optimizer"}})
                        
            self.wandb_logger.log({
                "hpo/best_lr": learning_rate,
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
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
            case "adamw":
                optimizer = torch.optim.AdamW(
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
            case _:
                self.logger.error(f"Unsupported Optimizer: {optimizer}")
                
                raise ValueError(f"Optimizer {optimizer} not supported")


        if self.reload_checkpoint:
            start_epoch = self._reload_latest_checkpoint() + 1 # +1 because _reload returns 0-indexed epoch

        # Setting AudioTextCollator for tokenizing texts in clap
        if self.method=="clap" and type(train_dataset[0]['text'][0])==str:
            self.collator= AudioTextCollator()
        elif self.method=="audio_clip" and "audio" in train_dataset[0]:
             self.collator= AudioMultimodalCollator()
        elif self.method=="wav2clip":
             self.collator= AudioMultimodalCollator() 
        else:
            self.collator=None


        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collator ,
            num_workers=self.num_workers,
        )
        first_train_batch = train_dataset[0]
        if "audio" not in first_train_batch or "image" not in first_train_batch or "text" not in first_train_batch:
            self.logger.warning(
                "[Dataset Check] Your dataset should return all 'audio', 'image' and 'text'  keys. "
                "Currently missing: "
                + ", ".join(k for k in ["audio", "image", "text"] if k not in first_train_batch)
            )

        total_batches_per_epoch = len(train_loader) # Used for global step calculation

        self.model.train()

        # Define epoch range
        # The loop iterates from (start_epoch-1) to (epochs-1) inclusive
        # So epoch variable inside loop will be 0-indexed from start_epoch-1
        # For logging, we use (epoch + 1) for 1-indexed epoch display/tracking
        epoch_range_iter = range(start_epoch - 1, epochs)

        match self.method.lower():
            case "clip":
                tmax = number_of_epochs * len(train_loader) + len(train_loader) // 4
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer, T_max=tmax, eta_min=1e-8
                )

                for epoch in tqdm( # epoch is 0-indexed loop variable
                    epoch_range_iter,
                    unit="epoch",
                    desc="CLIP Training",
                    leave=True,
                ):
                    with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}")
                        loss_per_epoch = self._train_clip(tepoch, optimizer, epoch, total_batches_per_epoch,) # Pass epoch_idx, total_batches_per_epoch
                        lr_scheduler.step()

                    # self.writer.add_scalar( # Commented out
                    #     f"{self.method.upper()}/Train/Loss", # Commented out
                    #     loss_per_epoch / len(train_loader), # Commented out
                    #     epoch + 1, # Commented out
                    # ) # Commented out
                    # self.writer.flush() # Commented out

                    # Log epoch-level metrics to W&B
                    epoch_step = (epoch + 1) * len(train_loader)        
                    if self.wandb_logger.is_active:
                        self.wandb_logger.log({
                            f"{self.method.upper()}/Train/epoch_loss": loss_per_epoch / len(train_loader),
                            f"{self.method.upper()}/Train/LR": optimizer.param_groups[0]["lr"],
                            "epoch": epoch + 1
                        }, step=epoch_step)

                    if hasattr(self, "_optuna_trial"):
                        self._optuna_trial.report(loss_per_epoch, epoch)
                        if self._optuna_trial.should_prune():
                            raise optuna.TrialPruned() 

                    if (epoch + 1) % self.checkpoint_interval == 0:
                        model_path = self.save_dir + "/{}_model_{}_epoch{}.pth".format( # Added / for path joining
                            self.method, self.timestamp, epoch + 1
                        ) 

                        torch.save(self.model.state_dict(), model_path)
                        self.logger.info(f"Model checkpoint saved: {model_path}")
                        # Save model checkpoint as W&B artifact
                        if self.wandb_logger.is_active:
                            self.wandb_logger.save_artifact(
                                model_path,
                                name=f"{self.method}-model-epoch-{epoch+1}",
                                type="model",
                                metadata={"epoch": epoch+1, "loss": loss_per_epoch / len(train_loader)}
                            )


            case "slip":
                tmax = number_of_epochs * len(train_loader) + len(train_loader) // 4
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer, T_max=tmax, eta_min=1e-5
                )

                for epoch in tqdm(
                    epoch_range_iter,
                    unit="epoch",
                    desc="SLIP Training",
                    leave=True,
                ):
                    with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}")
                        loss_per_epoch = self._train_slip(tepoch, optimizer, epoch, total_batches_per_epoch,) # Pass epoch_idx, total_batches_per_epoch
                        lr_scheduler.step()

                    # self.writer.add_scalar( # Commented out
                    #     f"{self.method.upper()}/Train/Loss", # Commented out
                    #     loss_per_epoch / len(train_loader), # Commented out
                    #     epoch + 1, # Commented out
                    # ) # Commented out
                    # self.writer.flush() # Commented out

                    # Log epoch-level metrics to W&B
                    epoch_step = (epoch + 1) * len(train_loader)        
                    if self.wandb_logger.is_active:
                        self.wandb_logger.log({
                            f"{self.method.upper()}/Train/epoch_loss": loss_per_epoch / len(train_loader),
                            f"{self.method.upper()}/Train/LR": optimizer.param_groups[0]["lr"],
                            "epoch": epoch + 1
                        }, step=epoch_step)

                    if hasattr(self, "_optuna_trial"):
                        self._optuna_trial.report(loss_per_epoch, epoch)
                        if self._optuna_trial.should_prune():
                            raise optuna.TrialPruned()  

                    if (epoch + 1) % self.checkpoint_interval == 0:
                        model_path = self.save_dir + "/{}_model_{}_epoch{}.pth".format( # Added / for path joining
                            self.method, self.timestamp, epoch + 1
                        )
                        torch.save(self.model.state_dict(), model_path)
                        self.logger.info(f"Model checkpoint saved: {model_path}")
                        # Save model checkpoint as W&B artifact
                        if self.wandb_logger.is_active:
                            self.wandb_logger.save_artifact(
                                model_path,
                                name=f"{self.method}-model-epoch-{epoch+1}",
                                type="model",
                                metadata={"epoch": epoch+1, "loss": loss_per_epoch / len(train_loader)}
                            )


            case "albef":
                tmax = number_of_epochs * len(train_loader) + len(train_loader) // 4
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=tmax, eta_min=1e-5
                )

                for epoch in tqdm(
                    epoch_range_iter,
                    unit="epoch",
                    desc="ALBEF Training",
                    leave=True,
                ):
                    with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}")
                        loss_per_epoch = self._train_albef(tepoch, optimizer, epoch, total_batches_per_epoch,) # Pass epoch_idx, total_batches_per_epoch
                        lr_scheduler.step()

                    # self.writer.add_scalar( # Commented out
                    #     f"{self.method.upper()}/Train/Loss", # Commented out
                    #     loss_per_epoch / len(train_loader), # Commented out
                    #     epoch + 1, # Commented out
                    # ) # Commented out
                    # self.writer.flush() # Commented out

                    # Log epoch-level metrics to W&B
                    epoch_step = (epoch + 1) * len(train_loader)        
                    if self.wandb_logger.is_active:
                        self.wandb_logger.log({
                            f"{self.method.upper()}/Train/epoch_loss": loss_per_epoch / len(train_loader),
                            f"{self.method.upper()}/Train/LR": optimizer.param_groups[0]["lr"],
                            "epoch": epoch + 1
                        }, step=epoch_step)

                    if hasattr(self, "_optuna_trial"):
                        self._optuna_trial.report(loss_per_epoch, epoch)
                        if self._optuna_trial.should_prune():
                            raise optuna.TrialPruned() 

                    if (epoch + 1) % self.checkpoint_interval == 0:
                        model_path = self.save_dir + "/{}_model_{}_epoch{}.pth".format( # Added / for path joining
                            self.method, self.timestamp, epoch + 1
                        )
                        torch.save(self.model.state_dict(), model_path)
                        self.logger.info(f"Model checkpoint saved: {model_path}")
                        # Save model checkpoint as W&B artifact
                        if self.wandb_logger.is_active:
                            self.wandb_logger.save_artifact(
                                model_path,
                                name=f"{self.method}-model-epoch-{epoch+1}",
                                type="model",
                                metadata={"epoch": epoch+1, "loss": loss_per_epoch / len(train_loader)}
                            )


            case "simvlm":
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=2000
                )

                for epoch in tqdm(
                    epoch_range_iter,
                    unit="epoch",
                    desc="SimVLM Training",
                    leave=True,
                ):
                    with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}")
                        loss_per_epoch = self._train_simvlm(tepoch, optimizer, epoch, total_batches_per_epoch,) # Pass epoch_idx, total_batches_per_epoch
                        lr_scheduler.step()

                    # self.writer.add_scalar( # Commented out
                    #     f"{self.method.upper()}/Train/Loss", # Commented out
                    #     loss_per_epoch / len(train_loader), # Commented out
                    #     epoch + 1, # Commented out
                    # ) # Commented out
                    # self.writer.flush() # Commented out

                    # Log epoch-level metrics to W&B
                    epoch_step = (epoch + 1) * len(train_loader)
                    if self.wandb_logger.is_active:
                        self.wandb_logger.log({
                            f"{self.method.upper()}/Train/epoch_loss": loss_per_epoch / len(train_loader),
                            f"{self.method.upper()}/Train/LR": optimizer.param_groups[0]["lr"],
                            "epoch": epoch + 1
                        }, step=epoch_step)

                    if hasattr(self, "_optuna_trial"):
                        self._optuna_trial.report(loss_per_epoch, epoch)
                        if self._optuna_trial.should_prune():
                            raise optuna.TrialPruned()  

                    if (epoch + 1) % self.checkpoint_interval == 0:
                        model_path = self.save_dir + "/{}_model_{}_epoch{}.pth".format( # Added / for path joining
                            self.method, self.timestamp, epoch + 1
                        )
                        torch.save(self.model.state_dict(), model_path)
                        self.logger.info(f"Model checkpoint saved: {model_path}")
                        # Save model checkpoint as W&B artifact
                        if self.wandb_logger.is_active:
                            self.wandb_logger.save_artifact(
                                model_path,
                                name=f"{self.method}-model-epoch-{epoch+1}",
                                type="model",
                                metadata={"epoch": epoch+1, "loss": loss_per_epoch / len(train_loader)}
                            )


            case "uniter_vqa":
                for epoch in tqdm(
                    epoch_range_iter,
                    unit="epoch",
                    desc="Uniter For VQA Training",
                    leave=True,
                ):
                    with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}")
                        loss_per_epoch = self._train_unitervqa(tepoch, optimizer, epoch, total_batches_per_epoch,) # Pass epoch_idx, total_batches_per_epoch

                    # self.writer.add_scalar( # Commented out
                    #     f"{self.method.upper()}/Train/Loss", # Commented out
                    #     loss_per_epoch / len(train_loader), # Commented out
                    #     epoch + 1, # Commented out
                    # ) # Commented out
                    # self.writer.flush() # Commented out

                    # Log epoch-level metrics to W&B
                    epoch_step = (epoch + 1) * len(train_loader)
                    if self.wandb_logger.is_active:
                        self.wandb_logger.log({
                            f"{self.method.upper()}/Train/epoch_loss": loss_per_epoch / len(train_loader),
                            f"{self.method.upper()}/Train/LR": optimizer.param_groups[0]["lr"],
                            "epoch": epoch + 1
                        }, step=epoch_step)

                    if hasattr(self, "_optuna_trial"):
                        self._optuna_trial.report(loss_per_epoch, epoch)
                        if self._optuna_trial.should_prune():
                            raise optuna.TrialPruned() 

                    if (epoch + 1) % self.checkpoint_interval == 0:
                        model_path = self.save_dir + "/{}_model_{}_epoch{}.pth".format( # Added / for path joining
                            self.method, self.timestamp, epoch + 1
                        )
                        torch.save(self.model.state_dict(), model_path)
                        self.logger.info(f"Model checkpoint saved: {model_path}")
                        # Save model checkpoint as W&B artifact
                        if self.wandb_logger.is_active:
                            self.wandb_logger.save_artifact(
                                model_path,
                                name=f"{self.method}-model-epoch-{epoch+1}",
                                type="model",
                                metadata={"epoch": epoch+1, "loss": loss_per_epoch / len(train_loader)}
                            )


            case "vse":
                for epoch in tqdm(
                    epoch_range_iter,
                    unit="epoch",
                    desc="VSE Training",
                    leave=True,
                ):
                    with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}")
                        loss_per_epoch = self._train_vse(tepoch, optimizer, epoch, total_batches_per_epoch,) # Pass epoch_idx, total_batches_per_epoch

                    # self.writer.add_scalar( # Commented out
                    #     f"{self.method.upper()}/Train/Loss", # Commented out
                    #     loss_per_epoch / len(train_loader), # Commented out
                    #     epoch + 1, # Commented out
                    # ) # Commented out
                    # self.writer.flush() # Commented out

                    # Log epoch-level metrics to W&B
                    epoch_step = (epoch + 1) * len(train_loader)
                    if self.wandb_logger.is_active:
                        self.wandb_logger.log({
                            f"{self.method.upper()}/Train/epoch_loss": loss_per_epoch / len(train_loader),
                            f"{self.method.upper()}/Train/AvgNumNegatives": np.mean(self._train_vse_last_num_negs), # Assuming a way to pass this
                            # Note: For _train_vse, `num_negs` is reset per epoch, so this average is for the current epoch's num_negs.
                            # If you need a global average, you'd need to accumulate it.
                            "epoch": epoch + 1,
                        }, step=epoch_step)

                    if hasattr(self, "_optuna_trial"):
                        self._optuna_trial.report(loss_per_epoch, epoch)
                        if self._optuna_trial.should_prune():
                            raise optuna.TrialPruned() 

                    if (epoch + 1) % self.checkpoint_interval == 0:
                        model_path = self.save_dir + "/{}_model_{}_epoch{}.pth".format( # Added / for path joining
                            self.method, self.timestamp, epoch + 1
                        )
                        torch.save(self.model.state_dict(), model_path)
                        self.logger.info(f"Model checkpoint saved: {model_path}")
                        # Save model checkpoint as W&B artifact
                        if self.wandb_logger.is_active:
                            self.wandb_logger.save_artifact(
                                model_path,
                                name=f"{self.method}-model-epoch-{epoch+1}",
                                type="model",
                                metadata={"epoch": epoch+1, "loss": loss_per_epoch / len(train_loader)}
                            )


            case "clap":
                for epoch in tqdm(
                    epoch_range_iter,
                    unit="epoch",
                    desc="CLAP Training",
                    leave=True,
                ):
                    with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}")
                        loss_per_epoch = self._train_clap(tepoch, optimizer, epoch, total_batches_per_epoch, use_embedding_logger=use_embedding_logger, logger_loader=logger_loader) # Pass epoch_idx, total_batches_per_epoch

                    # Log epoch-level metrics to W&B
                    epoch_step = (epoch + 1) * len(train_loader)
                    if self.wandb_logger.is_active:
                        self.wandb_logger.log({
                            f"{self.method.upper()}/Train/epoch_loss": loss_per_epoch / len(train_loader),
                            f"{self.method.upper()}/Train/LR": optimizer.param_groups[0]["lr"],
                            "epoch": epoch + 1
                        }, step=epoch_step)

                    if hasattr(self, "_optuna_trial"):
                        self._optuna_trial.report(loss_per_epoch, epoch)
                        if self._optuna_trial.should_prune():
                            raise optuna.TrialPruned() 

                    if (epoch + 1) % self.checkpoint_interval == 0:
                        model_path = self.save_dir + "/{}_model_{}_epoch{}.pth".format(
                            self.method, self.timestamp, epoch + 1
                        )
                        torch.save(self.model.state_dict(), model_path)
                        self.logger.info(f"Model checkpoint saved: {model_path}")
                        # Save model checkpoint as W&B artifact
                        if self.wandb_logger.is_active:
                            self.wandb_logger.save_artifact(
                                model_path,
                                name=f"{self.method}-model-epoch-{epoch+1}",
                                type="model",
                                metadata={"epoch": epoch+1, "loss": loss_per_epoch / len(train_loader)}
                            )


            case "audio_clip":
                for epoch in tqdm(
                    epoch_range_iter,
                    unit="epoch",
                    desc="Audio-CLIP Training",
                    leave=True,
                ):
                    with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}")
                        loss_per_epoch = self._train_audio_clip(tepoch, optimizer, epoch, total_batches_per_epoch, use_embedding_logger=use_embedding_logger, logger_loader=logger_loader) # Pass epoch_idx, total_batches_per_epoch

                    # Log epoch-level metrics to W&B
                    epoch_step = (epoch + 1) * len(train_loader)
                    if self.wandb_logger.is_active:
                        self.wandb_logger.log({
                            f"{self.method.upper()}/Train/epoch_loss": loss_per_epoch / len(train_loader),
                            f"{self.method.upper()}/Train/LR": optimizer.param_groups[0]["lr"],
                            "epoch": epoch + 1
                        }, step=epoch_step)

                    if hasattr(self, "_optuna_trial"):
                        self._optuna_trial.report(loss_per_epoch, epoch)
                        if self._optuna_trial.should_prune():
                            raise optuna.TrialPruned()         

                    if (epoch + 1) % self.checkpoint_interval == 0:
                        model_path = self.save_dir + "/{}_model_{}_epoch{}.pth".format(
                            self.method, self.timestamp, epoch + 1
                        )
                        torch.save(self.model.state_dict(), model_path)
                        self.logger.info(f"Model checkpoint saved: {model_path}")
                        # Save model checkpoint as W&B artifact
                        if self.wandb_logger.is_active:
                            self.wandb_logger.save_artifact(
                                model_path,
                                name=f"{self.method}-model-epoch-{epoch+1}",
                                type="model",
                                metadata={"epoch": epoch+1, "loss": loss_per_epoch / len(train_loader)}
                            )

            case "wav2clip":
                for epoch in tqdm(
                    epoch_range_iter,
                    unit="epoch",
                    desc="Wav2CLIP Training",
                    leave=True,
                ):
                    with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}")
                        loss_per_epoch = self._train_wav2clip(
                            tepoch, optimizer, epoch, total_batches_per_epoch, use_embedding_logger=use_embedding_logger, logger_loader=logger_loader
                        )

                    epoch_step = (epoch + 1) * len(train_loader)

                    if self.wandb_logger.is_active:
                        self.wandb_logger.log({
                            f"{self.method.upper()}/Train/epoch_loss": loss_per_epoch / len(train_loader),
                            f"{self.method.upper()}/Train/LR": optimizer.param_groups[0]["lr"],
                            "epoch": epoch + 1
                        }, step=epoch_step)

                    if hasattr(self, "_optuna_trial"):
                        self._optuna_trial.report(loss_per_epoch, epoch)
                        if self._optuna_trial.should_prune():
                            raise optuna.TrialPruned()

                    if (epoch + 1) % self.checkpoint_interval == 0:
                        model_path = self.save_dir + "/{}_model_{}_epoch{}.pth".format(
                            self.method, self.timestamp, epoch + 1
                        )
                        torch.save(self.model.state_dict(), model_path)
                        self.logger.info(f"Model checkpoint saved: {model_path}")
                        if self.wandb_logger.is_active:
                            self.wandb_logger.save_artifact(
                                model_path,
                                name=f"{self.method}-model-epoch-{epoch+1}",
                                type="model",
                                metadata={"epoch": epoch+1, "loss": loss_per_epoch / len(train_loader)}
                            )

            case _:
                self.logger.error(f"Unsupported method: {self.method}")
                
                raise ValueError(f"Method {self.method} not supported")

        # Save final model after all epochs
        final_model_path = self.save_dir + "/{}_model_{}_final.pth".format( # Changed to final
            self.method, self.timestamp
        )
        torch.save(self.model.state_dict(), final_model_path)
        self.logger.info(f"Final model saved: {final_model_path}")
        # Save final model as W&B artifact
        if self.wandb_logger.is_active:
            self.wandb_logger.save_artifact(
                final_model_path,
                name=f"{self.method}-model-final",
                type="model",
                metadata={"epochs_trained": epochs, "final_loss": loss_per_epoch / len(train_loader)} # Use final epoch's loss
            )

        # === Final animated embedding plot ===
        if use_embedding_logger:
            self.logger.info("Generating final embedding animation...")
            animation_path = self.embedding_logger.plot_all()
            self.logger.info(f"Embedding animation saved at: {animation_path}")

            if self.wandb_logger.is_active:
                import wandb
                self.wandb_logger.log(
                    {"media/embedding_animation": wandb.Html(animation_path)},
                    step=max(self.embedding_logger.steps) if self.embedding_logger.steps else epochs
                )
                self.logger.info("Embedding animation logged to Weights & Biases.")

        training_mode = "Main" if not hasattr(self, "_optuna_trial") else "HPO"
        if self.wandb_logger.is_active:
            self.wandb_logger.finish_run()
            self.logger.info(f"{training_mode} training process completed and W&B run finalized.")
        else:
            self.logger.info(f"{training_mode} training process completed.")




    def load_checkpoint(self, checkpont_dir: str):
        self.model.load_state_dict(torch.load(checkpont_dir, map_location=self.device)) # Add map_location


        self.logger.info(
            "\n"
            "---------------- Checkpoint ----------------\n"
            "Checkpoint loaded.\n"
            "--------------------------------------------"
        )


    def _reload_latest_checkpoint(self):
        checkpoints = os.listdir(self.save_dir)
        method_prefix = self.method + "_model_" # Filter by method
        filtered_checkpoints = [
            ckpt
            for ckpt in checkpoints
            if ckpt.endswith(".pth") and ckpt.startswith(method_prefix)
        ]

        sorted_checkpoints = sorted(
            [os.path.join(self.save_dir, i) for i in filtered_checkpoints], # Use filtered checkpoints
            key=os.path.getmtime,
        )

        if len(sorted_checkpoints) == 0:
            self.logger.warning(f"No checkpoints found for method '{self.method}' in {self.save_dir}. Starting from scratch.")
            return 0 # Return 0 for 0-indexed epoch if no checkpoint found

        self.load_checkpoint(sorted_checkpoints[-1])

        match = re.search(r"epoch(\d+)", sorted_checkpoints[-1])
        if match:
            epoch = int(match.group(1)) -1 # Return 0-indexed epoch
            self.logger.info(
                "\n"
                "---------------- Checkpoint Reload ----------------\n"
                f"Starting Epoch : {epoch + 1}\n" # Log 1-indexed epoch for user
                "---------------------------------------------------"
            )

        else:
            self.logger.warning("No epoch number found in the checkpoint name. Resuming from epoch 0.")
            epoch = 0 # Default to epoch 0 if not found

        return epoch # Return 0-indexed epoch



   